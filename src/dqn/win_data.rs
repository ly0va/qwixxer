//! Stored datasets for the win net: turn ends and the candidates of the
//! decisions win-search would search, labelled with full-game rollouts
//! (win and auxiliary labels), split into training and held-out games.
//!
//! Design: docs/superpowers/specs/2026-10-19-win-thermo-aux-design.md

use crate::dqn::win_distill::{sample_var, CandDecision};
use crate::dqn::win_train::TurnEnds;
use crate::game::{Game, Player};
use crate::state::State;
use crate::strategy::search::splitmix64;
use crate::strategy::search::{phase2_entry, rollout_final_states, GATE_MARGIN};
use crate::strategy::sim::SimGame;
use crate::strategy::Bot;
use burn::prelude::*;
use rand::{rngs::SmallRng, SeedableRng};
use rayon::prelude::*;
use std::cell::RefCell;
use std::fs::File;
use std::fs::OpenOptions;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::rc::Rc;

/// One labelled position in the turn-end frame (the other player to move
/// next). Every label is a mean over the position's rollouts.
#[derive(Clone, Copy, Debug)]
pub struct Sample {
    pub me: State,
    pub opp: State,
    /// Mean outcome for me (win 1, tie ½, loss 0).
    pub win: f32,
    /// Sample variance of the outcomes (0 for an ended position).
    pub win_var: f32,
    /// Final row totals / 12: mine red, yellow, green, blue, then the opponent's.
    pub marks: [f32; 8],
    /// Final strikes / 4: mine, the opponent's.
    pub strikes: [f32; 2],
    /// Final (my − opponent's) points / 100.
    pub diff: f32,
}

/// Exact equality: the boards byte for byte (`State`'s own `==` is a
/// dominance test) and every label.
impl PartialEq for Sample {
    fn eq(&self, o: &Self) -> bool {
        self.me.to_bytes() == o.me.to_bytes() && self.opp.to_bytes() == o.opp.to_bytes() && self.floats() == o.floats()
    }
}

impl Sample {
    /// Labels from the rollouts' final boards (`finals`: one (me, opp) per
    /// rollout; a single element for an ended position, its exact result).
    pub fn from_finals(me: State, opp: State, finals: &[Vec<State>]) -> Sample {
        let n = finals.len() as f32;
        let outcomes: Vec<f32> = finals.iter().map(|f| SimGame::outcome(f)).collect();
        let (mut marks, mut strikes, mut diff) = ([0.0f32; 8], [0.0f32; 2], 0.0f32);
        for f in finals {
            for side in 0..2 {
                for (r, &t) in f[side].row_totals().iter().enumerate() {
                    marks[4 * side + r] += t as f32 / 12.0 / n;
                }
                strikes[side] += f[side].strikes as f32 / 4.0 / n;
            }
            diff += (f[0].count_points() - f[1].count_points()) as f32 / 100.0 / n;
        }
        Sample {
            me,
            opp,
            win: outcomes.iter().sum::<f32>() / n,
            win_var: sample_var(&outcomes),
            marks,
            strikes,
            diff,
        }
    }

    /// The same sample from the other player's side.
    pub fn swapped(&self) -> Sample {
        let mut marks = [0.0f32; 8];
        marks[..4].copy_from_slice(&self.marks[4..]);
        marks[4..].copy_from_slice(&self.marks[..4]);
        Sample {
            me: self.opp,
            opp: self.me,
            win: 1.0 - self.win,
            win_var: self.win_var,
            marks,
            strikes: [self.strikes[1], self.strikes[0]],
            diff: -self.diff,
        }
    }

    /// The labels as stored: win, win_var, marks, strikes, diff.
    pub fn floats(&self) -> [f32; 13] {
        let mut f = [0.0f32; 13];
        f[0] = self.win;
        f[1] = self.win_var;
        f[2..10].copy_from_slice(&self.marks);
        f[10..12].copy_from_slice(&self.strikes);
        f[12] = self.diff;
        f
    }

    /// The inverse of [`Sample::floats`].
    pub fn from_floats(me: State, opp: State, f: [f32; 13]) -> Sample {
        let mut marks = [0.0f32; 8];
        marks.copy_from_slice(&f[2..10]);
        Sample {
            me,
            opp,
            win: f[0],
            win_var: f[1],
            marks,
            strikes: [f[10], f[11]],
            diff: f[12],
        }
    }
}

/// One labelled decision: a sample per kept candidate (best static value
/// first) and the CRN spread of each runner-up's difference to the first,
/// as [`crate::dqn::win_distill::CandGroup`] holds them.
#[derive(Clone, Debug, PartialEq)]
pub struct DecisionData {
    pub samples: Vec<Sample>,
    /// `pair_var[i − 1]`: sample variance of outcome(i) − outcome(0).
    pub pair_var: Vec<f32>,
    /// The entries had me to move next, so the samples are stored from the
    /// opponent's side ([`Sample::swapped`]).
    pub swapped: bool,
}

/// [`crate::dqn::win_distill::label_decision`] with every label: `k`
/// full-game CRN rollouts by `bot` per kept candidate.
pub fn label_decision_full(bot: &impl Bot, d: &CandDecision, k: usize) -> DecisionData {
    let finals = rollout_final_states(bot, &d.entries, d.seed, k, d.first_active, false);
    let swapped = d.first_active == 0;
    let samples = d
        .entries
        .iter()
        .zip(&finals)
        .map(|((e, _), f)| {
            assert_eq!(e.len(), 2, "the win net is 2-player only");
            let s = Sample::from_finals(e[0], e[1], f);
            if swapped {
                s.swapped()
            } else {
                s
            }
        })
        .collect();
    // An ended entry's exact result stands for every rollout.
    let outcomes: Vec<Vec<f32>> = finals
        .iter()
        .map(|f| {
            let o: Vec<f32> = f.iter().map(|states| SimGame::outcome(states)).collect();
            if o.len() == 1 {
                vec![o[0]; k]
            } else {
                o
            }
        })
        .collect();
    let pair_var = outcomes[1..]
        .iter()
        .map(|o| {
            let diffs: Vec<f32> = o.iter().zip(&outcomes[0]).map(|(a, b)| a - b).collect();
            sample_var(&diffs)
        })
        .collect();
    DecisionData {
        samples,
        pair_var,
        swapped,
    }
}

/// One side of the split: turn ends and labelled decisions, in game order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Split {
    pub turns: Vec<Sample>,
    pub decisions: Vec<DecisionData>,
}

impl Split {
    /// Candidate samples over all decisions.
    pub fn cand_samples(&self) -> usize {
        self.decisions.iter().map(|d| d.samples.len()).sum()
    }
}

/// `meta.json`: how the dataset was made, and its sizes so far. It is
/// rewritten after every batch, so it always describes the last complete one.
#[derive(Config, Debug)]
pub struct DataMeta {
    pub player: String,
    pub seed: u64,
    /// Games kept: games 0 to `games − 1`, in order.
    pub games: usize,
    pub rollouts: usize,
    pub cand_rollouts: usize,
    pub holdout: f64,
    /// Share of turn ends and decisions kept (and labelled). `None` in
    /// datasets written before it existed, which kept everything.
    pub keep: Option<f64>,
    pub train_turns: usize,
    pub train_decisions: usize,
    pub heldout_turns: usize,
    pub heldout_decisions: usize,
    /// `data.bin`'s length up to the last complete batch; bytes after it
    /// are a half-written batch from a stopped run.
    pub bytes: u64,
}

#[derive(Debug)]
pub struct Dataset {
    pub meta: DataMeta,
    pub train: Split,
    pub heldout: Split,
}

/// `win-data` settings.
#[derive(Debug)]
pub struct DataCfg {
    /// Win model dir that plays the games and the rollouts (recorded in the
    /// metadata; [`generate`] and [`write_dataset`] take the bot itself).
    pub player: String,
    pub out: String,
    /// Turn ends to reach in all; whole games are kept, so the count is reached or just passed.
    pub positions: usize,
    /// Rollouts per turn end.
    pub rollouts: usize,
    /// Rollouts per candidate.
    pub cand_rollouts: usize,
    /// Share of games held out (1 = all, as a benchmark).
    pub holdout: f64,
    /// Share of each game's turn ends and decisions kept, drawn per item
    /// before labelling.
    pub keep: f64,
    pub seed: u64,
    /// Continue the dataset already in `out`.
    pub resume: bool,
}

impl DataCfg {
    /// Every check that can fail, before any work.
    pub fn validate(&self) -> Result<(), String> {
        if self.positions == 0 {
            return Err("--positions must be at least 1".into());
        }
        if self.rollouts < 2 {
            return Err("--rollouts must be at least 2 (the noise floor needs a variance)".into());
        }
        if self.cand_rollouts < 2 {
            return Err("--cand-rollouts must be at least 2".into());
        }
        if !(0.0..=1.0).contains(&self.holdout) {
            return Err("--holdout must be in [0, 1]".into());
        }
        if !(self.keep > 0.0 && self.keep <= 1.0) {
            return Err("--keep must be in (0, 1]".into());
        }
        for f in ["config.json", "model.mpk"] {
            if !Path::new(&format!("{}/{f}", self.player)).exists() {
                return Err(format!("--player {}: {f} not found", self.player));
            }
        }
        if self.resume {
            for f in ["meta.json", "data.bin"] {
                if !Path::new(&format!("{}/{f}", self.out)).exists() {
                    return Err(format!("--resume {}: {f} not found, nothing to resume", self.out));
                }
            }
            let m = read_meta(&self.out).map_err(|e| format!("--resume {}: {e}", self.out))?;
            differs("--player", &self.player, &m.player)?;
            differs("--rollouts", &self.rollouts, &m.rollouts)?;
            differs("--cand-rollouts", &self.cand_rollouts, &m.cand_rollouts)?;
            differs("--holdout", &self.holdout, &m.holdout)?;
            differs("--keep", &self.keep, &m.keep.unwrap_or(1.0))?;
            differs("--seed", &self.seed, &m.seed)?;
        } else if Path::new(&self.out).exists() {
            return Err(format!(
                "{} already exists: move it aside, pick another --out, or add --resume",
                self.out
            ));
        }
        Ok(())
    }
}

fn differs<T: PartialEq + std::fmt::Debug>(flag: &str, given: &T, have: &T) -> Result<(), String> {
    if given == have {
        Ok(())
    } else {
        Err(format!("{flag} {given:?} differs from the dataset's {have:?}"))
    }
}

/// `--resume` settings for the dataset in `out`: every setting not given
/// comes from its `meta.json`; one that is given must match it.
pub fn resume_cfg(
    out: &str,
    positions: usize,
    player: Option<String>,
    rollouts: Option<usize>,
    cand_rollouts: Option<usize>,
    holdout: Option<f64>,
    keep: Option<f64>,
    seed: Option<u64>,
) -> Result<DataCfg, String> {
    let m = read_meta(out).map_err(|e| format!("--resume {out}: {e}"))?;
    let pick = |flag: &str, given: Option<String>, have: &String| -> Result<String, String> {
        given.map_or(Ok(have.clone()), |g| differs(flag, &g, have).map(|_| g))
    };
    fn take<T: PartialEq + std::fmt::Debug + Copy>(flag: &str, given: Option<T>, have: T) -> Result<T, String> {
        given.map_or(Ok(have), |g| differs(flag, &g, &have).map(|_| g))
    }
    Ok(DataCfg {
        player: pick("--player", player, &m.player)?,
        out: out.into(),
        positions,
        rollouts: take("--rollouts", rollouts, m.rollouts)?,
        cand_rollouts: take("--cand-rollouts", cand_rollouts, m.cand_rollouts)?,
        holdout: take("--holdout", holdout, m.holdout)?,
        keep: take("--keep", keep, m.keep.unwrap_or(1.0))?,
        seed: take("--seed", seed, m.seed)?,
        resume: true,
    })
}

/// Whether game `game` of a dataset seeded `seed` is held out; about
/// `share` of the games are.
pub fn held_out(seed: u64, game: u64, share: f64) -> bool {
    let h = splitmix64(seed ^ splitmix64(game ^ 0x4E1D_0001));
    ((h >> 11) as f64 / (1u64 << 53) as f64) < share
}

/// Whether item `item` of game `game` is kept, at share `p` (always at 1).
fn kept(seed: u64, game: u64, item: u64, p: f64) -> bool {
    if p >= 1.0 {
        return true;
    }
    let h = splitmix64(seed ^ splitmix64(game ^ 0x6B33_0002) ^ splitmix64(item));
    ((h >> 11) as f64 / (1u64 << 53) as f64) < p
}

/// [`kept`]'s item for a game's `i`-th turn end.
fn turn_item(i: u64) -> u64 {
    i
}

/// [`kept`]'s item for a game's `j`-th decision.
fn decision_item(j: u64) -> u64 {
    j | 1 << 32
}

/// One self-play game's labelled turn ends and candidate decisions. Game
/// `g` is seeded `seed + 1000·g` and its turn ends are rolled out from seed
/// `game seed ^ 0x5011_0075`, as in `rollout_round_by`, so the win labels
/// are the ones `win-pi --labels cand` would give. Only the [`kept`] turn
/// ends and decisions are labelled; the turn ends' rollouts share their
/// seeds, so a kept item's labels do not depend on `cfg.keep`.
fn game_samples<P: Bot + Clone + 'static>(player: &P, cfg: &DataCfg, g: u64) -> (Vec<Sample>, Vec<DecisionData>) {
    let game_seed = cfg.seed.wrapping_add(g.wrapping_mul(1000));
    let bufs: [Rc<RefCell<Vec<(State, State)>>>; 2] = Default::default();
    let caps: [Rc<RefCell<Vec<CandDecision>>>; 2] = Default::default();
    let players: Vec<Player> = (0..2)
        .map(|i| {
            let rec = TurnEnds {
                bot: player.clone(),
                recorded: bufs[i].clone(),
                capture: Some((GATE_MARGIN, caps[i].clone())),
            };
            Player::new(
                Box::new(rec),
                Box::new(SmallRng::seed_from_u64(game_seed.wrapping_add(i as u64))),
            )
        })
        .collect();
    Game::new(players).play();
    // The lock-propagated boards, as the turn-end inputs use them.
    let entries: Vec<(Vec<State>, bool)> = bufs
        .iter()
        .flat_map(|b| b.take())
        .enumerate()
        .filter(|&(i, _)| kept(cfg.seed, g, turn_item(i as u64), cfg.keep))
        .map(|(_, (me, opp))| phase2_entry(me, &[opp]))
        .collect();
    let turns = if entries.is_empty() {
        Vec::new()
    } else {
        rollout_final_states(player, &entries, game_seed ^ 0x5011_0075, cfg.rollouts, 1, false)
            .iter()
            .zip(&entries)
            .map(|(f, (b, _))| Sample::from_finals(b[0], b[1], f))
            .collect()
    };
    let decisions = caps
        .iter()
        .flat_map(|c| c.take())
        .enumerate()
        .filter(|&(j, _)| kept(cfg.seed, g, decision_item(j as u64), cfg.keep))
        .map(|(_, d)| label_decision_full(player, &d, cfg.cand_rollouts))
        .collect();
    (turns, decisions)
}

/// Games per batch at most; progress is saved after every batch.
fn batch_cap() -> u64 {
    (8 * rayon::current_num_threads()).max(256) as u64
}

/// One kept game: whether it is held out, its turn ends and its decisions.
type GameData = (bool, Vec<Sample>, Vec<DecisionData>);

/// Plays and labels whole games from index `game` on, in index order,
/// until the turn ends (`turns` so far) reach `cfg.positions`, and hands
/// each batch's kept games to `on_batch` with the running turn-end count.
/// The games kept depend only on `game`, `turns` and the target, never on
/// how they are batched.
fn run_games<P: Bot + Clone + 'static, E>(
    cfg: &DataCfg,
    player: &(dyn Fn() -> P + Sync),
    mut game: u64,
    mut turns: usize,
    mut on_batch: impl FnMut(Vec<GameData>, usize) -> Result<(), E>,
) -> Result<(), E> {
    let (start, mut games) = (turns, 0usize);
    while turns < cfg.positions {
        let per_game = if games == 0 {
            25.0 * cfg.keep
        } else {
            ((turns - start) as f64 / games as f64).max(0.1)
        };
        let batch = (((cfg.positions - turns) as f64 / per_game).ceil() as u64).clamp(1, batch_cap());
        let chunk: Vec<(Vec<Sample>, Vec<DecisionData>)> = (game..game + batch)
            .into_par_iter()
            .map_init(player, |player, g| game_samples(&*player, cfg, g))
            .collect();
        let mut out = Vec::new();
        for (g, (t, d)) in (game..).zip(chunk) {
            if turns >= cfg.positions {
                break;
            }
            turns += t.len();
            out.push((held_out(cfg.seed, g, cfg.holdout), t, d));
        }
        games += out.len();
        game += out.len() as u64;
        on_batch(out, turns)?;
    }
    Ok(())
}

impl DataMeta {
    fn fresh(cfg: &DataCfg) -> DataMeta {
        DataMeta {
            player: cfg.player.clone(),
            seed: cfg.seed,
            games: 0,
            rollouts: cfg.rollouts,
            cand_rollouts: cfg.cand_rollouts,
            holdout: cfg.holdout,
            keep: Some(cfg.keep),
            train_turns: 0,
            train_decisions: 0,
            heldout_turns: 0,
            heldout_decisions: 0,
            bytes: 0,
        }
    }

    /// Count one kept game.
    fn add(&mut self, heldout: bool, turns: usize, decisions: usize) {
        self.games += 1;
        if heldout {
            self.heldout_turns += turns;
            self.heldout_decisions += decisions;
        } else {
            self.train_turns += turns;
            self.train_decisions += decisions;
        }
    }
}

/// The dataset of `player`'s self-play (one bot per worker thread), in
/// memory: whole games in index order until `cfg.positions` turn ends,
/// each game's turn ends and decisions on one side of the split.
pub fn generate<P: Bot + Clone + 'static>(cfg: &DataCfg, player: &(dyn Fn() -> P + Sync)) -> Dataset {
    let (mut train, mut heldout, mut meta) = (Split::default(), Split::default(), DataMeta::fresh(cfg));
    let done: Result<(), std::convert::Infallible> = run_games(cfg, player, 0, 0, |batch, _| {
        for (h, t, d) in batch {
            meta.add(h, t.len(), d.len());
            let side = if h { &mut heldout } else { &mut train };
            side.turns.extend(t);
            side.decisions.extend(d);
        }
        Ok(())
    });
    let Ok(()) = done;
    Dataset { meta, train, heldout }
}

fn hms(secs: u64) -> String {
    format!("{}h{:02}m{:02}s", secs / 3600, secs / 60 % 60, secs % 60)
}

/// `win-data`: the dataset [`generate`] would make, written to `cfg.out`
/// as it goes: after every batch its games are appended to `data.bin` and
/// `meta.json` is rewritten, so a stopped run keeps every complete batch.
/// With `cfg.resume`, cuts off a half-written batch and continues from the
/// next game. A dataset made in parts is byte for byte the one made at once.
pub fn write_dataset<P: Bot + Clone + 'static>(cfg: &DataCfg, player: &(dyn Fn() -> P + Sync)) -> io::Result<DataMeta> {
    let dir = &cfg.out;
    let path = format!("{dir}/data.bin");
    let mut meta = if cfg.resume {
        let meta = read_meta(dir)?;
        let f = OpenOptions::new().write(true).open(&path)?;
        let len = f.metadata()?.len();
        if len < meta.bytes {
            return Err(shorter(&path, len, meta.bytes));
        }
        f.set_len(meta.bytes)?;
        f.sync_all()?;
        meta
    } else {
        std::fs::create_dir_all(dir)?;
        let mut f = File::create(&path)?;
        f.write_all(MAGIC)?;
        f.write_all(&VERSION.to_le_bytes())?;
        f.sync_all()?;
        let meta = DataMeta {
            bytes: 8,
            ..DataMeta::fresh(cfg)
        };
        write_meta(dir, &meta)?;
        meta
    };
    let mut file = OpenOptions::new().append(true).open(&path)?;
    let (start, first) = (std::time::Instant::now(), meta.train_turns + meta.heldout_turns);
    run_games(cfg, player, meta.games as u64, first, |batch, turns| {
        let mut w = BufWriter::new(&mut file);
        for (h, t, d) in &batch {
            put_record(&mut w, *h, t, d)?;
            meta.add(*h, t.len(), d.len());
        }
        w.flush()?;
        drop(w);
        file.sync_data()?;
        meta.bytes = file.metadata()?.len();
        write_meta(dir, &meta)?;
        let secs = start.elapsed().as_secs();
        let eta = (secs as f64 / (turns - first).max(1) as f64 * cfg.positions.saturating_sub(turns) as f64) as u64;
        println!(
            "win-data: {} games, {turns} / {} turn ends, {}, ETA {}",
            meta.games,
            cfg.positions,
            hms(secs),
            hms(eta)
        );
        Ok::<(), io::Error>(())
    })?;
    Ok(meta)
}

const MAGIC: &[u8; 4] = b"QWXD";
const VERSION: u32 = 2;

fn invalid(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

fn shorter(path: &str, len: u64, want: u64) -> io::Error {
    invalid(format!(
        "{path} is shorter ({len} bytes) than meta.json says ({want} bytes)"
    ))
}

fn read_meta(dir: &str) -> io::Result<DataMeta> {
    DataMeta::load(format!("{dir}/meta.json")).map_err(|e| invalid(format!("{dir}/meta.json: {e:?}")))
}

/// Write `meta.json` whole or not at all: a temporary file, then a rename.
fn write_meta(dir: &str, meta: &DataMeta) -> io::Result<()> {
    let tmp = format!("{dir}/meta.json.tmp");
    meta.save(&tmp)?;
    std::fs::rename(&tmp, format!("{dir}/meta.json"))
}

fn put_sample(w: &mut impl Write, s: &Sample) -> io::Result<()> {
    w.write_all(&s.me.to_bytes())?;
    w.write_all(&s.opp.to_bytes())?;
    for v in s.floats() {
        w.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

fn get_f32(r: &mut impl Read) -> io::Result<f32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(f32::from_le_bytes(b))
}

fn get_u32(r: &mut impl Read) -> io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn get_sample(r: &mut impl Read) -> io::Result<Sample> {
    let (mut me, mut opp) = ([0u8; 13], [0u8; 13]);
    r.read_exact(&mut me)?;
    r.read_exact(&mut opp)?;
    let mut f = [0.0f32; 13];
    for v in f.iter_mut() {
        *v = get_f32(r)?;
    }
    Ok(Sample::from_floats(State::from_bytes(me), State::from_bytes(opp), f))
}

/// One record: a side flag (1 = held out), then turn ends and decisions,
/// each counted. [`write_dataset`] writes one per game.
fn put_record(w: &mut impl Write, heldout: bool, turns: &[Sample], decisions: &[DecisionData]) -> io::Result<()> {
    w.write_all(&[heldout as u8])?;
    w.write_all(&(turns.len() as u32).to_le_bytes())?;
    for t in turns {
        put_sample(w, t)?;
    }
    w.write_all(&(decisions.len() as u32).to_le_bytes())?;
    for d in decisions {
        w.write_all(&[d.swapped as u8, d.samples.len() as u8])?;
        for t in &d.samples {
            put_sample(w, t)?;
        }
        for v in &d.pair_var {
            w.write_all(&v.to_le_bytes())?;
        }
    }
    Ok(())
}

/// Counts come from the file, so nothing is preallocated from them: a
/// damaged count runs into the end of the data instead.
fn get_record(r: &mut impl Read) -> io::Result<(bool, Vec<Sample>, Vec<DecisionData>)> {
    let mut flag = [0u8; 1];
    r.read_exact(&mut flag)?;
    let mut turns = Vec::new();
    for _ in 0..get_u32(r)? {
        turns.push(get_sample(r)?);
    }
    let mut decisions = Vec::new();
    for _ in 0..get_u32(r)? {
        let mut h = [0u8; 2];
        r.read_exact(&mut h)?;
        let samples = (0..h[1]).map(|_| get_sample(r)).collect::<io::Result<Vec<_>>>()?;
        let pair_var = (1..h[1]).map(|_| get_f32(r)).collect::<io::Result<Vec<_>>>()?;
        decisions.push(DecisionData {
            samples,
            pair_var,
            swapped: h[0] != 0,
        });
    }
    Ok((flag[0] != 0, turns, decisions))
}

/// Write an in-memory dataset to `dir`: `meta.json` (its counts taken from
/// the splits) and `data.bin` with one record per side.
pub fn save(dir: &str, ds: &Dataset) -> io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let path = format!("{dir}/data.bin");
    let mut w = BufWriter::new(File::create(&path)?);
    w.write_all(MAGIC)?;
    w.write_all(&VERSION.to_le_bytes())?;
    put_record(&mut w, false, &ds.train.turns, &ds.train.decisions)?;
    put_record(&mut w, true, &ds.heldout.turns, &ds.heldout.decisions)?;
    w.flush()?;
    drop(w);
    let meta = DataMeta {
        train_turns: ds.train.turns.len(),
        train_decisions: ds.train.decisions.len(),
        heldout_turns: ds.heldout.turns.len(),
        heldout_decisions: ds.heldout.decisions.len(),
        bytes: std::fs::metadata(&path)?.len(),
        ..ds.meta.clone()
    };
    write_meta(dir, &meta)
}

/// Read a dataset written by [`write_dataset`] or [`save`]: the records up
/// to `meta.json`'s length (a half-written batch after it is ignored).
pub fn load(dir: &str) -> io::Result<Dataset> {
    let meta = read_meta(dir)?;
    let path = format!("{dir}/data.bin");
    let file = File::open(&path)?;
    let len = file.metadata()?.len();
    if len < meta.bytes {
        return Err(shorter(&path, len, meta.bytes));
    }
    let mut r = BufReader::new(file).take(meta.bytes);
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(invalid(format!("{path} is not a win-data file")));
    }
    let version = get_u32(&mut r)?;
    if version != VERSION {
        return Err(invalid(format!("{path}: unsupported win-data version {version}")));
    }
    let (mut train, mut heldout) = (Split::default(), Split::default());
    while r.limit() > 0 {
        let (h, t, d) = get_record(&mut r)?;
        let side = if h { &mut heldout } else { &mut train };
        side.turns.extend(t);
        side.decisions.extend(d);
    }
    let counts = [
        train.turns.len(),
        train.decisions.len(),
        heldout.turns.len(),
        heldout.decisions.len(),
    ];
    let want = [
        meta.train_turns,
        meta.train_decisions,
        meta.heldout_turns,
        meta.heldout_decisions,
    ];
    if counts != want {
        return Err(invalid(format!(
            "{dir}: data.bin holds {counts:?} (train turn ends, decisions, held-out turn ends, decisions), meta.json says {want:?}"
        )));
    }
    Ok(Dataset { meta, train, heldout })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dqn::pair::PairStrategy;
    use crate::dqn::win::win_input;
    use crate::dqn::win_distill::label_decision;
    use crate::dqn::win_train::{rollout_round_by, CandCfg};
    use crate::state::Mark;
    use crate::strategy::search::GATE_MARGIN;
    use crate::strategy::search::{phase2_entry, rollout_final_states};

    #[test]
    fn labels_of_an_ended_position_are_exact() {
        let mut done = State::default();
        for _ in 0..4 {
            done.apply_strike();
        }
        let mut opp = State::default();
        opp.apply_mark(Mark { row: 0, number: 2 });
        let s = Sample::from_finals(done, opp, &[vec![done, opp]]);
        assert_eq!((s.win, s.win_var), (0.0, 0.0));
        assert_eq!(s.marks, [0.0, 0.0, 0.0, 0.0, 1.0 / 12.0, 0.0, 0.0, 0.0]);
        assert_eq!(s.strikes, [1.0, 0.0]);
        assert_eq!(s.diff, -21.0 / 100.0);
    }

    #[test]
    fn labels_average_the_rollouts() {
        let mut a = State::default();
        a.apply_mark(Mark { row: 0, number: 2 });
        a.apply_mark(Mark { row: 0, number: 3 });
        let mut b = State::default();
        b.apply_strike();
        // a: 3 points, b: −5; once each way round.
        let s = Sample::from_finals(State::default(), State::default(), &[vec![a, b], vec![b, a]]);
        assert_eq!((s.win, s.win_var), (0.5, 0.5));
        assert!((s.marks[0] - 1.0 / 12.0).abs() < 1e-7 && (s.marks[4] - 1.0 / 12.0).abs() < 1e-7);
        assert!((s.strikes[0] - 0.125).abs() < 1e-7 && (s.strikes[1] - 0.125).abs() < 1e-7);
        assert!(s.diff.abs() < 1e-7);
    }

    #[test]
    fn a_swapped_sample_is_the_other_players_view() {
        let mut a = State::default();
        a.apply_mark(Mark { row: 1, number: 4 });
        let b = State::default();
        let s = Sample {
            me: a,
            opp: b,
            win: 0.75,
            win_var: 0.2,
            marks: [1., 2., 3., 4., 5., 6., 7., 8.],
            strikes: [0.25, 0.5],
            diff: 0.1,
        };
        let t = s.swapped();
        assert_eq!((t.me.to_bytes(), t.opp.to_bytes()), (b.to_bytes(), a.to_bytes()));
        assert_eq!((t.win, t.win_var, t.diff), (0.25, 0.2, -0.1));
        assert_eq!(t.marks, [5., 6., 7., 8., 1., 2., 3., 4.]);
        assert_eq!(t.strikes, [0.5, 0.25]);
        assert_eq!(t.swapped(), s);
        assert_eq!(Sample::from_floats(s.me, s.opp, s.floats()), s);
    }

    #[test]
    fn full_labels_match_the_win_only_labels() {
        let bot = PairStrategy::load("pair_model");
        let opp = State::default();
        let mut marked = State::default();
        marked.apply_mark(Mark { row: 0, number: 2 });
        let mut done = State::default();
        for _ in 0..4 {
            done.apply_strike();
        }
        let live = vec![
            phase2_entry(State::default(), &[opp]),
            phase2_entry(done, &[opp]),
            phase2_entry(marked, &[opp]),
        ];
        let me_next = vec![phase2_entry(marked, &[opp]), phase2_entry(State::default(), &[opp])];
        for (entries, first_active) in [(live, 1), (me_next, 0)] {
            let d = CandDecision {
                entries,
                first_active,
                seed: 9,
            };
            let (g, full) = (label_decision(&bot, &d, 8), label_decision_full(&bot, &d, 8));
            assert_eq!(full.swapped, g.swapped);
            assert_eq!(full.pair_var, g.pair_var);
            assert_eq!(full.samples.len(), g.samples.len());
            for (s, (x, y)) in full.samples.iter().zip(&g.samples) {
                assert_eq!(win_input(&s.me, &s.opp), *x);
                assert_eq!(s.win, *y);
            }
            // Every auxiliary label swaps with the players, not only win.
            let finals = rollout_final_states(&bot, &d.entries, 9, 8, first_active, false);
            let e = &d.entries[0].0;
            let plain = Sample::from_finals(e[0], e[1], &finals[0]);
            let want = if first_active == 0 { plain.swapped() } else { plain };
            assert_eq!(full.samples[0], want);
        }
    }

    fn tmp(name: &str) -> String {
        let d = std::env::temp_dir().join(format!("qwixxer-win-data-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d.to_string_lossy().into_owned()
    }

    fn cfg(positions: usize, holdout: f64, seed: u64) -> DataCfg {
        DataCfg {
            player: "pair_model".into(),
            out: String::new(),
            positions,
            rollouts: 2,
            cand_rollouts: 4,
            holdout,
            keep: 1.0,
            seed,
            resume: false,
        }
    }

    #[test]
    fn kept_share_is_close_and_deterministic() {
        let share = (0..20_000u64).filter(|&i| kept(7, i / 40, i % 40, 0.15)).count();
        assert!((2700..=3300).contains(&share), "{share}");
        assert!((0..1000u64).all(|i| kept(7, i, i, 1.0)));
        assert_eq!(kept(7, 3, 9, 0.5), kept(7, 3, 9, 0.5));
        // Turn ends and decisions of one game are drawn apart.
        let both = (0..2000u64)
            .filter(|&i| kept(7, 0, turn_item(i), 0.5) == kept(7, 0, decision_item(i), 0.5))
            .count();
        assert!((800..=1200).contains(&both), "{both}");
    }

    #[test]
    fn a_kept_share_keeps_its_items_labels_unchanged() {
        let bot = PairStrategy::load("pair_model");
        let some = DataCfg {
            keep: 0.4,
            ..cfg(1, 0.0, 3)
        };
        let (mut turns, mut decisions) = (0, 0);
        for g in 0..6u64 {
            let (t, d) = game_samples(&bot, &cfg(1, 0.0, 3), g);
            let (kt, kd) = game_samples(&bot, &some, g);
            let want_t: Vec<Sample> = (0..t.len() as u64)
                .filter(|&i| kept(3, g, turn_item(i), 0.4))
                .map(|i| t[i as usize])
                .collect();
            let want_d: Vec<DecisionData> = (0..d.len() as u64)
                .filter(|&j| kept(3, g, decision_item(j), 0.4))
                .map(|j| d[j as usize].clone())
                .collect();
            assert_eq!((&kt, &kd), (&want_t, &want_d));
            assert!(kt.len() < t.len() || t.is_empty());
            turns += kt.len();
            decisions += kd.len();
        }
        assert!(turns > 0 && decisions > 0);
    }

    #[test]
    fn holdout_one_holds_out_every_game() {
        let ds = generate(&cfg(40, 1.0, 5), &|| PairStrategy::load("pair_model"));
        assert!(ds.train.turns.is_empty() && ds.train.decisions.is_empty());
        assert!(ds.heldout.turns.len() >= 40 && !ds.heldout.decisions.is_empty());
    }

    #[test]
    fn a_meta_without_keep_reads_as_keeping_everything() {
        let dir = tmp("old-meta");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            format!("{dir}/meta.json"),
            r#"{"player":"p","seed":42,"games":1,"rollouts":64,"cand_rollouts":64,"holdout":0.1,"locks":true,
               "train_turns":1,"train_decisions":0,"heldout_turns":0,"heldout_decisions":0,"bytes":8}"#,
        )
        .unwrap();
        let r = resume_cfg(&dir, 10, None, None, None, None, None, None).unwrap();
        assert_eq!(r.keep, 1.0);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn turn_ends_and_candidates_match_the_win_pi_round() {
        let pair = || PairStrategy::load("pair_model");
        let ds = generate(&cfg(40, 0.0, 11), &pair);
        let round = rollout_round_by(
            40,
            11,
            &pair,
            "pair_model",
            2,
            1.0,
            Some(CandCfg {
                k: 4,
                margin: GATE_MARGIN,
            }),
        );
        assert!(ds.heldout.turns.is_empty() && ds.heldout.decisions.is_empty());
        assert!(ds.train.turns.len() >= 40);
        for (s, (x, y)) in ds.train.turns.iter().zip(&round.positions) {
            assert_eq!(win_input(&s.me, &s.opp), *x);
            assert_eq!(s.win, *y);
        }
        assert_eq!(ds.train.decisions.len(), round.groups.len());
        for (d, g) in ds.train.decisions.iter().zip(&round.groups) {
            assert_eq!((d.swapped, &d.pair_var), (g.swapped, &g.pair_var));
            for (s, (x, y)) in d.samples.iter().zip(&g.samples) {
                assert_eq!(win_input(&s.me, &s.opp), *x);
                assert_eq!(s.win, *y);
            }
        }
        assert_eq!((ds.meta.train_turns, ds.meta.heldout_turns), (ds.train.turns.len(), 0));
    }

    #[test]
    fn generation_is_deterministic_and_splits_whole_games() {
        let pair = || PairStrategy::load("pair_model");
        let (a, b) = (generate(&cfg(300, 0.5, 3), &pair), generate(&cfg(300, 0.5, 3), &pair));
        assert_eq!((&a.train, &a.heldout), (&b.train, &b.heldout));
        assert!(!a.train.turns.is_empty() && !a.heldout.turns.is_empty());
        assert!(a.train.turns.len() + a.heldout.turns.len() >= 300);
        assert_eq!(a.meta.heldout_decisions, a.heldout.decisions.len());
    }

    #[test]
    fn held_out_share_is_close_and_deterministic() {
        let share = (0..10_000u64).filter(|&g| held_out(7, g, 0.1)).count();
        assert!((900..=1100).contains(&share), "{share}");
        assert!((0..1000u64).all(|g| !held_out(7, g, 0.0)));
        assert_eq!(held_out(7, 42, 0.5), held_out(7, 42, 0.5));
    }

    #[test]
    fn a_dataset_round_trips_through_its_files() {
        let dir = tmp("round-trip");
        let ds = generate(&cfg(40, 0.5, 5), &|| PairStrategy::load("pair_model"));
        save(&dir, &ds).unwrap();
        let back = load(&dir).unwrap();
        assert_eq!((&back.train, &back.heldout), (&ds.train, &ds.heldout));
        assert_eq!(
            (
                back.meta.games,
                back.meta.rollouts,
                back.meta.cand_rollouts,
                back.meta.player.as_str()
            ),
            (ds.meta.games, 2, 4, "pair_model")
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_damaged_file_is_an_error() {
        let dir = tmp("damaged");
        let ds = generate(&cfg(40, 0.5, 5), &|| PairStrategy::load("pair_model"));
        save(&dir, &ds).unwrap();
        let path = format!("{dir}/data.bin");
        let bytes = std::fs::read(&path).unwrap();
        std::fs::write(&path, &bytes[..bytes.len() - 7]).unwrap();
        assert!(load(&dir).is_err(), "truncated");
        let mut foreign = bytes.clone();
        foreign[0] = b'X';
        std::fs::write(&path, &foreign).unwrap();
        assert!(load(&dir).unwrap_err().to_string().contains("not a win-data file"));
        let mut newer = bytes.clone();
        newer[4] = 3;
        std::fs::write(&path, &newer).unwrap();
        assert!(load(&dir)
            .unwrap_err()
            .to_string()
            .contains("unsupported win-data version 3"));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn validate_refuses_bad_settings() {
        let base = tmp("validate");
        // `--player` is a win model dir.
        let player = format!("{base}/player");
        let mcfg = crate::dqn::win::WinModelConfig::new().with_width(4);
        crate::dqn::win_pi::save_win(&player, &mcfg, &mcfg.init::<crate::dqn::MyBackend>(&Default::default()));
        let ok = DataCfg {
            out: format!("{base}/out"),
            player: player.clone(),
            ..cfg(10, 0.1, 1)
        };
        assert_eq!(ok.validate(), Ok(()));
        let err = |c: DataCfg| c.validate().unwrap_err();
        let with = |f: &dyn Fn(&mut DataCfg)| {
            let mut c = DataCfg {
                out: ok.out.clone(),
                player: player.clone(),
                ..cfg(10, 0.1, 1)
            };
            f(&mut c);
            c
        };
        assert!(err(with(&|c| c.positions = 0)).contains("--positions"));
        assert!(err(with(&|c| c.rollouts = 1)).contains("--rollouts"));
        assert!(err(with(&|c| c.cand_rollouts = 1)).contains("--cand-rollouts"));
        assert!(err(with(&|c| c.holdout = 1.5)).contains("--holdout"));
        assert!(err(with(&|c| c.holdout = -0.1)).contains("--holdout"));
        assert_eq!(with(&|c| c.holdout = 1.0).validate(), Ok(()));
        assert!(err(with(&|c| c.keep = 0.0)).contains("--keep"));
        assert!(err(with(&|c| c.keep = 1.5)).contains("--keep"));
        assert!(err(with(&|c| c.player = format!("{base}/nope"))).contains("--player"));
        std::fs::create_dir_all(&ok.out).unwrap();
        assert!(err(with(&|_| {})).contains("already exists"));
        // --resume needs a dataset to continue.
        assert!(err(with(&|c| c.resume = true)).contains("nothing to resume"));
        std::fs::remove_dir_all(&base).unwrap();
    }

    /// `write` into `dir` with `positions` turn ends (seed 5, half held out).
    fn written(dir: &str, positions: usize, resume: bool) -> DataMeta {
        let c = DataCfg {
            out: dir.into(),
            resume,
            ..cfg(positions, 0.5, 5)
        };
        write_dataset(&c, &|| PairStrategy::load("pair_model")).unwrap()
    }

    #[test]
    fn a_written_dataset_loads_as_the_in_memory_one() {
        let dir = tmp("written");
        let meta = written(&dir, 60, false);
        let (back, want) = (
            load(&dir).unwrap(),
            generate(&cfg(60, 0.5, 5), &|| PairStrategy::load("pair_model")),
        );
        assert_eq!((&back.train, &back.heldout), (&want.train, &want.heldout));
        assert_eq!((meta.games, back.meta.games), (want.meta.games, want.meta.games));
        assert_eq!(meta.bytes, std::fs::metadata(format!("{dir}/data.bin")).unwrap().len());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn resuming_gives_the_same_files_as_one_run() {
        let (parts, whole) = (tmp("parts"), tmp("whole"));
        written(&parts, 40, false);
        written(&parts, 120, true);
        written(&whole, 120, false);
        let bin = |d: &str| std::fs::read(format!("{d}/data.bin")).unwrap();
        assert_eq!(bin(&parts), bin(&whole));
        let (a, b) = (load(&parts).unwrap(), load(&whole).unwrap());
        assert_eq!((a.meta.games, a.meta.bytes), (b.meta.games, b.meta.bytes));
        // Resuming a finished dataset with the same target adds nothing.
        written(&whole, 120, true);
        assert_eq!(bin(&whole), bin(&parts));
        for d in [parts, whole] {
            std::fs::remove_dir_all(&d).unwrap();
        }
    }

    #[test]
    fn a_half_written_batch_is_ignored_and_cut_on_resume() {
        let (dir, whole) = (tmp("half"), tmp("half-whole"));
        written(&dir, 40, false);
        let before = load(&dir).unwrap();
        // A run stopped mid-batch leaves bytes after the last complete one.
        let path = format!("{dir}/data.bin");
        let mut bytes = std::fs::read(&path).unwrap();
        bytes.extend_from_slice(&[0, 9, 9, 9, 9, 1, 2, 3]);
        std::fs::write(&path, &bytes).unwrap();
        let after = load(&dir).unwrap();
        assert_eq!((&after.train, &after.heldout), (&before.train, &before.heldout));
        written(&dir, 120, true);
        written(&whole, 120, false);
        assert_eq!(
            std::fs::read(&path).unwrap(),
            std::fs::read(format!("{whole}/data.bin")).unwrap()
        );
        for d in [dir, whole] {
            std::fs::remove_dir_all(&d).unwrap();
        }
    }

    #[test]
    fn a_file_shorter_than_its_meta_is_an_error() {
        let dir = tmp("shorter");
        written(&dir, 40, false);
        let path = format!("{dir}/data.bin");
        let bytes = std::fs::read(&path).unwrap();
        std::fs::write(&path, &bytes[..bytes.len() - 7]).unwrap();
        assert!(load(&dir).unwrap_err().to_string().contains("shorter"));
        let c = DataCfg {
            out: dir.clone(),
            resume: true,
            ..cfg(80, 0.5, 5)
        };
        assert!(write_dataset(&c, &|| PairStrategy::load("pair_model"))
            .unwrap_err()
            .to_string()
            .contains("shorter"));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn resume_takes_the_datasets_settings_and_refuses_others() {
        let base = tmp("resume-cfg");
        let player = format!("{base}/player");
        let mcfg = crate::dqn::win::WinModelConfig::new().with_width(4);
        crate::dqn::win_pi::save_win(&player, &mcfg, &mcfg.init::<crate::dqn::MyBackend>(&Default::default()));
        let dir = format!("{base}/data");
        let c = DataCfg {
            player: player.clone(),
            out: dir.clone(),
            ..cfg(40, 0.5, 5)
        };
        write_dataset(&c, &|| PairStrategy::load("pair_model")).unwrap();
        // Nothing given: everything comes from the dataset.
        let r = resume_cfg(&dir, 80, None, None, None, None, None, None).unwrap();
        assert_eq!(
            (
                r.player.as_str(),
                r.rollouts,
                r.cand_rollouts,
                r.holdout,
                r.keep,
                r.seed
            ),
            (player.as_str(), 2, 4, 0.5, 1.0, 5)
        );
        assert!(r.resume && r.positions == 80 && r.validate().is_ok());
        // Given and equal: fine; given and different: refused.
        assert!(resume_cfg(
            &dir,
            80,
            Some(player.clone()),
            Some(2),
            Some(4),
            Some(0.5),
            Some(1.0),
            Some(5)
        )
        .is_ok());
        let refuse = |r: Result<DataCfg, String>, flag: &str| assert!(r.unwrap_err().contains(flag));
        refuse(
            resume_cfg(&dir, 80, Some(format!("{base}/other")), None, None, None, None, None),
            "--player",
        );
        refuse(
            resume_cfg(&dir, 80, None, Some(64), None, None, None, None),
            "--rollouts",
        );
        refuse(
            resume_cfg(&dir, 80, None, None, Some(64), None, None, None),
            "--cand-rollouts",
        );
        refuse(
            resume_cfg(&dir, 80, None, None, None, Some(0.1), None, None),
            "--holdout",
        );
        refuse(resume_cfg(&dir, 80, None, None, None, None, Some(0.5), None), "--keep");
        refuse(resume_cfg(&dir, 80, None, None, None, None, None, Some(6)), "--seed");
        refuse(
            resume_cfg(&format!("{base}/nope"), 80, None, None, None, None, None, None),
            "--resume",
        );
        // A resume config that disagrees with the dataset fails validation too.
        assert!(DataCfg { seed: 6, ..r }.validate().unwrap_err().contains("--seed"));
        std::fs::remove_dir_all(&base).unwrap();
    }
}
