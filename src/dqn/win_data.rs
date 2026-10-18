//! Stored datasets for the win net: turn ends and the candidates of the
//! decisions win-search would search, labelled with full-game rollouts
//! (win and auxiliary labels), split into training and held-out games.
//!
//! Design: docs/superpowers/specs/2026-10-19-win-thermo-aux-design.md

use crate::dqn::win_distill::{sample_var, CandDecision};
use crate::dqn::win_pi::splitmix64;
use crate::dqn::win_train::TurnEnds;
use crate::game::{Game, Player};
use crate::state::State;
use crate::strategy::search::{phase2_entry, rollout_final_states, GATE_MARGIN};
use crate::strategy::sim::SimGame;
use crate::strategy::Bot;
use burn::prelude::*;
use rand::{rngs::SmallRng, SeedableRng};
use rayon::prelude::*;
use std::cell::RefCell;
use std::fs::File;
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

/// `meta.json`: how the dataset was made, and its sizes.
#[derive(Config, Debug)]
pub struct DataMeta {
    pub player: String,
    pub seed: u64,
    pub games: usize,
    pub rollouts: usize,
    pub cand_rollouts: usize,
    pub holdout: f64,
    pub train_turns: usize,
    pub train_decisions: usize,
    pub heldout_turns: usize,
    pub heldout_decisions: usize,
}

#[derive(Debug)]
pub struct Dataset {
    pub meta: DataMeta,
    pub train: Split,
    pub heldout: Split,
}

/// `win-data` settings.
pub struct DataCfg {
    /// Win model dir that plays the games and the rollouts (metadata only;
    /// [`generate`] takes the bot itself).
    pub player: String,
    pub out: String,
    /// Turn ends to reach; whole games are kept, so the count is reached or just passed.
    pub positions: usize,
    /// Rollouts per turn end.
    pub rollouts: usize,
    /// Rollouts per candidate.
    pub cand_rollouts: usize,
    /// Share of games held out.
    pub holdout: f64,
    pub seed: u64,
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
        if !(0.0..1.0).contains(&self.holdout) {
            return Err("--holdout must be in [0, 1)".into());
        }
        for f in ["config.json", "model.mpk"] {
            if !Path::new(&format!("{}/{f}", self.player)).exists() {
                return Err(format!("--player {}: {f} not found", self.player));
            }
        }
        if Path::new(&self.out).exists() {
            return Err(format!(
                "{} already exists: move it aside or pick another --out",
                self.out
            ));
        }
        Ok(())
    }
}

/// Whether game `game` of a dataset seeded `seed` is held out; about
/// `share` of the games are.
pub fn held_out(seed: u64, game: u64, share: f64) -> bool {
    let h = splitmix64(seed ^ splitmix64(game ^ 0x4E1D_0001));
    ((h >> 11) as f64 / (1u64 << 53) as f64) < share
}

/// One self-play game's labelled turn ends and candidate decisions. Game
/// `g` is seeded `seed + 1000·g` and its turn ends are rolled out from seed
/// `game seed ^ 0x5011_0075`, as in `rollout_round_by`, so the win labels
/// are the ones `win-pi --labels cand` would give.
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
        .map(|(me, opp)| phase2_entry(me, &[opp]))
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
        .map(|d| label_decision_full(player, &d, cfg.cand_rollouts))
        .collect();
    (turns, decisions)
}

/// The dataset of `player`'s self-play (one bot per worker thread): whole
/// games in index order until `cfg.positions` turn ends, each game's turn
/// ends and decisions on one side of the split.
pub fn generate<P: Bot + Clone + 'static>(cfg: &DataCfg, player: &(dyn Fn() -> P + Sync)) -> Dataset {
    let (mut train, mut heldout) = (Split::default(), Split::default());
    let (mut games, mut kept, mut turns) = (0u64, 0usize, 0usize);
    while turns < cfg.positions {
        let per_game = if kept == 0 { 25.0 } else { turns as f64 / kept as f64 };
        let batch = (((cfg.positions - turns) as f64 / per_game).ceil() as u64).max(64);
        let chunk: Vec<(Vec<Sample>, Vec<DecisionData>)> = (games..games + batch)
            .into_par_iter()
            .map_init(player, |player, g| game_samples(&*player, cfg, g))
            .collect();
        for (g, (t, d)) in (games..).zip(chunk) {
            if turns >= cfg.positions {
                break;
            }
            turns += t.len();
            kept += 1;
            let side = if held_out(cfg.seed, g, cfg.holdout) {
                &mut heldout
            } else {
                &mut train
            };
            side.turns.extend(t);
            side.decisions.extend(d);
        }
        games += batch;
    }
    Dataset {
        meta: DataMeta {
            player: cfg.player.clone(),
            seed: cfg.seed,
            games: kept,
            rollouts: cfg.rollouts,
            cand_rollouts: cfg.cand_rollouts,
            holdout: cfg.holdout,
            train_turns: train.turns.len(),
            train_decisions: train.decisions.len(),
            heldout_turns: heldout.turns.len(),
            heldout_decisions: heldout.decisions.len(),
        },
        train,
        heldout,
    }
}

const MAGIC: &[u8; 4] = b"QWXD";
const VERSION: u32 = 1;

fn invalid(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
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

fn get_u64(r: &mut impl Read) -> io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
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

fn put_split(w: &mut impl Write, s: &Split) -> io::Result<()> {
    w.write_all(&(s.turns.len() as u64).to_le_bytes())?;
    for t in &s.turns {
        put_sample(w, t)?;
    }
    w.write_all(&(s.decisions.len() as u64).to_le_bytes())?;
    for d in &s.decisions {
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
/// damaged count runs into the end of the file instead.
fn get_split(r: &mut impl Read) -> io::Result<Split> {
    let mut s = Split::default();
    for _ in 0..get_u64(r)? {
        s.turns.push(get_sample(r)?);
    }
    for _ in 0..get_u64(r)? {
        let mut h = [0u8; 2];
        r.read_exact(&mut h)?;
        let samples = (0..h[1]).map(|_| get_sample(r)).collect::<io::Result<Vec<_>>>()?;
        let pair_var = (1..h[1]).map(|_| get_f32(r)).collect::<io::Result<Vec<_>>>()?;
        s.decisions.push(DecisionData {
            samples,
            pair_var,
            swapped: h[0] != 0,
        });
    }
    Ok(s)
}

/// Write `dir/meta.json` and `dir/data.bin` (magic, version, then the
/// training and held-out splits; little-endian).
pub fn save(dir: &str, ds: &Dataset) -> io::Result<()> {
    std::fs::create_dir_all(dir)?;
    ds.meta.save(format!("{dir}/meta.json"))?;
    let mut w = BufWriter::new(File::create(format!("{dir}/data.bin"))?);
    w.write_all(MAGIC)?;
    w.write_all(&VERSION.to_le_bytes())?;
    put_split(&mut w, &ds.train)?;
    put_split(&mut w, &ds.heldout)?;
    w.flush()
}

/// Read a dataset written by [`save`].
pub fn load(dir: &str) -> io::Result<Dataset> {
    let meta = DataMeta::load(format!("{dir}/meta.json")).map_err(|e| invalid(format!("{dir}/meta.json: {e:?}")))?;
    let mut r = BufReader::new(File::open(format!("{dir}/data.bin"))?);
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(invalid(format!("{dir}/data.bin is not a win-data file")));
    }
    let mut v = [0u8; 4];
    r.read_exact(&mut v)?;
    let version = u32::from_le_bytes(v);
    if version != VERSION {
        return Err(invalid(format!(
            "{dir}/data.bin: unsupported win-data version {version}"
        )));
    }
    let train = get_split(&mut r)?;
    let heldout = get_split(&mut r)?;
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
            seed,
        }
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
        newer[4] = 2;
        std::fs::write(&path, &newer).unwrap();
        assert!(load(&dir)
            .unwrap_err()
            .to_string()
            .contains("unsupported win-data version 2"));
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
        assert!(err(with(&|c| c.holdout = 1.0)).contains("--holdout"));
        assert!(err(with(&|c| c.player = format!("{base}/nope"))).contains("--player"));
        std::fs::create_dir_all(&ok.out).unwrap();
        assert!(err(with(&|_| {})).contains("already exists"));
        std::fs::remove_dir_all(&base).unwrap();
    }
}
