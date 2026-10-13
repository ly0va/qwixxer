//! Decision-time search: gated truncated rollouts bootstrapped by a win
//! probability. Generic over any [`WinProb`] bot.
//!
//! Design: docs/superpowers/specs/2026-06-10-pair-search-design.md

use super::bot_impl::{
    active_phase1_choices, active_phase2_choices, eval_decision, is_safe_lock, mark_choices_with,
    opp_best_phase1_score, passive_phase1_choices, passive_phase1_impl, phase1_plan_choices_with, Decision,
};
use super::sim::{BatchedRollouts, SimGame};
use super::Bot;
use super::Strategy;
use crate::game::DiceSource;
use crate::state::{Mark, State};
use rand::rngs::SmallRng;
use rand::SeedableRng;

/// Default candidates searched per decision (top by static value); also the
/// endgame gate's lock check.
pub const K_CANDIDATES: usize = 2;
/// Sampled futures per candidate (CRN: dice shared across candidates) at a
/// fixed sample count: the Phase 13–25 production value, still used by
/// [`SearchConfig::legacy`].
pub const K_SAMPLES: usize = 128;
/// Production sample cap per candidate with sequential sampling (Phase 26).
pub const K_SAMPLES_SEQ: usize = 512;
/// Production sequential start: samples per candidate before the first
/// stopping check, doubled up to [`K_SAMPLES_SEQ`] (Phase 26).
pub const SEQ_START: usize = 32;
/// Top-2 static value gap (bot's evaluate units) below which search triggers.
pub const GATE_MARGIN: f32 = 0.15;
/// Full turns simulated after completing the current turn, as a multiple of
/// the player count (1 = one full round).
pub const HORIZON_ROUNDS: usize = 1;

/// Capability for search leaf scoring: a calibrated win probability, batched
/// over independent (our_state, opp_states) groups in one inference call.
pub trait WinProb: Bot {
    /// P(group's player finishes ahead of its leading opponent), one per group.
    fn win_prob_multi(&self, groups: &[(&State, &[State])]) -> Vec<f32>;
}

/// Standard normal CDF via the Abramowitz–Stegun erf approximation
/// (7.1.26, |error| < 1.5e-7).
pub fn phi(z: f32) -> f32 {
    0.5 * (1.0 + erf(z as f64 / std::f64::consts::SQRT_2)) as f32
}

fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592)
            * t
            * (-x * x).exp();
    sign * y
}

fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

fn absorb_state(mut h: u64, s: &State) -> u64 {
    let totals = s.row_totals();
    let frees = s.row_free_values();
    for i in 0..4 {
        // (total, free) fully determines a row; locked == free.is_none().
        h = splitmix64(h ^ (((totals[i] as u64) << 8) | frees[i].map_or(0xFF, |f| f as u64)));
    }
    splitmix64(h ^ s.strikes as u64)
}

/// Deterministic seed from the decision context. Candidate-independent by
/// construction (the candidate is not hashed), so all candidates of one
/// decision share dice streams (common random numbers), and identical
/// contexts replay identically (reproducible benches).
pub fn context_seed(our: &State, opps: &[State], dice: [u8; 6]) -> u64 {
    let mut h = absorb_state(0x9E3779B97F4A7C15, our);
    for o in opps {
        h = absorb_state(h, o);
    }
    let packed = dice.iter().fold(0u64, |acc, &d| (acc << 8) | d as u64);
    splitmix64(h ^ packed)
}

/// Per-player dice-stream seed for one rollout sample.
pub fn sample_player_seed(decision_seed: u64, sample: usize, player: usize) -> u64 {
    splitmix64(decision_seed ^ ((sample as u64) << 16) ^ player as u64)
}

/// Ordered white-pair stratum (0..36) of a roll: `(W1 - 1) * 6 + (W2 - 1)`.
pub fn white_stratum(dice: [u8; 6]) -> usize {
    (dice[0] as usize - 1) * 6 + (dice[1] as usize - 1)
}

/// Like [`sample_player_seed`], but the stream's first roll lands its white
/// dice in a fixed stratum: `sample % 36`, permuted per player so the
/// players' first rolls are not lock-stepped. With a sample count that is a
/// multiple of 36, every player's first rollout roll has exactly uniform
/// white dice; colors and all later rolls stay pseudo-random. Found by a
/// deterministic search over seed variants (~36 tries expected).
pub fn stratified_player_seed(decision_seed: u64, sample: usize, player: usize) -> u64 {
    // Multipliers coprime to 36 make each per-player schedule a bijection.
    const MULT: [usize; 4] = [1, 5, 7, 11];
    let target = ((sample % 36) * MULT[player % 4] + 13 * player) % 36;
    let base = sample_player_seed(decision_seed, sample, player);
    (0u64..)
        .map(|attempt| splitmix64(base ^ attempt.wrapping_mul(0x9E37_79B9_7F4A_7C15)))
        .find(|&s| white_stratum(SmallRng::seed_from_u64(s).roll()) == target)
        .expect("some seed hits every stratum")
}

/// Rollout dice-stream seed: stratified or plain.
fn rollout_seed(stratify: bool, decision_seed: u64, sample: usize, player: usize) -> u64 {
    if stratify {
        stratified_player_seed(decision_seed, sample, player)
    } else {
        sample_player_seed(decision_seed, sample, player)
    }
}

/// Why search fired (for diagnostics).
#[derive(Default, Debug, Clone)]
pub struct SearchStats {
    pub active_decisions: u32,
    /// Decisions that reached the gate (>= 2 candidates, not meta-forced).
    pub eligible: u32,
    pub gate_close: u32,
    pub gate_endgame: u32,
    pub searched: u32,
    /// Search picked a different move than static evaluation.
    pub disagreements: u32,
    /// ...of which a gate (close OR endgame) had fired.
    pub disagreements_gated: u32,
    /// Top-2 static value gaps at eligible decisions.
    pub gaps: Vec<f32>,
    /// Passive (white-sum) decisions searched (subset of `searched`).
    pub passive_searched: u32,
    /// Forced safe locks compared against alternatives by rollouts.
    pub lock_adjudications: u32,
    /// ...of which an alternative beat the lock.
    pub lock_overrides: u32,
    /// Rollout samples per candidate, summed over truncated searches
    /// (`/ searched` = mean samples used; below `k_samples` with `seq`).
    pub rollout_samples: u64,
}

/// Search knobs. `Default` is the production configuration (Phase 21:
/// passive search + lock adjudication; Phase 25: blend 0.7; Phase 26:
/// sequential sampling 32 → 512);
/// [`SearchConfig::legacy`] is the Phase 13–20 production search.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SearchConfig {
    /// Sampled futures per candidate in the truncated (gated) search.
    pub k_samples: usize,
    /// Also search passive (white-sum) decisions.
    pub search_passive: bool,
    /// Adjudicate forced safe locks with this many full-game rollouts per
    /// compared candidate; `None` = the meta-rule forces the lock.
    pub lock_k: Option<usize>,
    /// Stratify each player's first rollout roll over the 36 ordered white
    /// pairs (exact balance when the sample count is a multiple of 36).
    pub stratify: bool,
    /// Pick by `(1 − λ)·Φ(static z) + λ·rollout mean` (shrink the noisy
    /// rollout estimate toward the static value); `None` = rollout mean alone.
    pub blend: Option<f32>,
    /// Sequential sampling: start with this many samples per candidate and
    /// double (up to `k_samples`) until the pick is decisive.
    pub seq: Option<usize>,
    /// Sequential stopping threshold: stop once every candidate's blended
    /// score difference to the static pick exceeds `stop` × λ·SE.
    pub stop: f32,
    /// Close-gate margin: search when the top-2 static gap is below this.
    pub gate: f32,
    /// Candidates searched per decision (top by static value), at least 2.
    /// With `seq`, candidates that fall decisively behind stop sampling.
    pub cands: usize,
}

/// Production rollout/static blend weight (Phase 25).
pub const BLEND: f32 = 0.7;
/// Default sequential stopping threshold (in standard errors).
pub const SEQ_STOP: f32 = 2.0;

/// Full-game rollouts per compared candidate at a forced safe lock.
pub const LOCK_K: usize = 256;

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            k_samples: K_SAMPLES_SEQ,
            search_passive: true,
            lock_k: Some(LOCK_K),
            stratify: false,
            blend: Some(BLEND),
            seq: Some(SEQ_START),
            stop: SEQ_STOP,
            gate: GATE_MARGIN,
            cands: K_CANDIDATES,
        }
    }
}

impl SearchConfig {
    /// Phase 13–20 production search: active decisions only, safe locks
    /// forced by the meta-rule, pure rollout mean, fixed 128 samples. Kept
    /// for reproducing recorded analyses and training recipes.
    pub fn legacy() -> Self {
        Self {
            k_samples: K_SAMPLES,
            search_passive: false,
            lock_k: None,
            blend: None,
            seq: None,
            ..Self::default()
        }
    }

    /// Parse a comma-separated spec on top of the production defaults:
    /// `passive` / `nopassive`, `lock=N` / `nolock` (lock-adjudication
    /// rollouts), `strat`, `k=N` (truncated-search samples), `blend=λ`
    /// (λ in [0, 1]) / `noblend`, `seq=N` / `noseq`
    /// (sequential sampling from N up to k; `noseq` = fixed k), `cands=N`
    /// (candidates searched, ≥ 2), `stop=C` (sequential threshold, may be
    /// `inf`), `gate=G` (close-gate margin ≥ 0). Empty = production.
    pub fn parse(spec: &str) -> Result<Self, String> {
        let mut cfg = Self::default();
        let num = |key: &str, v: &str| v.parse::<usize>().map_err(|e| format!("{key}={v}: {e}"));
        let real = |key: &str, v: &str, max: f32| match v.parse::<f32>() {
            Ok(x) if (0.0..=max).contains(&x) => Ok(x),
            Ok(_) => Err(format!("{key}={v}: out of range [0, {max}]")),
            Err(e) => Err(format!("{key}={v}: {e}")),
        };
        for tok in spec.split(',').map(str::trim).filter(|t| !t.is_empty()) {
            match tok.split_once('=') {
                None if tok == "passive" => cfg.search_passive = true,
                None if tok == "nopassive" => cfg.search_passive = false,
                None if tok == "nolock" => cfg.lock_k = None,
                None if tok == "strat" => cfg.stratify = true,
                None if tok == "noblend" => cfg.blend = None,
                None if tok == "noseq" => cfg.seq = None,
                Some(("k", v)) => cfg.k_samples = num("k", v)?,
                Some(("lock", v)) => cfg.lock_k = Some(num("lock", v)?),
                Some(("blend", v)) => cfg.blend = Some(real("blend", v, 1.0)?),
                Some(("seq", v)) => cfg.seq = Some(num("seq", v)?),
                Some(("stop", v)) => cfg.stop = real("stop", v, f32::INFINITY)?,
                Some(("gate", v)) => cfg.gate = real("gate", v, f32::MAX)?,
                Some(("cands", v)) => cfg.cands = num("cands", v)?,
                _ => {
                    return Err(format!(
                        "unknown search option {tok:?} (passive, nopassive, lock=N, nolock, strat, k=N, \
                     blend=λ, noblend, seq=N, noseq, stop=C, gate=G, cands=N)"
                    ))
                }
            }
        }
        if cfg.seq == Some(0) {
            return Err("seq must be at least 1".into());
        }
        if cfg.cands < 2 {
            return Err("cands must be at least 2".into());
        }
        Ok(cfg)
    }
}

impl std::fmt::Display for SearchConfig {
    /// Compact spec that [`SearchConfig::parse`] reads back.
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "k={}", self.k_samples)?;
        write!(f, "{}", if self.search_passive { ",passive" } else { ",nopassive" })?;
        match self.lock_k {
            Some(k) => write!(f, ",lock={k}")?,
            None => write!(f, ",nolock")?,
        }
        if self.stratify {
            write!(f, ",strat")?;
        }
        match self.blend {
            Some(l) => write!(f, ",blend={l}")?,
            None => write!(f, ",noblend")?,
        }
        match self.seq {
            Some(s) => write!(f, ",seq={s}")?,
            None => write!(f, ",noseq")?,
        }
        if self.stop != SEQ_STOP {
            write!(f, ",stop={}", self.stop)?;
        }
        if self.gate != GATE_MARGIN {
            write!(f, ",gate={}", self.gate)?;
        }
        if self.cands != K_CANDIDATES {
            write!(f, ",cands={}", self.cands)?;
        }
        Ok(())
    }
}

pub struct SearchBot<B: WinProb> {
    pub bot: B,
    /// Search every eligible decision regardless of gates (diagnostics).
    pub force: bool,
    pub stats: Option<std::rc::Rc<std::cell::RefCell<SearchStats>>>,
    pub cfg: SearchConfig,
}

impl<B: WinProb> SearchBot<B> {
    pub fn new(bot: B) -> Self {
        Self::with_config(bot, SearchConfig::default())
    }

    pub fn with_config(bot: B, cfg: SearchConfig) -> Self {
        if cfg.stratify {
            assert!(
                cfg.k_samples % 36 == 0 && cfg.lock_k.is_none_or(|k| k % 36 == 0),
                "stratify needs sample counts that are a multiple of 36: {cfg:?}"
            );
        }
        Self {
            bot,
            force: false,
            stats: None,
            cfg,
        }
    }
}

impl<B: WinProb> std::fmt::Debug for SearchBot<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "SearchBot({:?})", self.bot)
    }
}

/// A shortlisted candidate: the move, its static value, its post-move state.
struct Candidate {
    mark: Option<Mark>,
    value: f32,
    post: State,
}

/// Gate predicate over (static value, post-move state) pairs sorted desc by
/// value. `our` is the pre-decision state (lock comparisons are relative to
/// it). `pub(crate)` so training-time distillation can reuse the exact gate
/// semantics; the (value, post) signature avoids exposing the private
/// `Candidate` type.
pub(crate) fn gates(values: &[(f32, State)], our: &State, opps: &[State]) -> (bool, bool) {
    gates_with(values, our, opps, GATE_MARGIN)
}

/// [`gates`] with an explicit close-gate margin ([`SearchConfig::gate`]).
pub(crate) fn gates_with(values: &[(f32, State)], our: &State, opps: &[State], margin: f32) -> (bool, bool) {
    let close = values[0].0 - values[1].0 < margin;
    let endgame = our.count_locked() >= 1
        || opps.iter().any(|s| s.count_locked() >= 1)
        || our.strikes >= 3
        || opps.iter().any(|s| s.strikes >= 3)
        || values
            .iter()
            .take(K_CANDIDATES)
            .any(|(_, post)| post.count_locked() > our.count_locked());
    (close, endgame)
}

/// Deterministic completion of our turn after a phase-1 mark choice:
/// opponents' (simultaneous) phase-1 marks are `sim_opp` and our phase 2 is
/// re-decided. Returns (all-player states, game-ended). Extracted from
/// `SearchBot::active_phase1` for reuse by training-time distillation.
pub fn phase1_entry(
    bot: &impl Bot,
    state: &State,
    sim_opp: &[State],
    dice: [u8; 6],
    mark: Option<Mark>,
) -> (Vec<State>, bool) {
    let mut our = *state;
    if let Some(m) = mark {
        our.apply_mark(m);
    }
    let mut all: Vec<State> = std::iter::once(our).chain(sim_opp.iter().copied()).collect();
    SimGame::propagate_locks(&mut all);
    if SimGame::game_over(&all) {
        return (all, true);
    }
    let view = SimGame::opp_view(&all, 0);
    let d = active_phase2_choices(&all[0], &view, dice, mark.is_some());
    match eval_decision(bot, d, &view) {
        Some(m) => all[0].apply_mark(m),
        None if mark.is_none() => all[0].apply_strike(),
        None => {}
    }
    SimGame::propagate_locks(&mut all);
    let ended = SimGame::game_over(&all);
    (all, ended)
}

/// Post-phase-2 entry: candidate post-state + current opponents.
/// Extracted from `SearchBot::active_phase2`.
pub fn phase2_entry(post: State, opps: &[State]) -> (Vec<State>, bool) {
    let mut all: Vec<State> = std::iter::once(post).chain(opps.iter().copied()).collect();
    SimGame::propagate_locks(&mut all);
    let ended = SimGame::game_over(&all);
    (all, ended)
}

/// Complete the WHOLE current turn deterministically for a passive-phase-1
/// candidate, in the entry's `[us, opps...]` frame.
///
/// Frame mapping (verified against `Game::play`): the game passes player i's
/// opponents as `opp_states = [pre_phase1[(i+1)%n], ..]` (turn-ordered from i)
/// and `active_player` = `(active + n − i) % n − 1` = the index of the active
/// player *within that opp_states array* (relative-to-receiver, 0-based into
/// opps), NOT an absolute seat. So in the entry `[us, opps...]`, the active
/// player sits at entry index `active_player + 1`, and the player who acts
/// first in the rollout (the seat after the active one) is at entry index
/// `(active_player + 2) % n` in our-relative frame.
///
/// Approximation (identical across candidates, which is all that matters for
/// the contrast): the real phase-1 was simultaneous; here our candidate mark
/// is applied first, then every OTHER passive player's phase 1, then the
/// active player's phase 1 + phase 2, with lock propagation between steps.
pub fn passive_entry(
    bot: &impl Bot,
    state: &State,
    opps: &[State],
    dice: [u8; 6],
    active_player: usize,
    mark: Option<Mark>,
) -> (Vec<State>, bool) {
    let n = opps.len() + 1;
    // entry index of the active player (us = 0, opps start at 1).
    let active_idx = active_player + 1;

    let mut all: Vec<State> = std::iter::once(*state).chain(opps.iter().copied()).collect();
    if let Some(m) = mark {
        all[0].apply_mark(m);
    }
    SimGame::propagate_locks(&mut all);
    if SimGame::game_over(&all) {
        return (all, true);
    }

    // Other passive players' (simultaneous) phase 1 — every entry seat that is
    // neither us (0) nor the active player. We apply sequentially with lock
    // propagation; this is the same approximation search makes for sim_opp.
    for j in 1..n {
        if j == active_idx {
            continue;
        }
        let view = SimGame::opp_view(&all, j);
        if let Some(m) = passive_phase1_impl(bot, &all[j], &view, dice) {
            all[j].apply_mark(m);
        }
    }
    SimGame::propagate_locks(&mut all);
    if SimGame::game_over(&all) {
        return (all, true);
    }

    // Active player's phase 1.
    let view = SimGame::opp_view(&all, active_idx);
    let (d, _) = active_phase1_choices(bot, &all[active_idx], &view, dice);
    let p1 = eval_decision(bot, d, &view);
    if let Some(m) = p1 {
        all[active_idx].apply_mark(m);
    }
    SimGame::propagate_locks(&mut all);
    if SimGame::game_over(&all) {
        return (all, true);
    }

    // Active player's phase 2.
    let view = SimGame::opp_view(&all, active_idx);
    let d = active_phase2_choices(&all[active_idx], &view, dice, p1.is_some());
    match eval_decision(bot, d, &view) {
        Some(m) => all[active_idx].apply_mark(m),
        None if p1.is_none() => all[active_idx].apply_strike(),
        None => {}
    }
    SimGame::propagate_locks(&mut all);
    let ended = SimGame::game_over(&all);
    (all, ended)
}

/// Phase 16 comparison set at a forced safe lock, as indices into
/// `sorted` (rule-free candidates, best static value first): the lock, the
/// best non-lock alternative, and the runner-up safe lock if any. `None` when
/// there is nothing to compare (lock absent, or every candidate a safe lock).
pub(crate) fn select_lock_trio(state: &State, sorted: &[Option<Mark>], lock: Mark) -> Option<Vec<usize>> {
    let lock_i = sorted.iter().position(|&m| m == Some(lock))?;
    let alt_i = (0..sorted.len()).find(|&i| !is_safe_lock(state, sorted[i]))?;
    let mut trio = vec![lock_i, alt_i];
    if let Some(i2) = (0..sorted.len()).find(|&i| i != lock_i && is_safe_lock(state, sorted[i])) {
        trio.push(i2);
    }
    Some(trio)
}

/// Entry-frame seat that takes the first rollout turn after a passive
/// decision: the seat after the active one (`active_player` indexes the
/// receiver's turn-ordered `opp_states`; see [`passive_entry`]).
pub fn passive_first_active(active_player: usize, n: usize) -> usize {
    (active_player + 2) % n
}

/// Full-game CRN rollouts: every sample rolled to completion; returns the
/// final all-player states per entry per sample (ended entries: one element,
/// the entry itself). `first_active` = first player to act in the rollout.
/// Used by training-time distillation; the truncated production scorer in
/// `search_pick` is unaffected.
pub(crate) fn rollout_final_states(
    bot: &impl Bot,
    entries: &[(Vec<State>, bool)],
    seed: u64,
    k: usize,
    first_active: usize,
    stratify: bool,
) -> Vec<Vec<Vec<State>>> {
    let n = entries[0].0.len();
    let mut sims: Vec<SimGame> = Vec::new();
    let mut sim_owner: Vec<usize> = Vec::new();
    for (ei, (states, ended)) in entries.iter().enumerate() {
        if *ended {
            continue;
        }
        for s in 0..k {
            sims.push(SimGame {
                states: states.clone(),
                active: first_active,
                rngs: (0..n)
                    .map(|p| SmallRng::seed_from_u64(rollout_seed(stratify, seed, s, p)))
                    .collect(),
                over: false,
            });
            sim_owner.push(ei);
        }
    }
    let mut driver = BatchedRollouts::new(bot, sims);
    let mut turns = 0;
    while !driver.all_over() {
        driver.step_turn();
        turns += 1;
        assert!(turns <= 200, "rollout exceeded 200 turns — game-end invariant violated");
    }
    let mut out: Vec<Vec<Vec<State>>> = entries
        .iter()
        .map(|(states, ended)| {
            if *ended {
                vec![states.clone()]
            } else {
                Vec::with_capacity(k)
            }
        })
        .collect();
    for (i, sim) in driver.sims.iter().enumerate() {
        out[sim_owner[i]].push(sim.states.clone());
    }
    out
}

/// Blended score of candidate `i` minus candidate `j` from their CRN-paired
/// samples: `(1 − λ)·ΔΦ(static z) + λ·mean(x_i − x_j)`, with λ = `blend`
/// (1 = pure rollout mean when `None`), and that difference's standard
/// error `λ·SE(mean(x_i − x_j))`.
fn paired_score(cands: &[Candidate], values: &[Vec<f32>], cfg: &SearchConfig, i: usize, j: usize) -> (f32, f32) {
    let k = values[i].len() as f32;
    let diffs = values[i].iter().zip(&values[j]).map(|(a, b)| a - b);
    let mean = diffs.clone().sum::<f32>() / k;
    let var = diffs.map(|d| (d - mean) * (d - mean)).sum::<f32>() / (k - 1.0).max(1.0);
    let lambda = cfg.blend.unwrap_or(1.0);
    let score = (1.0 - lambda) * (phi(cands[i].value) - phi(cands[j].value)) + lambda * mean;
    (score, lambda * (var / k).sqrt())
}

/// One round of successive elimination over the still-`active` candidates
/// (`values[e][k]` = sample k's leaf value, CRN-paired across entries; all
/// active entries hold the same samples). The leader is the best paired
/// score, ties keeping the lower (static) index. Every other active
/// candidate the leader beats by at least `stop` standard errors is
/// eliminated. Returns the leader and whether it is the last one standing.
/// With two candidates this is a single paired test of the runner-up
/// against the static pick.
fn eliminate_by_paired_rollouts(
    cands: &[Candidate],
    values: &[Vec<f32>],
    active: &mut [bool],
    cfg: &SearchConfig,
) -> (usize, bool) {
    let live: Vec<usize> = (0..values.len()).filter(|&e| active[e]).collect();
    let mut best = live[0];
    for &e in &live[1..] {
        if paired_score(cands, values, cfg, e, best).0 > 0.0 {
            best = e;
        }
    }
    for &e in &live {
        let (gap, se) = paired_score(cands, values, cfg, best, e);
        // An infinite `stop` never eliminates (and avoids ∞·0 = NaN at zero SE).
        if e != best && cfg.stop.is_finite() && gap >= cfg.stop * se {
            active[e] = false;
        }
    }
    (best, active.iter().filter(|&&a| a).count() == 1)
}

impl<B: WinProb> SearchBot<B> {
    /// Leaf values of samples `ks` for every non-ended entry: one batched
    /// rollout of `HORIZON_ROUNDS` rounds, exact outcome where the game ended,
    /// else the bot's win prob. Sample `k`'s dice depend only on (seed, k,
    /// player), so ranges compose: samples are CRN across entries and
    /// identical however `ks` is split. Ended entries get their outcome;
    /// entries not in `active` are skipped (empty).
    fn rollout_values(
        &self,
        entries: &[(Vec<State>, bool)],
        active: &[bool],
        seed: u64,
        first_active: usize,
        ks: std::ops::Range<usize>,
    ) -> Vec<Vec<f32>> {
        let n = entries[0].0.len();
        let mut sims: Vec<SimGame> = Vec::new();
        let mut sim_owner: Vec<usize> = Vec::new(); // entry index per sim
        for (ei, (states, ended)) in entries.iter().enumerate() {
            if *ended || !active[ei] {
                continue;
            }
            for k in ks.clone() {
                sims.push(SimGame {
                    states: states.clone(),
                    active: first_active,
                    rngs: (0..n)
                        .map(|p| SmallRng::seed_from_u64(rollout_seed(self.cfg.stratify, seed, k, p)))
                        .collect(),
                    over: false,
                });
                sim_owner.push(ei);
            }
        }

        let mut driver = BatchedRollouts::new(&self.bot, sims);
        for _ in 0..(HORIZON_ROUNDS * n) {
            if driver.all_over() {
                break;
            }
            driver.step_turn();
        }

        // Score: exact outcome for finished sims, win prob for survivors
        // (one batched call over all survivors of all entries).
        let survivor_idx: Vec<usize> = (0..driver.sims.len()).filter(|&i| !driver.sims[i].over).collect();
        let survivor_views: Vec<Vec<State>> = survivor_idx
            .iter()
            .map(|&i| SimGame::opp_view(&driver.sims[i].states, 0))
            .collect();
        let groups: Vec<(&State, &[State])> = survivor_idx
            .iter()
            .zip(&survivor_views)
            .map(|(&i, view)| (&driver.sims[i].states[0], view.as_slice()))
            .collect();
        let probs = self.bot.win_prob_multi(&groups);
        let mut prob_iter = probs.into_iter();
        let mut out: Vec<Vec<f32>> = entries
            .iter()
            .zip(active)
            .map(|((states, ended), &on)| {
                if !on {
                    Vec::new()
                } else if *ended {
                    vec![SimGame::outcome(states); ks.len()]
                } else {
                    Vec::with_capacity(ks.len())
                }
            })
            .collect();
        for (i, sim) in driver.sims.iter().enumerate() {
            let p = if sim.over {
                SimGame::outcome(&sim.states)
            } else {
                prob_iter.next().unwrap()
            };
            out[sim_owner[i]].push(p);
        }
        out
    }

    /// Shared post-static-eval flow: search the top-K entries, pick.
    /// `entries[i]` = (full player states after candidate i's deterministic
    /// completion of the current turn, game-already-over flag). Returns the
    /// index of the winning candidate. Takes `&self` (stats use RefCell) so
    /// `self.bot` can be borrowed by the rollout driver without fuss.
    /// `first_active` = entry-frame seat that takes the first rollout turn.
    fn search_pick(
        &self,
        cands: &[Candidate],
        entries: Vec<(Vec<State>, bool)>,
        seed: u64,
        first_active: usize,
    ) -> usize {
        debug_assert_eq!(entries.len(), cands.len().min(self.cfg.cands));
        let k_max = self.cfg.k_samples;
        let mut k = self.cfg.seq.unwrap_or(k_max).min(k_max);
        let mut active = vec![true; entries.len()];
        let mut values = self.rollout_values(&entries, &active, seed, first_active, 0..k);
        let mut sampled = k * entries.len();
        loop {
            let (best, decided) = eliminate_by_paired_rollouts(cands, &values, &mut active, &self.cfg);
            if decided || k >= k_max {
                if let Some(st) = &self.stats {
                    // Samples per candidate, averaged over the shortlist.
                    st.borrow_mut().rollout_samples += (sampled / entries.len()) as u64;
                }
                return best;
            }
            let next = (2 * k).min(k_max);
            let more = self.rollout_values(&entries, &active, seed, first_active, k..next);
            for ((v, m), &on) in values.iter_mut().zip(more).zip(&active) {
                if on {
                    v.extend(m);
                }
            }
            sampled += (next - k) * active.iter().filter(|&&a| a).count();
            k = next;
        }
    }

    /// Forced safe lock `lock` vs the rule-free alternatives (Phase 16
    /// semantics, see [`select_lock_trio`]) by `cfg.lock_k` full-game CRN
    /// rollouts each; the highest mean outcome wins, ties keep the lock.
    /// `collapse` merges plans sharing a phase-1 mark (active phase 1).
    /// `entry` completes the current turn for a (mark, post-state) candidate.
    #[allow(clippy::too_many_arguments)]
    fn adjudicate_lock(
        &self,
        state: &State,
        lock: Mark,
        rule_free: Decision,
        baseline: State,
        eval_opps: &[State],
        collapse: bool,
        entry: impl Fn(Option<Mark>, State) -> (Vec<State>, bool),
        seed: u64,
        first_active: usize,
    ) -> Option<Mark> {
        let Some(k) = self.cfg.lock_k else {
            return Some(lock);
        };
        let post_of = |m: Option<Mark>| match m {
            Some(m) => {
                let mut s = *state;
                s.apply_mark(m);
                s
            }
            None => baseline,
        };
        let compared: Vec<(Option<Mark>, State)> = match rule_free {
            Decision::Forced(alt) if alt == Some(lock) => return Some(lock),
            Decision::Forced(alt) => vec![(Some(lock), post_of(Some(lock))), (alt, post_of(alt))],
            Decision::Choices(plans) => {
                let states: Vec<State> = plans.iter().map(|(_, s)| *s).collect();
                let values = self.bot.evaluate_batch(&states, eval_opps);
                let mut cands: Vec<(Option<Mark>, State, f32)> = Vec::new();
                for ((m, s), &v) in plans.iter().zip(&values) {
                    match cands.iter_mut().find(|c| collapse && c.0 == *m) {
                        Some(c) if v > c.2 => *c = (*m, *s, v),
                        Some(_) => {}
                        None => cands.push((*m, *s, v)),
                    }
                }
                cands.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
                let marks: Vec<Option<Mark>> = cands.iter().map(|c| c.0).collect();
                match select_lock_trio(state, &marks, lock) {
                    Some(trio) => trio.into_iter().map(|i| (cands[i].0, cands[i].1)).collect(),
                    None => return Some(lock),
                }
            }
        };

        let entries: Vec<(Vec<State>, bool)> = compared.iter().map(|&(m, post)| entry(m, post)).collect();
        let finals = rollout_final_states(&self.bot, &entries, seed, k, first_active, self.cfg.stratify);
        let means: Vec<f32> = finals
            .iter()
            .map(|f| f.iter().map(|states| SimGame::outcome(states)).sum::<f32>() / f.len() as f32)
            .collect();
        let mut best = 0;
        for i in 1..means.len() {
            if means[i] > means[best] {
                best = i;
            }
        }
        if let Some(stats) = &self.stats {
            let mut s = stats.borrow_mut();
            s.lock_adjudications += 1;
            if best != 0 {
                s.lock_overrides += 1;
            }
        }
        compared[best].0
    }

    /// Gate bookkeeping. Returns true if search should run. Never searches
    /// with no opponents (solo is unsupported; the simulator's outcome and
    /// leader logic require at least one opponent).
    fn gate_and_record(&self, cands: &[Candidate], our: &State, opps: &[State]) -> Option<(bool, bool)> {
        if opps.is_empty() {
            return None;
        }
        if let Some(stats) = &self.stats {
            stats.borrow_mut().eligible += 1;
            stats.borrow_mut().gaps.push(cands[0].value - cands[1].value);
        }
        let value_posts: Vec<(f32, State)> = cands.iter().map(|c| (c.value, c.post)).collect();
        let (close, endgame) = gates_with(&value_posts, our, opps, self.cfg.gate);
        if let Some(stats) = &self.stats {
            let mut s = stats.borrow_mut();
            if close {
                s.gate_close += 1;
            }
            if endgame {
                s.gate_endgame += 1;
            }
        }
        if self.force || close || endgame {
            Some((close, endgame))
        } else {
            None
        }
    }

    fn record_search_result(&self, gated: bool, disagreed: bool) {
        if let Some(stats) = &self.stats {
            let mut s = stats.borrow_mut();
            s.searched += 1;
            if disagreed {
                s.disagreements += 1;
                if gated {
                    s.disagreements_gated += 1;
                }
            }
        }
    }
}

impl<B: WinProb> Strategy for SearchBot<B> {
    fn active_phase1(&mut self, state: &State, opp_states: &[State], dice: [u8; 6]) -> Option<Mark> {
        if let Some(stats) = &self.stats {
            stats.borrow_mut().active_decisions += 1;
        }
        let (decision, sim_opp) = active_phase1_choices(&self.bot, state, opp_states, dice);
        let plans = match decision {
            Decision::Forced(Some(lock)) if self.cfg.lock_k.is_some() && is_safe_lock(state, Some(lock)) => {
                let rule_free = phase1_plan_choices_with(state, &sim_opp, dice, false);
                return self.adjudicate_lock(
                    state,
                    lock,
                    rule_free,
                    *state,
                    &sim_opp,
                    true,
                    |m, _| phase1_entry(&self.bot, state, &sim_opp, dice, m),
                    context_seed(state, opp_states, dice),
                    1 % (opp_states.len() + 1),
                );
            }
            Decision::Forced(m) => return m,
            Decision::Choices(c) => c,
        };
        let states: Vec<State> = plans.iter().map(|(_, s)| *s).collect();
        let values = self.bot.evaluate_batch(&states, &sim_opp);

        // Collapse plans to distinct phase-1 marks: each keeps its best plan's
        // static value (for shortlisting) and that plan's end-state (for the
        // lock gate). The rollout re-decides phase 2 itself.
        let mut cands: Vec<Candidate> = Vec::new();
        for (i, (p1, _)) in plans.iter().enumerate() {
            match cands.iter_mut().find(|c| c.mark == *p1) {
                Some(c) if values[i] > c.value => {
                    c.value = values[i];
                    c.post = states[i];
                }
                Some(_) => {}
                None => cands.push(Candidate {
                    mark: *p1,
                    value: values[i],
                    post: states[i],
                }),
            }
        }
        cands.sort_by(|a, b| b.value.partial_cmp(&a.value).unwrap());

        let Some((close, endgame)) = (if cands.len() < 2 {
            None
        } else {
            self.gate_and_record(&cands, state, opp_states)
        }) else {
            return cands[0].mark;
        };

        // Entry per shortlisted phase-1 mark: opponents' (shared, simultaneous)
        // phase-1 marks are sim_opp; complete OUR turn deterministically.
        let shortlist = &cands[..cands.len().min(self.cfg.cands)];
        let entries: Vec<(Vec<State>, bool)> = shortlist
            .iter()
            .map(|c| phase1_entry(&self.bot, state, &sim_opp, dice, c.mark))
            .collect();

        let seed = context_seed(state, opp_states, dice);
        let pick = self.search_pick(shortlist, entries, seed, 1 % (opp_states.len() + 1));
        self.record_search_result(close || endgame, pick != 0);
        shortlist[pick].mark
    }

    fn active_phase2(&mut self, state: &State, opp_states: &[State], dice: [u8; 6], has_marked: bool) -> Option<Mark> {
        if let Some(stats) = &self.stats {
            stats.borrow_mut().active_decisions += 1;
        }
        let choices = match active_phase2_choices(state, opp_states, dice, has_marked) {
            Decision::Forced(Some(lock)) if self.cfg.lock_k.is_some() && is_safe_lock(state, Some(lock)) => {
                let mut baseline = *state;
                if !has_marked {
                    baseline.apply_strike();
                }
                let opp_best = opp_states.iter().map(|s| s.count_points()).max().unwrap_or(0);
                let marks = state.generate_color_moves(dice);
                let rule_free = mark_choices_with(state, &marks, baseline, opp_best, false);
                return self.adjudicate_lock(
                    state,
                    lock,
                    rule_free,
                    baseline,
                    opp_states,
                    false,
                    |_, post| phase2_entry(post, opp_states),
                    context_seed(state, opp_states, dice),
                    1 % (opp_states.len() + 1),
                );
            }
            Decision::Forced(m) => return m,
            Decision::Choices(c) => c,
        };
        let states: Vec<State> = choices.iter().map(|(_, s)| *s).collect();
        let values = self.bot.evaluate_batch(&states, opp_states);
        let mut cands: Vec<Candidate> = choices
            .iter()
            .zip(&values)
            .map(|((m, s), &v)| Candidate {
                mark: *m,
                value: v,
                post: *s,
            })
            .collect();
        cands.sort_by(|a, b| b.value.partial_cmp(&a.value).unwrap());

        let Some((close, endgame)) = (if cands.len() < 2 {
            None
        } else {
            self.gate_and_record(&cands, state, opp_states)
        }) else {
            return cands[0].mark;
        };

        // Entry per candidate: post state + lock propagation; turn is then over.
        let shortlist = &cands[..cands.len().min(self.cfg.cands)];
        let entries: Vec<(Vec<State>, bool)> = shortlist.iter().map(|c| phase2_entry(c.post, opp_states)).collect();

        let seed = context_seed(state, opp_states, dice);
        let pick = self.search_pick(shortlist, entries, seed, 1 % (opp_states.len() + 1));
        self.record_search_result(close || endgame, pick != 0);
        shortlist[pick].mark
    }

    fn passive_phase1(
        &mut self,
        state: &State,
        opp_states: &[State],
        dice: [u8; 6],
        active_player: usize,
    ) -> Option<Mark> {
        if !self.cfg.search_passive && self.cfg.lock_k.is_none() {
            return passive_phase1_impl(&self.bot, state, opp_states, dice);
        }
        let choices = match passive_phase1_choices(state, opp_states, dice) {
            Decision::Forced(Some(lock)) if self.cfg.lock_k.is_some() && is_safe_lock(state, Some(lock)) => {
                let white_sum = dice[0] + dice[1];
                let marks = state.generate_white_moves(white_sum);
                let opp_best = opp_best_phase1_score(opp_states, white_sum);
                let rule_free = mark_choices_with(state, &marks, *state, opp_best, false);
                return self.adjudicate_lock(
                    state,
                    lock,
                    rule_free,
                    *state,
                    opp_states,
                    false,
                    |m, _| passive_entry(&self.bot, state, opp_states, dice, active_player, m),
                    context_seed(state, opp_states, dice),
                    passive_first_active(active_player, opp_states.len() + 1),
                );
            }
            Decision::Forced(m) => return m,
            Decision::Choices(c) => c,
        };
        if !self.cfg.search_passive {
            return eval_decision(&self.bot, Decision::Choices(choices), opp_states);
        }
        let states: Vec<State> = choices.iter().map(|(_, s)| *s).collect();
        let values = self.bot.evaluate_batch(&states, opp_states);
        let mut cands: Vec<Candidate> = choices
            .iter()
            .zip(&values)
            .map(|((m, s), &v)| Candidate {
                mark: *m,
                value: v,
                post: *s,
            })
            .collect();
        cands.sort_by(|a, b| b.value.partial_cmp(&a.value).unwrap());

        let Some((close, endgame)) = (if cands.len() < 2 {
            None
        } else {
            self.gate_and_record(&cands, state, opp_states)
        }) else {
            return cands[0].mark;
        };

        // Entry per candidate: our mark, then the rest of the turn (other
        // passive players, the active player's phase 1 + 2) completed by the
        // bot; the rollout starts at the seat after the active player.
        let shortlist = &cands[..cands.len().min(self.cfg.cands)];
        let entries: Vec<(Vec<State>, bool)> = shortlist
            .iter()
            .map(|c| passive_entry(&self.bot, state, opp_states, dice, active_player, c.mark))
            .collect();

        let seed = context_seed(state, opp_states, dice);
        let first_active = passive_first_active(active_player, opp_states.len() + 1);
        let pick = self.search_pick(shortlist, entries, seed, first_active);
        self.record_search_result(close || endgame, pick != 0);
        if let Some(stats) = &self.stats {
            stats.borrow_mut().passive_searched += 1;
        }
        shortlist[pick].mark
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dqn::pair::{PairModelConfig, PairStrategy};
    use crate::dqn::MyBackend;
    use crate::state::Mark;
    use crate::strategy::Strategy;

    fn test_bot() -> PairStrategy {
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        PairStrategy::from_model(PairModelConfig::new().init::<MyBackend>(&device), device)
    }

    #[test]
    fn phi_matches_known_values() {
        assert!((phi(0.0) - 0.5).abs() < 1e-6);
        assert!((phi(1.0) - 0.8413).abs() < 1e-3);
        assert!((phi(1.96) - 0.9750).abs() < 1e-3);
        assert!((phi(-1.0) - (1.0 - phi(1.0))).abs() < 1e-6);
        assert!(phi(10.0) > 0.9999);
        assert!(phi(-10.0) < 0.0001);
    }

    #[test]
    fn context_seed_is_deterministic_and_sensitive() {
        let a = State::default();
        let mut b = State::default();
        b.apply_mark(Mark { row: 0, number: 5 });
        let dice = [1, 2, 3, 4, 5, 6];

        assert_eq!(context_seed(&a, &[b], dice), context_seed(&a, &[b], dice));
        assert_ne!(context_seed(&a, &[b], dice), context_seed(&b, &[a], dice));
        assert_ne!(context_seed(&a, &[b], dice), context_seed(&a, &[b], [2, 1, 3, 4, 5, 6]));
        // Sample/player streams are distinct.
        let s = context_seed(&a, &[b], dice);
        assert_ne!(sample_player_seed(s, 0, 0), sample_player_seed(s, 0, 1));
        assert_ne!(sample_player_seed(s, 0, 0), sample_player_seed(s, 1, 0));
    }

    #[test]
    fn search_decisions_are_deterministic() {
        // SearchBot holds no RNG state: repeating the same decision context on
        // the same instance must reproduce the same move (context-hashed dice).
        let mut bot = SearchBot::new(test_bot());
        bot.force = true; // exercise the search path regardless of gates
        let state = State::default();
        let opps = [State::default()];
        let dice = [3, 4, 2, 3, 5, 1];
        let a = bot.active_phase1(&state, &opps, dice);
        let b = bot.active_phase1(&state, &opps, dice);
        assert_eq!(a, b);
        let c = bot.active_phase2(&state, &opps, dice, false);
        let d = bot.active_phase2(&state, &opps, dice, false);
        assert_eq!(c, d);
    }

    #[test]
    fn gate_predicate_truth_table() {
        let mk = |value: f32, post: State| (value, post);
        let fresh = State::default();
        let mut locked_one = State::default();
        for n in 2..=6 {
            locked_one.apply_mark(Mark { row: 0, number: n });
        }
        locked_one.apply_mark(Mark { row: 0, number: 12 }); // locks red

        // Close gap, no endgame: close gate only.
        let cands = [mk(1.00, fresh), mk(0.95, fresh)];
        assert_eq!(gates(&cands, &fresh, &[fresh]), (true, false));
        // Wide gap, no endgame: nothing fires.
        let cands = [mk(1.00, fresh), mk(0.50, fresh)];
        assert_eq!(gates(&cands, &fresh, &[fresh]), (false, false));
        // Wide gap, opponent has a locked row: endgame fires.
        assert_eq!(gates(&cands, &fresh, &[locked_one]), (false, true));
        // Wide gap, our candidate would lock: endgame fires.
        let cands = [mk(1.00, locked_one), mk(0.50, fresh)];
        assert_eq!(gates(&cands, &fresh, &[fresh]), (false, true));
        // Wide gap, we sit at 3 strikes: endgame fires.
        let mut striked = State::default();
        for _ in 0..3 {
            striked.apply_strike();
        }
        let cands = [mk(1.00, fresh), mk(0.50, fresh)];
        assert_eq!(gates(&cands, &striked, &[fresh]), (false, true));
    }

    #[test]
    fn forced_decisions_bypass_search() {
        // A state with a safe lock available must return the lock without
        // search: 5 marks in red, white sum 12 completes it (first lock,
        // doesn't end the game).
        let mut bot = SearchBot::with_config(test_bot(), SearchConfig::legacy());
        bot.force = true;
        let mut state = State::default();
        for n in 2..=6 {
            state.apply_mark(Mark { row: 0, number: n });
        }
        let opps = [State::default()];
        let dice = [6, 6, 1, 1, 1, 1]; // white sum 12
        let m = bot.active_phase1(&state, &opps, dice);
        assert_eq!(m, Some(Mark { row: 0, number: 12 }));
    }

    #[test]
    fn fallback_when_gates_closed() {
        // Ungated decisions must be deterministic and legal (static path).
        let mut bot = SearchBot::new(test_bot());
        let state = State::default();
        let opps = [State::default()];
        let m1 = bot.active_phase2(&state, &opps, [3, 4, 2, 3, 5, 1], true);
        let m2 = bot.active_phase2(&state, &opps, [3, 4, 2, 3, 5, 1], true);
        assert_eq!(m1, m2);
    }

    #[test]
    fn passive_first_active_is_seat_after_active() {
        assert_eq!(passive_first_active(0, 2), 0); // 1v1: we act next
        assert_eq!(passive_first_active(0, 3), 2);
        assert_eq!(passive_first_active(1, 3), 0);
        assert_eq!(passive_first_active(2, 4), 0);
    }

    #[test]
    fn passive_entry_completes_the_active_players_turn() {
        let bot = test_bot();
        let us = State::default();
        let opps = [State::default(), State::default()];
        let dice = [3, 4, 2, 3, 5, 1];
        for active_player in 0..2 {
            let (all, ended) = passive_entry(&bot, &us, &opps, dice, active_player, None);
            assert!(!ended);
            // The active player (entry index active_player + 1) marked or struck.
            let a = all[active_player + 1];
            assert!(
                a.row_totals().iter().sum::<u8>() + a.strikes > 0,
                "active {active_player}: {all:?}"
            );
        }
    }

    #[test]
    fn passive_search_off_matches_static_pipeline() {
        let mut bot = SearchBot::with_config(test_bot(), SearchConfig::legacy());
        bot.force = true;
        let mut mid = State::default();
        mid.apply_mark(Mark { row: 0, number: 4 });
        mid.apply_mark(Mark { row: 3, number: 10 });
        for state in [State::default(), mid] {
            for dice in [[3, 4, 2, 3, 5, 1], [1, 1, 6, 6, 6, 6], [6, 5, 1, 2, 3, 4]] {
                let opps = [mid];
                let expected = passive_phase1_impl(&bot.bot, &state, &opps, dice);
                assert_eq!(bot.passive_phase1(&state, &opps, dice, 0), expected);
            }
        }
    }

    #[test]
    fn passive_search_runs_when_enabled() {
        let stats = std::rc::Rc::new(std::cell::RefCell::new(SearchStats::default()));
        let cfg = SearchConfig {
            search_passive: true,
            ..Default::default()
        };
        let mut bot = SearchBot::with_config(test_bot(), cfg);
        bot.force = true;
        bot.stats = Some(stats.clone());
        let state = State::default();
        let opps = [State::default()];
        let dice = [3, 4, 2, 3, 5, 1]; // white sum 7: four marks + skip
        let a = bot.passive_phase1(&state, &opps, dice, 0);
        let b = bot.passive_phase1(&state, &opps, dice, 0);
        assert_eq!(a, b);
        assert_eq!(stats.borrow().passive_searched, 2);
        // Also in a 3-player frame (active player = second opponent).
        let opps3 = [State::default(), State::default()];
        bot.passive_phase1(&state, &opps3, dice, 1);
        assert_eq!(stats.borrow().passive_searched, 3);
    }

    /// Red 2..6 marked: red 12 is a safe (first) lock.
    fn lockable_state() -> State {
        let mut s = State::default();
        for n in 2..=6 {
            s.apply_mark(Mark { row: 0, number: n });
        }
        s
    }

    #[test]
    fn lock_trio_picks_lock_best_alternative_and_runner_up_lock() {
        let mut s = lockable_state();
        for n in [2, 3, 4, 5, 6] {
            s.apply_mark(Mark { row: 1, number: n }); // yellow lockable too
        }
        let red12 = Mark { row: 0, number: 12 };
        let ylw12 = Mark { row: 1, number: 12 };
        let grn9 = Mark { row: 2, number: 9 };
        // Sorted by static value: yellow lock, skip, red lock, green 9.
        let sorted = [Some(ylw12), None, Some(red12), Some(grn9)];
        assert_eq!(select_lock_trio(&s, &sorted, red12), Some(vec![2, 1, 0]));
        // Lock not among the candidates, or only safe locks: nothing to compare.
        assert_eq!(select_lock_trio(&s, &[None, Some(grn9)], red12), None);
        assert_eq!(select_lock_trio(&s, &[Some(red12), Some(ylw12)], red12), None);
        // No runner-up lock: pair only.
        assert_eq!(
            select_lock_trio(&s, &[Some(grn9), Some(red12)], red12),
            Some(vec![1, 0])
        );
    }

    fn lock_bot(stats: &std::rc::Rc<std::cell::RefCell<SearchStats>>) -> SearchBot<PairStrategy> {
        let cfg = SearchConfig {
            lock_k: Some(16),
            ..Default::default()
        };
        let mut bot = SearchBot::with_config(test_bot(), cfg);
        bot.stats = Some(stats.clone());
        bot
    }

    #[test]
    fn forced_locks_are_adjudicated_at_every_site() {
        let stats = std::rc::Rc::new(std::cell::RefCell::new(SearchStats::default()));
        let mut bot = lock_bot(&stats);
        let state = lockable_state();
        let opps = [State::default()];
        let red12 = Some(Mark { row: 0, number: 12 });

        // Active phase 1: white sum 12.
        let p1 = [6, 6, 1, 1, 1, 1];
        let a = bot.active_phase1(&state, &opps, p1);
        assert_eq!(a, bot.active_phase1(&state, &opps, p1), "deterministic");
        assert_eq!(stats.borrow().lock_adjudications, 2);
        // Active phase 2: white 6 + red 6.
        let p2 = [6, 1, 6, 1, 1, 1];
        let b = bot.active_phase2(&state, &opps, p2, true);
        assert_eq!(stats.borrow().lock_adjudications, 3);
        // Passive: white sum 12.
        bot.passive_phase1(&state, &opps, p1, 0);
        assert_eq!(stats.borrow().lock_adjudications, 4);
        let _ = (a, b, red12);
        let s = stats.borrow();
        assert!(s.lock_overrides <= s.lock_adjudications);
    }

    #[test]
    fn locks_stay_forced_when_adjudication_is_off() {
        let mut bot = SearchBot::with_config(test_bot(), SearchConfig::legacy());
        let state = lockable_state();
        let opps = [State::default()];
        let red12 = Some(Mark { row: 0, number: 12 });
        assert_eq!(bot.active_phase1(&state, &opps, [6, 6, 1, 1, 1, 1]), red12);
        assert_eq!(bot.active_phase2(&state, &opps, [6, 1, 6, 1, 1, 1], true), red12);
        assert_eq!(bot.passive_phase1(&state, &opps, [6, 6, 1, 1, 1, 1], 0), red12);
    }

    #[test]
    fn stratified_seeds_balance_each_players_first_whites() {
        let decision = 0xC0FFEE;
        for player in 0..4 {
            let mut counts = [0u32; 36];
            for sample in 0..144 {
                let seed = stratified_player_seed(decision, sample, player);
                assert_eq!(seed, stratified_player_seed(decision, sample, player), "deterministic");
                let first = SmallRng::seed_from_u64(seed).roll();
                counts[white_stratum(first)] += 1;
            }
            assert!(counts.iter().all(|&c| c == 4), "player {player}: {counts:?}");
        }
        // Players follow different stratum schedules.
        let first_whites = |p: usize| -> Vec<usize> {
            (0..36)
                .map(|k| white_stratum(SmallRng::seed_from_u64(stratified_player_seed(decision, k, p)).roll()))
                .collect()
        };
        assert_ne!(first_whites(0), first_whites(1));
    }

    #[test]
    fn stratified_search_is_deterministic() {
        let cfg = SearchConfig {
            k_samples: 36,
            stratify: true,
            lock_k: None,
            ..Default::default()
        };
        let mut bot = SearchBot::with_config(test_bot(), cfg);
        bot.force = true;
        let state = State::default();
        let opps = [State::default()];
        let dice = [3, 4, 2, 3, 5, 1];
        assert_eq!(
            bot.active_phase1(&state, &opps, dice),
            bot.active_phase1(&state, &opps, dice)
        );
    }

    #[test]
    #[should_panic(expected = "multiple of 36")]
    fn stratify_requires_whole_strata() {
        let cfg = SearchConfig {
            k_samples: 128,
            stratify: true,
            ..Default::default()
        };
        SearchBot::with_config(test_bot(), cfg);
    }

    #[test]
    fn search_config_parses_blend_and_gate() {
        let cfg = SearchConfig::parse("blend=0.4,gate=0.3").unwrap();
        assert_eq!(cfg.blend, Some(0.4));
        assert_eq!(cfg.gate, 0.3);
        assert_eq!(SearchConfig::parse(&cfg.to_string()), Ok(cfg));
        // Production blends at 0.7 (Phase 25) with the 0.15 gate.
        assert_eq!(SearchConfig::default().blend, Some(BLEND));
        assert_eq!(SearchConfig::default().gate, GATE_MARGIN);
        assert_eq!(SearchConfig::parse("noblend").unwrap().blend, None);
        assert!(SearchConfig::parse("blend=1.5").is_err());
        assert!(SearchConfig::parse("blend=x").is_err());
        assert!(SearchConfig::parse("gate=-1").is_err());
    }

    #[test]
    fn blend_zero_reproduces_static_pick() {
        // λ = 0 scores the shortlist by Φ(static z) alone, so forced search
        // must return exactly the static pipeline's move.
        use crate::strategy::bot_impl::{active_phase1_impl, active_phase2_impl};
        let cfg = SearchConfig {
            blend: Some(0.0),
            ..SearchConfig::legacy()
        };
        let mut bot = SearchBot::with_config(test_bot(), cfg);
        bot.force = true;
        let mut mid = State::default();
        mid.apply_mark(Mark { row: 0, number: 4 });
        mid.apply_mark(Mark { row: 3, number: 10 });
        for state in [State::default(), mid] {
            for dice in [[3, 4, 2, 3, 5, 1], [1, 1, 6, 6, 6, 6], [6, 5, 1, 2, 3, 4]] {
                let opps = [mid];
                let p1 = active_phase1_impl(&bot.bot, &state, &opps, dice);
                assert_eq!(bot.active_phase1(&state, &opps, dice), p1);
                for has_marked in [false, true] {
                    let p2 = active_phase2_impl(&bot.bot, &state, &opps, dice, has_marked);
                    assert_eq!(bot.active_phase2(&state, &opps, dice, has_marked), p2);
                }
            }
        }
    }

    #[test]
    fn search_config_parses_seq_stop() {
        let cfg = SearchConfig::parse("seq=32,stop=inf").unwrap();
        assert_eq!((cfg.seq, cfg.stop), (Some(32), f32::INFINITY));
        assert_eq!(SearchConfig::parse(&cfg.to_string()), Ok(cfg));
        let cfg = SearchConfig::parse("seq=16,stop=1.5,cands=4").unwrap();
        assert_eq!(cfg.cands, 4);
        assert!(SearchConfig::parse("cands=1").is_err());
        assert_eq!(SearchConfig::parse(&cfg.to_string()), Ok(cfg));
        assert!(SearchConfig::parse("seq=0").is_err());
        // Production samples sequentially; noseq gives a fixed k.
        assert_eq!(SearchConfig::default().seq, Some(SEQ_START));
        assert_eq!(SearchConfig::parse("k=128,noseq").unwrap().seq, None);
        assert!(SearchConfig::parse("stop=-1").is_err());
    }

    fn cand(value: f32) -> Candidate {
        Candidate {
            mark: None,
            value,
            post: State::default(),
        }
    }

    fn pick(cands: &[Candidate], values: &[Vec<f32>], cfg: &SearchConfig) -> (usize, bool) {
        eliminate_by_paired_rollouts(cands, values, &mut vec![true; values.len()], cfg)
    }

    #[test]
    fn paired_pick_blends_and_stops() {
        let cands = [cand(0.1), cand(0.0)];
        let noblend = SearchConfig {
            blend: None,
            ..SearchConfig::default()
        };
        // Constant paired difference +0.1: zero SE, decisive, rollouts win
        // unless the static term outweighs them.
        let values = vec![vec![0.5; 8], vec![0.6; 8]];
        assert_eq!(pick(&cands, &values, &noblend), (1, true));
        let static_only = SearchConfig {
            blend: Some(0.0),
            ..noblend
        };
        assert_eq!(pick(&cands, &values, &static_only).0, 0);
        // Noisy difference with mean +0.01 (SE ≈ 0.038): not decisive at
        // stop=2; the rollouts alone prefer 1, blend 0.7 keeps the static
        // pick (ΔΦ ≈ −0.04).
        let a: Vec<f32> = (0..8).map(|i| if i % 2 == 0 { 0.0 } else { 1.0 }).collect();
        let b: Vec<f32> = a
            .iter()
            .enumerate()
            .map(|(i, x)| x + if i % 2 == 0 { 0.11 } else { -0.09 })
            .collect();
        let values = vec![a, b];
        assert_eq!(pick(&cands, &values, &noblend), (1, false));
        assert_eq!(pick(&cands, &values, &SearchConfig::default()), (0, false));
        let lax = SearchConfig { stop: 0.1, ..noblend };
        assert_eq!(pick(&cands, &values, &lax), (1, true));
    }

    #[test]
    fn elimination_drops_only_decisively_worse_candidates() {
        let cands = [cand(0.1), cand(0.0), cand(-0.1)];
        let noblend = SearchConfig {
            blend: None,
            ..SearchConfig::default()
        };
        // 1 is a noisy +0.01 over 0 (undecided); 2 is a constant −0.2.
        let a: Vec<f32> = (0..8).map(|i| if i % 2 == 0 { 0.0 } else { 1.0 }).collect();
        let b: Vec<f32> = a
            .iter()
            .enumerate()
            .map(|(i, x)| x + if i % 2 == 0 { 0.11 } else { -0.09 })
            .collect();
        let c: Vec<f32> = a.iter().map(|x| x - 0.2).collect();
        let values = vec![a, b, c];
        let mut active = vec![true; 3];
        assert_eq!(
            eliminate_by_paired_rollouts(&cands, &values, &mut active, &noblend),
            (1, false)
        );
        assert_eq!(active, [true, true, false]);
        // Already-eliminated entries are ignored (stale/empty samples).
        let values = vec![values[0].clone(), values[1].clone(), Vec::new()];
        assert_eq!(
            eliminate_by_paired_rollouts(&cands, &values, &mut active, &noblend),
            (1, false)
        );
        // Infinite stop never eliminates.
        let never = SearchConfig {
            stop: f32::INFINITY,
            ..noblend
        };
        let mut active = vec![true; 3];
        let values = vec![
            values[0].clone(),
            values[1].clone(),
            values[0].iter().map(|x| x - 0.2).collect(),
        ];
        assert_eq!(
            eliminate_by_paired_rollouts(&cands, &values, &mut active, &never),
            (1, false)
        );
        assert_eq!(active, [true; 3]);
    }

    #[test]
    fn rollout_sample_ranges_compose() {
        // Sequential sampling relies on samples [0, a) ++ [a, b) == [0, b).
        let bot = SearchBot::new(test_bot());
        let a = State::default();
        let mut marked = a;
        marked.apply_mark(Mark { row: 0, number: 5 });
        let entries = vec![phase2_entry(a, &[a]), phase2_entry(marked, &[a])];
        let on = [true, true];
        let whole = bot.rollout_values(&entries, &on, 11, 1, 0..24);
        let mut split = bot.rollout_values(&entries, &on, 11, 1, 0..8);
        for (v, m) in split.iter_mut().zip(bot.rollout_values(&entries, &on, 11, 1, 8..24)) {
            v.extend(m);
        }
        assert_eq!(whole, split);
        // An inactive entry is skipped without changing the others' samples.
        let only1 = bot.rollout_values(&entries, &[false, true], 11, 1, 0..24);
        assert!(only1[0].is_empty());
        assert_eq!(only1[1], whole[1]);
    }

    #[test]
    fn seq_with_infinite_stop_matches_fixed_k() {
        // Never decisive → doubles to k_samples and must pick exactly as
        // the fixed-k search does, for 2 and 3 candidates.
        let base = SearchConfig {
            k_samples: 32,
            ..SearchConfig::legacy()
        };
        let seq = SearchConfig {
            seq: Some(4),
            stop: f32::INFINITY,
            ..base
        };
        // One bot (test_bot() is randomly initialised), config swapped per call.
        let mut bot = SearchBot::with_config(test_bot(), base);
        bot.force = true;
        let mut mid = State::default();
        mid.apply_mark(Mark { row: 0, number: 4 });
        mid.apply_mark(Mark { row: 3, number: 10 });
        for state in [State::default(), mid] {
            for dice in [[3, 4, 2, 3, 5, 1], [1, 1, 6, 6, 6, 6], [6, 5, 1, 2, 3, 4]] {
                let opps = [mid];
                for cands in [2, 3] {
                    let mut picks = Vec::new();
                    for cfg in [base, seq] {
                        bot.cfg = SearchConfig { cands, ..cfg };
                        picks.push((
                            bot.active_phase1(&state, &opps, dice),
                            bot.active_phase2(&state, &opps, dice, false),
                        ));
                    }
                    assert_eq!(picks[0], picks[1]);
                }
            }
        }
    }

    #[test]
    fn gate_margin_is_configurable() {
        let fresh = State::default();
        let cands = [(1.00, fresh), (0.80, fresh)];
        assert_eq!(gates_with(&cands, &fresh, &[fresh], GATE_MARGIN), (false, false));
        assert_eq!(gates_with(&cands, &fresh, &[fresh], 0.3), (true, false));
    }

    #[test]
    fn search_config_parses_and_round_trips() {
        assert_eq!(SearchConfig::parse(""), Ok(SearchConfig::default()));
        let cfg = SearchConfig::parse("passive, lock=256,strat,k=144,noseq").unwrap();
        assert_eq!(
            cfg,
            SearchConfig {
                k_samples: 144,
                search_passive: true,
                lock_k: Some(256),
                stratify: true,
                blend: Some(BLEND),
                seq: None,
                stop: SEQ_STOP,
                gate: GATE_MARGIN,
                cands: K_CANDIDATES,
            }
        );
        assert_eq!(SearchConfig::parse(&cfg.to_string()), Ok(cfg));
        // Production default (Phase 26) and the pre-Phase-21 legacy config.
        assert_eq!(
            SearchConfig::default().to_string(),
            "k=512,passive,lock=256,blend=0.7,seq=32"
        );
        assert_eq!(
            SearchConfig::legacy().to_string(),
            "k=128,nopassive,nolock,noblend,noseq"
        );
        assert_eq!(
            SearchConfig::parse("k=128,nopassive,nolock,noblend,noseq"),
            Ok(SearchConfig::legacy())
        );
        assert_eq!(
            SearchConfig::parse(&SearchConfig::legacy().to_string()),
            Ok(SearchConfig::legacy())
        );
        assert!(SearchConfig::parse("bogus").is_err());
        assert!(SearchConfig::parse("k=abc").is_err());
    }

    #[test]
    fn rollout_final_states_full_games_deterministic() {
        let bot = test_bot();
        let a = State::default();
        let b = State::default();
        let entries = vec![phase2_entry(a, &[b]), {
            let mut marked = a;
            marked.apply_mark(Mark { row: 0, number: 5 });
            phase2_entry(marked, &[b])
        }];
        let r1 = rollout_final_states(&bot, &entries, 7, 8, 1 % 2, false);
        let r2 = rollout_final_states(&bot, &entries, 7, 8, 1 % 2, false);
        assert_eq!(r1.len(), 2);
        assert_eq!(r1[0].len(), 8);
        // Every rollout reached a real game end and is reproducible.
        for (e1, e2) in r1.iter().zip(&r2) {
            for (s1, s2) in e1.iter().zip(e2) {
                assert!(SimGame::game_over(s1));
                assert_eq!(
                    s1.iter().map(|s| s.count_points()).collect::<Vec<_>>(),
                    s2.iter().map(|s| s.count_points()).collect::<Vec<_>>()
                );
            }
        }
    }

    #[test]
    fn stats_are_recorded() {
        let stats = std::rc::Rc::new(std::cell::RefCell::new(SearchStats::default()));
        let mut bot = SearchBot::new(test_bot());
        bot.force = true;
        bot.stats = Some(stats.clone());
        let state = State::default();
        let opps = [State::default()];
        bot.active_phase1(&state, &opps, [3, 4, 2, 3, 5, 1]);
        let s = stats.borrow();
        assert_eq!(s.active_decisions, 1);
        assert!(s.eligible <= 1);
        assert_eq!(s.gaps.len() as u32, s.eligible);
    }
}
