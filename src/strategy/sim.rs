//! Lightweight game simulator for decision-time search. Replicates the
//! mechanics of `Game::play` (simultaneous phase 1 on pre-snapshots, lock
//! propagation, game-over ordering, phase 2) on plain `State`s, entrable
//! mid-turn, with all players' decisions driven by a `Bot`.

use super::bot_impl::{
    active_phase1_impl, active_phase2_choices, eval_decision, eval_decision_next, passive_phase1_choices,
    phase1_plan_choices, Decision,
};
use super::{Bot, Next};
use crate::game::DiceSource;
use crate::state::{Mark, State};
use rand::rngs::SmallRng;

/// Decision fidelity inside simulations.
#[derive(Clone, Copy, PartialEq)]
pub enum Fidelity {
    /// Identical to the blanket Strategy impl (incl. nested opponent
    /// simulation in active phase 1). Used by the Game-equivalence test.
    Full,
    /// Active phase 1 evaluates plans against current opponent states — no
    /// nested opponent simulation. Used inside search rollouts (one net
    /// evaluation per decision; sims are approximations).
    Lite,
}

pub struct SimGame {
    pub states: Vec<State>,
    /// Player to take the next turn. Player 0 is the searcher's seat.
    pub active: usize,
    /// Per-player dice streams (mirrors `Game`, where each player rolls).
    pub rngs: Vec<SmallRng>,
    pub over: bool,
}

impl SimGame {
    pub fn n(&self) -> usize {
        self.states.len()
    }

    pub fn game_over(states: &[State]) -> bool {
        states.iter().any(|s| s.strikes >= 4) || states.iter().map(|s| s.count_locked()).max().unwrap() >= 2
    }

    pub fn propagate_locks(states: &mut [State]) {
        let mut locked = [false; 4];
        for s in states.iter() {
            let l = s.locked();
            for row in 0..4 {
                locked[row] |= l[row];
            }
        }
        for s in states.iter_mut() {
            s.lock(locked);
        }
    }

    /// Turn-ordered opponent view for player `j` over `states`.
    pub fn opp_view(states: &[State], j: usize) -> Vec<State> {
        let n = states.len();
        (1..n).map(|off| states[(j + off) % n]).collect()
    }

    /// Exact outcome for player 0: 1.0 win, 0.5 top tie, 0.0 loss.
    pub fn outcome(states: &[State]) -> f32 {
        let our = states[0].count_points();
        let best_opp = states[1..].iter().map(|s| s.count_points()).max().unwrap();
        match our.cmp(&best_opp) {
            std::cmp::Ordering::Greater => 1.0,
            std::cmp::Ordering::Equal => 0.5,
            std::cmp::Ordering::Less => 0.0,
        }
    }
}

/// Advance one sim by one full turn (roll, phase 1 all players, locks,
/// game-over, phase 2, locks, game-over, advance). Reference implementation —
/// the lockstep driver must stay decision-equivalent to this.
pub fn play_sim_turn(bot: &impl Bot, sim: &mut SimGame, fidelity: Fidelity) {
    if sim.over {
        return;
    }
    let n = sim.n();
    let dice = sim.rngs[sim.active].roll();
    let snapshot = sim.states.clone();

    // Phase 1: all players decide against pre-phase1 snapshots.
    let mut marks: Vec<Option<Mark>> = Vec::with_capacity(n);
    for j in 0..n {
        let view = SimGame::opp_view(&snapshot, j);
        let m = if j == sim.active {
            match fidelity {
                Fidelity::Full => active_phase1_impl(bot, &snapshot[j], &view, dice),
                Fidelity::Lite => eval_decision(bot, phase1_plan_choices(&snapshot[j], &view, dice), &view),
            }
        } else {
            eval_decision_next(bot, passive_phase1_choices(&snapshot[j], &view, dice), &view, Next::Me)
        };
        marks.push(m);
    }
    let has_marked = marks[sim.active].is_some();
    for (j, m) in marks.iter().enumerate() {
        if let Some(m) = m {
            sim.states[j].apply_mark(*m);
        }
    }
    SimGame::propagate_locks(&mut sim.states);
    if SimGame::game_over(&sim.states) {
        sim.over = true;
        return;
    }

    // Phase 2: active player only.
    let view = SimGame::opp_view(&sim.states, sim.active);
    let decision = active_phase2_choices(&sim.states[sim.active], &view, dice, has_marked);
    match eval_decision(bot, decision, &view) {
        Some(m) => sim.states[sim.active].apply_mark(m),
        None if !has_marked => sim.states[sim.active].apply_strike(),
        None => {}
    }
    SimGame::propagate_locks(&mut sim.states);
    if SimGame::game_over(&sim.states) {
        sim.over = true;
        return;
    }

    sim.active = (sim.active + 1) % n;
}

/// Advances many sims in lockstep, batching each phase's net evaluations of
/// ALL live sims into one `evaluate_batch_multi` call. Decisions inside are
/// Lite-fidelity (see [`Fidelity::Lite`]).
pub struct BatchedRollouts<'a, B: Bot> {
    pub bot: &'a B,
    pub sims: Vec<SimGame>,
}

/// One unevaluated decision gathered from a sim.
struct PendingChoice {
    sim: usize,
    player: usize,
    cands: Vec<(Option<Mark>, State)>,
    states: Vec<State>,
    view: Vec<State>,
    /// Who moves next after this choice: `Me` for a passive player.
    next: Next,
}

impl<'a, B: Bot> BatchedRollouts<'a, B> {
    pub fn new(bot: &'a B, sims: Vec<SimGame>) -> Self {
        Self { bot, sims }
    }

    pub fn all_over(&self) -> bool {
        self.sims.iter().all(|s| s.over)
    }

    /// Resolve a batch of pending choices with one multi-group evaluation.
    /// Returns (sim, player, chosen mark) triples.
    fn resolve(&self, pending: Vec<PendingChoice>) -> Vec<(usize, usize, Option<Mark>)> {
        let groups: Vec<(&[State], &[State], Next)> = pending
            .iter()
            .map(|p| (p.states.as_slice(), p.view.as_slice(), p.next))
            .collect();
        let values = self.bot.evaluate_batch_multi_next(&groups);
        pending
            .into_iter()
            .zip(values)
            .map(|(p, vals)| (p.sim, p.player, p.cands[super::bot_impl::argmax(&vals)].0))
            .collect()
    }

    /// Advance every live sim by one full turn.
    pub fn step_turn(&mut self) {
        let live: Vec<usize> = (0..self.sims.len()).filter(|&i| !self.sims[i].over).collect();
        if live.is_empty() {
            return;
        }

        // ---- Phase 1: roll + gather all players' decisions ----
        let mut dice_of: Vec<[u8; 6]> = Vec::with_capacity(live.len());
        let mut snapshots: Vec<Vec<State>> = Vec::with_capacity(live.len());
        for &si in &live {
            let sim = &mut self.sims[si];
            let active = sim.active;
            dice_of.push(sim.rngs[active].roll());
            snapshots.push(sim.states.clone());
        }

        let mut marks: Vec<Vec<Option<Mark>>> = live.iter().map(|&si| vec![None; self.sims[si].n()]).collect();
        let mut pending: Vec<PendingChoice> = Vec::new();
        for (li, &si) in live.iter().enumerate() {
            let sim = &self.sims[si];
            let n = sim.n();
            let dice = dice_of[li];
            for j in 0..n {
                let view = SimGame::opp_view(&snapshots[li], j);
                let decision = if j == sim.active {
                    phase1_plan_choices(&snapshots[li][j], &view, dice)
                } else {
                    passive_phase1_choices(&snapshots[li][j], &view, dice)
                };
                match decision {
                    Decision::Forced(m) => marks[li][j] = m,
                    Decision::Choices(cands) => {
                        let states: Vec<State> = cands.iter().map(|(_, s)| *s).collect();
                        pending.push(PendingChoice {
                            sim: li,
                            player: j,
                            cands,
                            states,
                            view,
                            next: if j == sim.active { Next::Opp } else { Next::Me },
                        });
                    }
                }
            }
        }
        for (li, j, m) in self.resolve(pending) {
            marks[li][j] = m;
        }

        // ---- Apply phase 1, locks, game-over ----
        let mut has_marked: Vec<bool> = Vec::with_capacity(live.len());
        for (li, &si) in live.iter().enumerate() {
            let sim = &mut self.sims[si];
            has_marked.push(marks[li][sim.active].is_some());
            for (j, m) in marks[li].iter().enumerate() {
                if let Some(m) = m {
                    sim.states[j].apply_mark(*m);
                }
            }
            SimGame::propagate_locks(&mut sim.states);
            if SimGame::game_over(&sim.states) {
                sim.over = true;
            }
        }

        // ---- Phase 2: active players of still-live sims ----
        let mut pending: Vec<PendingChoice> = Vec::new();
        let mut phase2_marks: Vec<Option<Option<Mark>>> = vec![None; live.len()]; // outer None = pending
        for (li, &si) in live.iter().enumerate() {
            let sim = &self.sims[si];
            if sim.over {
                continue;
            }
            let view = SimGame::opp_view(&sim.states, sim.active);
            match active_phase2_choices(&sim.states[sim.active], &view, dice_of[li], has_marked[li]) {
                Decision::Forced(m) => phase2_marks[li] = Some(m),
                Decision::Choices(cands) => {
                    let states: Vec<State> = cands.iter().map(|(_, s)| *s).collect();
                    pending.push(PendingChoice {
                        sim: li,
                        player: sim.active,
                        cands,
                        states,
                        view,
                        next: Next::Opp,
                    });
                }
            }
        }
        for (li, _, m) in self.resolve(pending) {
            phase2_marks[li] = Some(m);
        }

        for (li, &si) in live.iter().enumerate() {
            let sim = &mut self.sims[si];
            if sim.over {
                continue;
            }
            match phase2_marks[li].expect("phase 2 decision missing") {
                Some(m) => sim.states[sim.active].apply_mark(m),
                None if !has_marked[li] => sim.states[sim.active].apply_strike(),
                None => {}
            }
            SimGame::propagate_locks(&mut sim.states);
            if SimGame::game_over(&sim.states) {
                sim.over = true;
                continue;
            }
            sim.active = (sim.active + 1) % sim.n();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dqn::pair::{PairModelConfig, PairStrategy};
    use crate::dqn::MyBackend;
    use crate::game::{Game, Player};
    use rand::SeedableRng;
    use std::sync::Arc;

    #[test]
    fn sim_turn_loop_matches_game_play() {
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let model = Arc::new(PairModelConfig::new().init::<MyBackend>(&device));

        for &n in &[2usize, 3, 4] {
            for seed in 0..3u64 {
                // Reference: real Game with PairStrategy players.
                let players: Vec<Player> = (0..n)
                    .map(|i| {
                        Player::new(
                            Box::new(PairStrategy::from_shared(model.clone(), device)),
                            Box::new(SmallRng::seed_from_u64(1000 * seed + i as u64)),
                        )
                    })
                    .collect();
                let mut game = Game::new(players);
                game.play();

                // Sim: same model, same per-player dice streams, full fidelity.
                let bot = PairStrategy::from_shared(model.clone(), device);
                let mut sim = SimGame {
                    states: vec![State::default(); n],
                    active: 0,
                    rngs: (0..n)
                        .map(|i| SmallRng::seed_from_u64(1000 * seed + i as u64))
                        .collect(),
                    over: false,
                };
                let mut guard = 0;
                while !sim.over {
                    play_sim_turn(&bot, &mut sim, Fidelity::Full);
                    guard += 1;
                    assert!(guard < 500, "sim did not terminate");
                }

                for (i, p) in game.players.iter().enumerate() {
                    assert_eq!(
                        sim.states[i], p.state,
                        "n={n} seed={seed} player={i}: sim diverged from Game::play"
                    );
                }
            }
        }

        // Trained weights play lock-seeking games, covering the 2-locks
        // game-over path and cross-player lock propagation (the random-init
        // model above only ever ends games by strikes).
        let trained = PairStrategy::load("pair_model");
        let model = trained.model.clone();
        let mut saw_lock_end = false;
        for &n in &[2usize, 3, 4] {
            for seed in 0..2u64 {
                let players: Vec<Player> = (0..n)
                    .map(|i| {
                        Player::new(
                            Box::new(PairStrategy::from_shared(model.clone(), device)),
                            Box::new(SmallRng::seed_from_u64(5000 * seed + i as u64)),
                        )
                    })
                    .collect();
                let mut game = Game::new(players);
                game.play();

                let bot = PairStrategy::from_shared(model.clone(), device);
                let mut sim = SimGame {
                    states: vec![State::default(); n],
                    active: 0,
                    rngs: (0..n)
                        .map(|i| SmallRng::seed_from_u64(5000 * seed + i as u64))
                        .collect(),
                    over: false,
                };
                let mut guard = 0;
                while !sim.over {
                    play_sim_turn(&bot, &mut sim, Fidelity::Full);
                    guard += 1;
                    assert!(guard < 500, "sim did not terminate");
                }
                saw_lock_end |= sim.states.iter().map(|s| s.count_locked()).max().unwrap() >= 2;
                for (i, p) in game.players.iter().enumerate() {
                    assert_eq!(sim.states[i], p.state, "trained n={n} seed={seed} player={i}");
                }
            }
        }
        assert!(saw_lock_end, "no trained game ended via locks — lock path not covered");
    }

    #[test]
    fn lockstep_driver_matches_sequential_lite_sims() {
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let model = Arc::new(PairModelConfig::new().init::<MyBackend>(&device));
        let bot = PairStrategy::from_shared(model, device);

        let make_sims = || -> Vec<SimGame> {
            (0..8u64)
                .map(|k| SimGame {
                    states: vec![State::default(); 2],
                    active: 0,
                    rngs: (0..2).map(|i| SmallRng::seed_from_u64(7000 + 10 * k + i)).collect(),
                    over: false,
                })
                .collect()
        };

        // Sequential reference (Lite fidelity).
        let mut seq = make_sims();
        for _ in 0..30 {
            for sim in seq.iter_mut() {
                play_sim_turn(&bot, sim, Fidelity::Lite);
            }
        }

        // Lockstep.
        let mut batched = BatchedRollouts::new(&bot, make_sims());
        for _ in 0..30 {
            batched.step_turn();
        }

        for (a, b) in seq.iter().zip(&batched.sims) {
            assert_eq!(a.over, b.over);
            assert_eq!(a.active, b.active);
            assert_eq!(a.states, b.states);
        }

        // The over-mask path must actually be exercised: with 30 turns the
        // random-init bot strikes out some sims at staggered times.
        assert!(
            batched.sims.iter().any(|s| s.over),
            "no sim finished — over-mask path untested"
        );
    }

    use crate::strategy::recorder::Recorder;
    use crate::strategy::Next;

    /// Seat 0 (active) has no strikes, seat 1 (passive) two: every
    /// candidate with two or more strikes is the passive player's.
    fn striped(seed: u64) -> SimGame {
        let mut passive = State::default();
        passive.apply_strike();
        passive.apply_strike();
        SimGame {
            states: vec![State::default(), passive],
            active: 0,
            rngs: (0..2).map(|p| SmallRng::seed_from_u64(seed * 2 + p)).collect(),
            over: false,
        }
    }

    fn assert_frames(calls: &[crate::strategy::recorder::Call]) {
        assert!(calls.iter().any(|c| c.next == Next::Me) && calls.iter().any(|c| c.next == Next::Opp));
        for c in calls {
            let passive = c.strikes >= 2;
            assert_eq!(c.next, if passive { Next::Me } else { Next::Opp }, "{c:?}");
        }
    }

    #[test]
    fn a_sim_turn_evaluates_the_passive_player_with_it_moving_next() {
        for fidelity in [Fidelity::Lite, Fidelity::Full] {
            for seed in 0..6 {
                let bot = Recorder::default();
                play_sim_turn(&bot, &mut striped(seed), fidelity);
                assert_frames(&bot.take());
            }
        }
    }

    #[test]
    fn batched_rollouts_evaluate_the_passive_player_with_it_moving_next() {
        let bot = Recorder::default();
        let mut driver = BatchedRollouts::new(&bot, (0..8).map(striped).collect());
        driver.step_turn();
        assert_frames(&bot.take());
    }
}
