use super::{Bot, Next};
use crate::state::{Mark, State};
use std::cmp::Ordering;

fn prune_dominated<T>(items: &mut Vec<T>, state_of: impl Fn(&T) -> &State) {
    let n = items.len();
    let mut dominated = vec![false; n];
    for i in 0..n {
        if dominated[i] {
            continue;
        }
        for j in (i + 1)..n {
            if dominated[j] {
                continue;
            }
            match state_of(&items[i]).partial_cmp(state_of(&items[j])) {
                Some(Ordering::Greater) => dominated[j] = true,
                Some(Ordering::Less) => {
                    dominated[i] = true;
                    break;
                }
                _ => {}
            }
        }
    }
    let mut idx = 0;
    items.retain(|_| {
        let keep = !dominated[idx];
        idx += 1;
        keep
    });
}

pub(crate) fn opp_best_phase1_score(opp_states: &[State], white_sum: u8) -> isize {
    opp_states
        .iter()
        .map(|opp| {
            let base = opp.count_points();
            opp.generate_white_moves(white_sum)
                .iter()
                .map(|&m| {
                    let mut s = *opp;
                    s.apply_mark(m);
                    s.count_points()
                })
                .max()
                .unwrap_or(base)
                .max(base)
        })
        .max()
        .unwrap_or(0)
}

/// If any mark locks a row without ending the game, force it.
pub(crate) fn find_safe_lock(state: &State, marks: &[Mark]) -> Option<Mark> {
    marks
        .iter()
        .copied()
        .filter(|&m| state.would_lock_row(m))
        .filter(|&m| {
            let mut s = *state;
            s.apply_mark(m);
            !s.would_end_game()
        })
        .max_by_key(|&m| {
            let mut s = *state;
            s.apply_mark(m);
            s.count_points()
        })
}

/// Is `mark` a safe lock from `state` (locks a row without ending the game)?
pub(crate) fn is_safe_lock(state: &State, mark: Option<Mark>) -> bool {
    match mark {
        Some(m) if state.would_lock_row(m) => {
            let mut s = *state;
            s.apply_mark(m);
            !s.would_end_game()
        }
        _ => false,
    }
}

/// Outcome of the pure-logic half of a decision pipeline: either the meta
/// rules fully determine the move, or a filtered+pruned candidate list
/// remains for value-based selection.
/// Pub for analysis examples (see examples/divergence.rs).
pub enum Decision {
    Forced(Option<Mark>),
    /// (move, post-move state) pairs; `None` = skip/baseline.
    Choices(Vec<(Option<Mark>, State)>),
}

/// Index of the maximum value, matching `Iterator::max_by` semantics
/// (last maximum wins) so refactored paths pick identical moves.
pub fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap()
        .0
}

pub fn eval_decision(bot: &impl Bot, decision: Decision, eval_opps: &[State]) -> Option<Mark> {
    eval_decision_next(bot, decision, eval_opps, Next::Opp)
}

/// [`eval_decision`] with the candidates valued in the frame `next`.
pub fn eval_decision_next(bot: &impl Bot, decision: Decision, eval_opps: &[State], next: Next) -> Option<Mark> {
    match decision {
        Decision::Forced(m) => m,
        Decision::Choices(cands) => {
            let states: Vec<State> = cands.iter().map(|(_, s)| *s).collect();
            let values = bot.evaluate_batch_next(&states, eval_opps, next);
            cands[argmax(&values)].0
        }
    }
}

/// Core decision logic shared by passive_phase1 and active_phase2.
///
/// Given candidate marks and a baseline state (skip for passive, skip-or-strike
/// for phase2), applies meta-rules and returns either a forced move or a
/// filtered+pruned candidate list for value-based selection.
///
/// The safe-lock force is gated by `force_lock`. With `force_lock = false`
/// the lock becomes an ordinary candidate (rule-free pipeline used by
/// search-value distillation and [`NoLock`](super::NoLock) bots); `true`
/// reproduces production.
pub(crate) fn mark_choices_with(
    state: &State,
    marks: &[Mark],
    baseline: State,
    opp_best: isize,
    force_lock: bool,
) -> Decision {
    if marks.is_empty() {
        return Decision::Forced(None);
    }

    if force_lock {
        if let Some(m) = find_safe_lock(state, marks) {
            return Decision::Forced(Some(m));
        }
    }
    // TODO: smart strike

    let mark_states: Vec<State> = marks
        .iter()
        .map(|&m| {
            let mut s = *state;
            s.apply_mark(m);
            s
        })
        .collect();

    // Force highest-scoring winning game-end (mark or baseline).
    // Marks first — a winning lock beats a winning strike (avoids -5 penalty).
    if let Some((mark, _)) = mark_states
        .iter()
        .enumerate()
        .map(|(i, &s)| (Some(marks[i]), s))
        .chain(std::iter::once((None, baseline)))
        .filter(|(_, post)| post.would_end_game() && post.count_points() > opp_best)
        .max_by_key(|(_, post)| post.count_points())
    {
        return Decision::Forced(mark);
    }

    // Build candidates: marks + baseline. Filter out losing game-ends.
    // This handles "must mark" automatically: if baseline is a losing strike,
    // it gets filtered and only marks survive.
    let mut cands: Vec<(Option<Mark>, State)> = mark_states
        .iter()
        .enumerate()
        .map(|(i, &s)| (Some(marks[i]), s))
        .chain(std::iter::once((None, baseline)))
        .filter(|(_, s)| !(s.would_end_game() && s.count_points() < opp_best))
        .collect();

    if cands.is_empty() {
        return Decision::Forced(None);
    }
    if cands.len() == 1 {
        return Decision::Forced(cands[0].0);
    }

    prune_dominated(&mut cands, |(_, s)| s);

    if cands.is_empty() {
        return Decision::Forced(None);
    }
    if cands.len() == 1 {
        return Decision::Forced(cands[0].0);
    }

    Decision::Choices(cands)
}

// ---------------------------------------------------------------------------

/// The deciding bot's own passive decision: with
/// [`Bot::passive_lookahead`], each candidate is valued at the end of the
/// active player's turn ([`passive_lookahead_values`]); otherwise the plain
/// [`passive_phase1_impl`]. Predicted opponents never look ahead: they go
/// through the plain pipeline.
pub(crate) fn passive_decide(
    bot: &impl Bot,
    state: &State,
    opp_states: &[State],
    dice: [u8; 6],
    active_player: usize,
) -> Option<Mark> {
    match passive_phase1_choices_with(state, opp_states, dice, bot.force_lock()) {
        Decision::Choices(cands) if bot.passive_lookahead() => {
            let marks: Vec<Option<Mark>> = cands.iter().map(|(m, _)| *m).collect();
            let values = passive_lookahead_values(bot, state, opp_states, dice, active_player, &marks);
            marks[argmax(&values)]
        }
        d => eval_decision_next(bot, d, opp_states, Next::Me),
    }
}

/// Each passive candidate `marks[i]` valued at the end of the active
/// player's turn, completed by [`passive_turn_ends`]: a finished game by its
/// result ([`Bot::terminal_value`]), otherwise by the net in the frame of
/// whoever moves next.
pub fn passive_lookahead_values(
    bot: &impl Bot,
    state: &State,
    opp_states: &[State],
    dice: [u8; 6],
    active_player: usize,
    marks: &[Option<Mark>],
) -> Vec<f32> {
    let ends = passive_turn_ends(bot, state, opp_states, dice, active_player, marks);
    let n = opp_states.len() + 1;
    let next = if (active_player + 2) % n == 0 {
        Next::Me
    } else {
        Next::Opp
    };
    let groups: Vec<(&[State], &[State], Next)> = ends
        .iter()
        .filter(|(_, ended)| !ended)
        .map(|(all, _)| (&all[..1], &all[1..], next))
        .collect();
    let mut leaves = bot.evaluate_batch_multi_next(&groups).into_iter().map(|v| v[0]);
    ends.iter()
        .map(|(all, ended)| {
            if *ended {
                bot.terminal_value(all)
            } else {
                leaves.next().unwrap()
            }
        })
        .collect()
}

/// The rest of the turn after each passive candidate `marks[i]`, in the
/// engine's order (`Game::play`). Every other player's phase-1 mark is
/// decided once, from the boards before phase 1, since phase 1 is
/// simultaneous and none of them sees ours. It is applied with ours and
/// locks propagate. Then, unless the game is over, the active player's
/// phase 2 is decided against the boards it now sees, and locks propagate
/// again. Returns (`[us, opps...]`, game ended) per candidate.
/// `active_player` indexes `opp_states`, as in `Strategy::passive_phase1`.
/// Every prediction is the bot's plain decision (no lookahead).
pub fn passive_turn_ends(
    bot: &impl Bot,
    state: &State,
    opp_states: &[State],
    dice: [u8; 6],
    active_player: usize,
    marks: &[Option<Mark>],
) -> Vec<(Vec<State>, bool)> {
    passive_turn_predictions(bot, state, opp_states, dice, active_player, marks)
        .into_iter()
        .map(|(all, ended, _)| (all, ended))
        .collect()
}

/// The active player's turn as [`passive_turn_ends`] predicts it: its
/// white-sum mark, then its colour mark or a strike (neither when the game
/// ended after phase 1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PredictedTurn {
    pub white: Option<Mark>,
    pub colour: Option<Mark>,
    pub strike: bool,
}

/// [`passive_turn_ends`] with the active player's predicted turn per
/// candidate.
pub fn passive_turn_predictions(
    bot: &impl Bot,
    state: &State,
    opp_states: &[State],
    dice: [u8; 6],
    active_player: usize,
    marks: &[Option<Mark>],
) -> Vec<(Vec<State>, bool, PredictedTurn)> {
    use super::sim::SimGame;
    let n = opp_states.len() + 1;
    let active = active_player + 1;
    let pre: Vec<State> = std::iter::once(*state).chain(opp_states.iter().copied()).collect();
    let phase1: Vec<Option<Mark>> = (1..n)
        .map(|j| {
            let view = SimGame::opp_view(&pre, j);
            if j == active {
                active_phase1_impl(bot, &pre[j], &view, dice)
            } else {
                passive_phase1_impl(bot, &pre[j], &view, dice)
            }
        })
        .collect();
    let active_marked = phase1[active - 1].is_some();
    marks
        .iter()
        .map(|&mark| {
            let mut all = pre.clone();
            for (j, m) in std::iter::once(mark).chain(phase1.iter().copied()).enumerate() {
                if let Some(m) = m {
                    all[j].apply_mark(m);
                }
            }
            SimGame::propagate_locks(&mut all);
            let mut turn = PredictedTurn {
                white: phase1[active - 1],
                colour: None,
                strike: false,
            };
            if SimGame::game_over(&all) {
                return (all, true, turn);
            }
            let view = SimGame::opp_view(&all, active);
            let d = active_phase2_choices_with(&all[active], &view, dice, active_marked, bot.force_lock());
            turn.colour = eval_decision(bot, d, &view);
            match turn.colour {
                Some(m) => all[active].apply_mark(m),
                None if !active_marked => {
                    all[active].apply_strike();
                    turn.strike = true;
                }
                None => {}
            }
            SimGame::propagate_locks(&mut all);
            let ended = SimGame::game_over(&all);
            (all, ended, turn)
        })
        .collect()
}

pub fn passive_phase1_choices(state: &State, opp_states: &[State], dice: [u8; 6]) -> Decision {
    passive_phase1_choices_with(state, opp_states, dice, true)
}

pub fn passive_phase1_choices_with(state: &State, opp_states: &[State], dice: [u8; 6], force_lock: bool) -> Decision {
    let white_sum = dice[0] + dice[1];
    let marks = state.generate_white_moves(white_sum);
    let opp_best = opp_best_phase1_score(opp_states, white_sum);
    mark_choices_with(state, &marks, *state, opp_best, force_lock)
}

/// The passive decision; in 2-player the passive player moves next.
pub fn passive_phase1_impl(bot: &impl Bot, state: &State, opp_states: &[State], dice: [u8; 6]) -> Option<Mark> {
    eval_decision_next(
        bot,
        passive_phase1_choices_with(state, opp_states, dice, bot.force_lock()),
        opp_states,
        Next::Me,
    )
}

pub fn active_phase2_choices(state: &State, opp_states: &[State], dice: [u8; 6], has_marked: bool) -> Decision {
    active_phase2_choices_with(state, opp_states, dice, has_marked, true)
}

pub fn active_phase2_choices_with(
    state: &State,
    opp_states: &[State],
    dice: [u8; 6],
    has_marked: bool,
    force_lock: bool,
) -> Decision {
    let opp_best = opp_states.iter().map(|s| s.count_points()).max().unwrap_or(0);
    let marks = state.generate_color_moves(dice);
    let baseline = if has_marked {
        *state
    } else {
        let mut s = *state;
        s.apply_strike();
        s
    };
    mark_choices_with(state, &marks, baseline, opp_best, force_lock)
}

pub(crate) fn active_phase2_impl(
    bot: &impl Bot,
    state: &State,
    opp_states: &[State],
    dice: [u8; 6],
    has_marked: bool,
) -> Option<Mark> {
    eval_decision(
        bot,
        active_phase2_choices_with(state, opp_states, dice, has_marked, bot.force_lock()),
        opp_states,
    )
}

// ---------------------------------------------------------------------------

/// Simulate each opponent's likely phase1 decision by running the full
/// passive_phase1 pipeline (safe lock, endgame rules, evaluate) from
/// their perspective.
fn simulate_opp_phase1(bot: &impl Bot, state: &State, opp_states: &[State], dice: [u8; 6]) -> Vec<State> {
    opp_states
        .iter()
        .enumerate()
        .map(|(i, opp)| {
            let their_opps: Vec<State> = std::iter::once(*state)
                .chain(opp_states.iter().enumerate().filter(|(j, _)| *j != i).map(|(_, s)| *s))
                .collect();
            match passive_phase1_impl(bot, opp, &their_opps, dice) {
                Some(m) => {
                    let mut s = *opp;
                    s.apply_mark(m);
                    s
                }
                None => *opp,
            }
        })
        .collect()
}

/// Pure-logic phase-1 plan pipeline. `comparison_opps` supplies `opp_best`
/// for the winning/losing endgame filters — predicted post-phase1 states in
/// the full pipeline, current states in the simulator's lite mode. Returned
/// `Choices` carry (phase1 mark, plan end-state); the phase-2 part of each
/// plan is internal (the chooser only commits phase 1).
///
/// The safe-lock force is gated by `force_lock` (the deciding bot's
/// [`Bot::force_lock`]): `false` keeps the lock an ordinary candidate.
///
/// The game ends after phase 1 when our board and `comparison_opps`
/// together have two closed rows (or four strikes): then there is no colour
/// phase, and no strike for marking nothing (`Game::play`).
pub(crate) fn phase1_plan_choices_with(
    state: &State,
    comparison_opps: &[State],
    dice: [u8; 6],
    force_lock: bool,
) -> Decision {
    let white_sum = dice[0] + dice[1];
    let opp_best = comparison_opps.iter().map(|s| s.count_points()).max().unwrap_or(0);
    let ends = |post: &State| ends_with(post, comparison_opps);

    let white_marks = state.generate_white_moves(white_sum);

    let mut plans: Vec<(Option<Mark>, Option<Mark>, State)> = Vec::new();

    // No white mark: the colour phase, or nothing if the opponents' white
    // marks already end the game.
    let no_white = if ends(state) {
        *state
    } else {
        let mut s = *state;
        s.apply_strike();
        s
    };
    plans.push((None, None, no_white));
    if !ends(state) {
        // Color-only singles
        for &cm in &state.generate_color_moves(dice) {
            let mut s = *state;
            s.apply_mark(cm);
            plans.push((None, Some(cm), s));
        }
    }
    // White-only singles
    for &wm in &white_marks {
        let mut s = *state;
        s.apply_mark(wm);
        plans.push((Some(wm), None, s));
    }
    // Doubles, unless the white mark ends the game
    for &wm in &white_marks {
        let mut post_white = *state;
        post_white.apply_mark(wm);
        if ends(&post_white) {
            continue;
        }
        for &cm in &post_white.generate_color_moves(dice) {
            let mut s = post_white;
            s.apply_mark(cm);
            plans.push((Some(wm), Some(cm), s));
        }
    }

    if plans.is_empty() {
        return Decision::Forced(None);
    }

    // Force best winning game-end
    let winning = plans
        .iter()
        .enumerate()
        .filter(|(_, (_, _, post))| ends(post) && post.count_points() > opp_best)
        .max_by_key(|(_, (_, _, post))| post.count_points());
    if let Some((i, _)) = winning {
        return Decision::Forced(plans[i].0);
    }

    // Remove losing game-ends
    plans.retain(|(_, _, post)| !(ends(post) && post.count_points() < opp_best));
    if plans.is_empty() {
        plans.push((None, None, no_white));
    }

    // Force safe lock
    if force_lock {
        for (phase1, _, _) in &plans {
            if let Some(m) = phase1 {
                if state.would_lock_row(*m) && {
                    let mut s = *state;
                    s.apply_mark(*m);
                    !ends(&s)
                } {
                    return Decision::Forced(Some(*m));
                }
            }
        }
    }

    prune_dominated(&mut plans, |(_, _, s)| s);

    Decision::Choices(plans.into_iter().map(|(p1, _, s)| (p1, s)).collect())
}

/// Whether the game is over with our board `post` and `opps`, locks shared.
fn ends_with(post: &State, opps: &[State]) -> bool {
    let mut all: Vec<State> = std::iter::once(*post).chain(opps.iter().copied()).collect();
    super::sim::SimGame::propagate_locks(&mut all);
    super::sim::SimGame::game_over(&all)
}

/// Full phase-1 pipeline: simulate opponents' phase-1 responses, then run the
/// plan pipeline against them. Returns the decision plus the simulated
/// post-phase1 opponent states (also the evaluation context for Choices).
pub fn active_phase1_choices(
    bot: &impl Bot,
    state: &State,
    opp_states: &[State],
    dice: [u8; 6],
) -> (Decision, Vec<State>) {
    // Simulate opponents' likely phase1 marks to get predicted post-phase1
    // opponent states. This replaces the RISKY filter: instead of conservatively
    // removing plans that might be invalidated, we predict what opponents will
    // do and plan around it.
    let sim_opp = simulate_opp_phase1(bot, state, opp_states, dice);
    let decision = phase1_plan_choices_with(state, &sim_opp, dice, bot.force_lock());
    (decision, sim_opp)
}

pub(crate) fn active_phase1_impl(bot: &impl Bot, state: &State, opp_states: &[State], dice: [u8; 6]) -> Option<Mark> {
    let (decision, sim_opp) = active_phase1_choices(bot, state, opp_states, dice);
    eval_decision(bot, decision, &sim_opp)
}

// ---------------------------------------------------------------------------

impl<T: Bot + std::fmt::Debug> super::Strategy for T {
    fn active_phase1(&mut self, state: &State, opp_states: &[State], dice: [u8; 6]) -> Option<Mark> {
        active_phase1_impl(self, state, opp_states, dice)
    }

    fn active_phase2(&mut self, state: &State, opp_states: &[State], dice: [u8; 6], has_marked: bool) -> Option<Mark> {
        active_phase2_impl(self, state, opp_states, dice, has_marked)
    }

    fn passive_phase1(
        &mut self,
        state: &State,
        opp_states: &[State],
        dice: [u8; 6],
        active_player: usize,
    ) -> Option<Mark> {
        passive_decide(self, state, opp_states, dice, active_player)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::Mark;

    /// A state with a safe lock available: red 2..6 marked, white sum 12
    /// completes it (first lock, does not end the game).
    fn lockable_state() -> State {
        let mut s = State::default();
        for n in 2..=6 {
            s.apply_mark(Mark { row: 0, number: n });
        }
        s
    }

    #[test]
    fn mark_choices_with_force_matches_production_and_without_skips_lock() {
        let s = lockable_state();
        let marks = vec![Mark { row: 0, number: 12 }, Mark { row: 1, number: 5 }];
        // Forced path: identical to mark_choices.
        match mark_choices_with(&s, &marks, s, 0, true) {
            Decision::Forced(Some(m)) => assert_eq!(m, Mark { row: 0, number: 12 }),
            d => panic!("expected forced lock, got {:?}", matches!(d, Decision::Choices(_))),
        }
        // Rule-free path: the lock is a candidate, not forced.
        match mark_choices_with(&s, &marks, s, 0, false) {
            Decision::Choices(c) => {
                assert!(c.iter().any(|(m, _)| *m == Some(Mark { row: 0, number: 12 })));
                assert!(c.len() >= 2);
            }
            Decision::Forced(_) => panic!("rule-free pipeline must not force here"),
        }
        // find_safe_lock is exposed and agrees.
        assert_eq!(find_safe_lock(&s, &marks), Some(Mark { row: 0, number: 12 }));
    }

    #[test]
    fn phase1_plan_choices_with_force_matches_production() {
        let s = lockable_state();
        let opps = [State::default()];
        let dice = [6, 6, 1, 1, 1, 1]; // white sum 12 completes the red lock
        match phase1_plan_choices_with(&s, &opps, dice, true) {
            Decision::Forced(Some(m)) => assert_eq!(m, Mark { row: 0, number: 12 }),
            _ => panic!("expected forced phase-1 lock"),
        }
        match phase1_plan_choices_with(&s, &opps, dice, false) {
            Decision::Choices(c) => {
                assert!(c.iter().any(|(m, _)| *m == Some(Mark { row: 0, number: 12 })))
            }
            Decision::Forced(_) => panic!("rule-free phase-1 pipeline must not force here"),
        }
    }

    fn marked(marks: &[(usize, u8)]) -> State {
        let mut s = State::default();
        for &(row, number) in marks {
            s.apply_mark(Mark { row, number });
        }
        s
    }

    /// The opponent has closed blue (28 points). I can close red with a
    /// white 12, the second closed row, so the game ends after phase 1
    /// with my red at 28: 29 with my yellow 2. A colour mark after it
    /// (yellow 9: +2) never happens.
    fn second_lock(opp_extra: &[(usize, u8)]) -> (State, State, [u8; 6]) {
        let blue = [(3, 12), (3, 11), (3, 10), (3, 9), (3, 8), (3, 2)];
        let opp = marked(&[&blue[..], opp_extra].concat());
        let mut me = marked(&[(0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (1, 2)]);
        me.lock(opp.locked());
        (me, opp, [6, 6, 1, 3, 1, 1]) // white 12; yellow 6 + 3 = 9
    }

    const RED12: Mark = Mark { row: 0, number: 12 };

    #[test]
    fn a_white_mark_that_ends_the_game_is_never_followed_by_a_colour_mark() {
        // Opponent 29: red 12 alone ties, so it stays a candidate, as is.
        let (me, opp, dice) = second_lock(&[(1, 2)]);
        let mut alone = me;
        alone.apply_mark(RED12);
        assert_eq!((alone.count_points(), opp.count_points()), (29, 29));
        let Decision::Choices(c) = phase1_plan_choices_with(&me, &[opp], dice, false) else {
            panic!("a tie is no forced win")
        };
        let posts: Vec<State> = c.iter().filter(|(m, _)| *m == Some(RED12)).map(|(_, s)| *s).collect();
        assert_eq!(posts, [alone]);
    }

    #[test]
    fn a_white_mark_that_ends_the_game_behind_is_not_a_forced_win() {
        // Opponent 30: red 12 alone loses 29–30; with yellow 9 it would
        // have won 31–30.
        let (me, opp, dice) = second_lock(&[(1, 2), (2, 12)]);
        assert_eq!(opp.count_points(), 30);
        match phase1_plan_choices_with(&me, &[opp], dice, false) {
            Decision::Forced(m) => assert_ne!(m, Some(RED12)),
            Decision::Choices(c) => assert!(c.iter().all(|(m, _)| *m != Some(RED12)), "a losing end"),
        }
        use crate::strategy::recorder::Points;
        assert_ne!(active_phase1_impl(&Points, &me, &[opp], dice), Some(RED12));
    }

    #[test]
    fn when_the_opponents_white_mark_ends_the_game_there_is_no_colour_phase() {
        // The opponent's predicted white mark closes red as well as blue:
        // my phase 1 is my whole turn, and passing costs no strike.
        let blue = [(3, 12), (3, 11), (3, 10), (3, 9), (3, 8), (3, 2)];
        let red = [(0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 12)];
        let opp = marked(&[&blue[..], &red[..]].concat());
        let mut me = marked(&[(1, 2)]);
        me.lock(marked(&blue).locked());
        let dice = [3, 4, 1, 3, 1, 1]; // white 7; colour sums 4..10
        let whites = me.generate_white_moves(7);
        assert!(!whites.is_empty());
        let posts: Vec<State> = match phase1_plan_choices_with(&me, &[opp], dice, false) {
            Decision::Choices(c) => c.into_iter().map(|(_, s)| s).collect(),
            Decision::Forced(m) => vec![m.map_or(me, |m| {
                let mut s = me;
                s.apply_mark(m);
                s
            })],
        };
        for post in posts {
            let ok = post == me
                || whites.iter().any(|&w| {
                    let mut s = me;
                    s.apply_mark(w);
                    s == post
                });
            assert!(ok, "only my white mark or nothing: {post:?}");
        }
    }

    use crate::strategy::recorder::ShunsPoints;

    #[test]
    fn a_bot_without_the_lock_rule_may_decline_a_safe_lock() {
        use crate::strategy::{NoLock, Strategy};
        let (s, opps) = (lockable_state(), [State::default()]);
        let lock = Some(Mark { row: 0, number: 12 });
        let white = [6, 6, 1, 1, 1, 1]; // white sum 12
        let colour = [6, 1, 6, 1, 1, 1]; // white 6 + red 6
        let (mut on, mut off) = (ShunsPoints, NoLock(ShunsPoints));
        assert_eq!(on.passive_phase1(&s, &opps, white, 1), lock);
        assert_eq!(on.active_phase1(&s, &opps, white), lock);
        assert_eq!(on.active_phase2(&s, &opps, colour, true), lock);
        assert_eq!(off.passive_phase1(&s, &opps, white, 1), None);
        assert_ne!(off.active_phase1(&s, &opps, white), lock);
        assert_eq!(off.active_phase2(&s, &opps, colour, true), None);
    }

    #[test]
    fn the_opponents_predicted_reply_follows_the_bots_rule() {
        use crate::strategy::NoLock;
        // The opponent can lock red with the white 12; the bot shuns points,
        // so only the rule makes it lock.
        let (me, opp) = (State::default(), lockable_state());
        let dice = [6, 6, 1, 1, 1, 1];
        let red = |s: Vec<State>| s[0].locked()[0];
        assert!(red(simulate_opp_phase1(&ShunsPoints, &me, &[opp], dice)));
        assert!(!red(simulate_opp_phase1(&NoLock(ShunsPoints), &me, &[opp], dice)));
    }

    use crate::strategy::recorder::Recorder;
    use crate::strategy::Next;

    fn nexts(bot: &Recorder) -> Vec<Next> {
        bot.take().into_iter().map(|c| c.next).collect()
    }

    #[test]
    fn passive_decisions_say_i_move_next_and_active_ones_the_opponent() {
        let bot = Recorder::default();
        let (state, opps) = (State::default(), [State::default()]);
        let dice = [3, 4, 2, 5, 6, 2];
        passive_phase1_impl(&bot, &state, &opps, dice);
        assert_eq!(nexts(&bot), [Next::Me]);
        active_phase2_impl(&bot, &state, &opps, dice, false);
        assert_eq!(nexts(&bot), [Next::Opp]);
        // The opponent's simulated passive response, then my plans.
        active_phase1_impl(&bot, &state, &opps, dice);
        assert_eq!(nexts(&bot), [Next::Me, Next::Opp]);
        assert!(eval_decision(&bot, passive_phase1_choices(&state, &opps, dice), &opps).is_some());
        assert_eq!(nexts(&bot), [Next::Opp], "eval_decision is the opponent-next case");
    }

    /// Me: red can lock with a white 12. The active opponent: yellow can lock
    /// with it too, and green and blue have five marks each, so it leads.
    fn race() -> (State, [State; 1], [u8; 6]) {
        use crate::strategy::recorder::{lockable, lockable_row};
        let mut active = lockable_row(1);
        for row in [2, 3] {
            for n in (8..=12).rev() {
                active.apply_mark(Mark { row, number: n });
            }
        }
        (lockable(), [active], [6, 6, 1, 1, 1, 1])
    }

    #[test]
    fn lookahead_declines_a_lock_that_ends_the_game_in_a_loss() {
        use crate::strategy::recorder::Points;
        use crate::strategy::{Lookahead, NoLock};
        let (me, opps, dice) = race();
        let red12 = Some(Mark { row: 0, number: 12 });
        // Against the board before its turn, my lock is the most points...
        assert_eq!(passive_decide(&NoLock(Points), &me, &opps, dice, 0), red12);
        // ...but the opponent locks yellow with the same white 12: two locks
        // end the game before its colour phase, and I am behind.
        let pick = passive_decide(&NoLock(Lookahead(Points)), &me, &opps, dice, 0);
        assert_ne!(pick, red12);
        assert!(pick.is_some(), "a mark still beats skipping");
    }

    #[test]
    fn a_turn_that_ends_the_game_is_valued_by_its_result() {
        use crate::strategy::recorder::Points;
        use crate::strategy::{Lookahead, NoLock};
        let (me, opps, dice) = race();
        let bot = NoLock(Lookahead(Points));
        let red12 = Some(Mark { row: 0, number: 12 });
        let green12 = Some(Mark { row: 2, number: 12 });
        let values = passive_lookahead_values(&bot, &me, &opps, dice, 0, &[red12, green12]);
        assert_eq!(values[0], -50.0, "a loss");
        assert!(values[1] > -50.0 && values[1] < 0.0, "{values:?}");
    }

    #[test]
    fn a_turn_that_goes_on_is_valued_in_the_next_movers_frame() {
        use crate::strategy::recorder::Framed;
        use crate::strategy::Lookahead;
        let dice = [3, 4, 2, 5, 6, 2];
        let me = State::default();
        let skip = [None];
        // Two players: I move after the active player.
        let two = passive_lookahead_values(&Lookahead(Framed), &me, &[State::default()], dice, 0, &skip);
        assert_eq!(two, [1.0]);
        // Three players, the first opponent active: the second one is next.
        let three = [State::default(); 2];
        assert_eq!(
            passive_lookahead_values(&Lookahead(Framed), &me, &three, dice, 0, &skip),
            [-1.0]
        );
        // The second opponent active: I am next.
        assert_eq!(
            passive_lookahead_values(&Lookahead(Framed), &me, &three, dice, 1, &skip),
            [1.0]
        );
    }

    #[test]
    fn a_predicted_opponent_reply_never_looks_ahead() {
        use crate::strategy::Lookahead;
        let (state, opps) = (State::default(), [State::default()]);
        let dice = [3, 4, 2, 5, 6, 2];
        let plain = Recorder::default();
        active_phase1_impl(&plain, &state, &opps, dice);
        let looking = Lookahead(Recorder::default());
        active_phase1_impl(&looking, &state, &opps, dice);
        assert_eq!(looking.0.take(), plain.take());
    }

    #[test]
    fn the_predicted_turn_rebuilds_the_active_board() {
        use crate::strategy::recorder::Points;
        use crate::strategy::NoLock;
        let bot = NoLock(Points);
        let (me, opps, dice) = race();
        let red12 = Some(Mark { row: 0, number: 12 });
        let green12 = Some(Mark { row: 2, number: 12 });
        let marks = [None, red12, green12];
        let turns = passive_turn_predictions(&bot, &me, &opps, dice, 0, &marks);
        let ends = passive_turn_ends(&bot, &me, &opps, dice, 0, &marks);
        for ((all, ended, turn), end) in turns.iter().zip(&ends) {
            assert_eq!((all, ended), (&end.0, &end.1));
            let mut active = opps[0];
            for m in [turn.white, turn.colour].into_iter().flatten() {
                active.apply_mark(m);
            }
            if turn.strike {
                active.apply_strike();
            }
            active.lock(all[1].locked());
            assert_eq!(active, all[1], "{turn:?}");
        }
        // The opponent's white 12 locks yellow; with my red lock the game
        // ends before its colour phase.
        let yellow12 = Some(Mark { row: 1, number: 12 });
        assert_eq!(
            turns[1].2,
            PredictedTurn {
                white: yellow12,
                colour: None,
                strike: false
            }
        );
        assert!(turns[1].1);
        assert!(turns.iter().all(|t| t.2.white == yellow12), "phase 1 ignores my mark");
    }

    #[test]
    fn without_lookahead_a_passive_decision_is_the_plain_one() {
        let bot = Recorder::default();
        let (state, opps) = (State::default(), [State::default()]);
        let dice = [3, 4, 2, 5, 6, 2];
        assert_eq!(
            passive_decide(&bot, &state, &opps, dice, 0),
            passive_phase1_impl(&bot, &state, &opps, dice)
        );
        assert_eq!(nexts(&bot), [Next::Me, Next::Me]);
    }
}
