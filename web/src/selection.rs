//! The human's legal selections (spec "Which squares are legal").

use qwixxer::state::{Mark, State};

/// What the human may confirm on their own turn, given the bot's revealed
/// white-sum mark.
#[derive(Debug, Clone, PartialEq)]
pub struct ActiveOptions {
    /// Legal selections in application order: `[white]`, `[colour]` or
    /// `[white, colour]`. Never empty; each set appears once.
    pub selections: Vec<Vec<Mark>>,
    /// The strike box may be confirmed (alone).
    pub strike: bool,
    /// Confirming nothing is legal ("Pass"): the game ends after phase 1.
    pub pass: bool,
}

pub fn union(a: [bool; 4], b: [bool; 4]) -> [bool; 4] {
    [a[0] | b[0], a[1] | b[1], a[2] | b[2], a[3] | b[3]]
}

pub fn n_locked(l: [bool; 4]) -> usize {
    l.iter().filter(|&&x| x).count()
}

fn set_key(sel: &[Mark]) -> Vec<(usize, u8)> {
    let mut k: Vec<(usize, u8)> = sel.iter().map(|m| (m.row, m.number)).collect();
    k.sort_unstable();
    k
}

/// Spec rules 1–3: phase-1 marks are judged on the pre-roll boards (the
/// bot's simultaneous lock never blocks your white sum); phase 2 is judged
/// after both phase-1 marks and lock propagation; if two rows are locked
/// after phase 1 the turn ends there (no colour mark, no strike).
pub fn active_options(you: &State, bot: &State, bot_white: Option<Mark>, dice: [u8; 6]) -> ActiveOptions {
    let (turns, ends_after_bot) = attributed_turns(you, bot, bot_white, dice);
    let mut seen: Vec<Vec<(usize, u8)>> = Vec::new();
    let mut selections = Vec::new();
    for (w, c) in turns {
        let sel: Vec<Mark> = w.into_iter().chain(c).collect();
        let k = set_key(&sel);
        if !seen.contains(&k) {
            seen.push(k);
            selections.push(sel);
        }
    }
    ActiveOptions {
        selections,
        strike: !ends_after_bot,
        pass: ends_after_bot,
    }
}

/// A marking turn as (white-sum mark, colour mark).
pub type Attributed = (Option<Mark>, Option<Mark>);

/// Every legal marking turn as (white-sum mark, colour mark), by the rules of
/// [`active_options`]: one square can appear both as a white-sum mark and as
/// a colour mark. Also whether the other's white-sum mark ends the game
/// after phase 1.
pub fn attributed_turns(
    you: &State,
    bot: &State,
    bot_white: Option<Mark>,
    dice: [u8; 6],
) -> (Vec<Attributed>, bool) {
    let white_sum = dice[0] + dice[1];
    let mut bot_after = *bot;
    if let Some(w) = bot_white {
        bot_after.apply_mark(w);
    }
    let bot_locks = bot_after.locked();
    let base_locks = union(you.locked(), bot_locks);
    let ends_after_bot = n_locked(base_locks) >= 2;

    let mut turns = Vec::new();
    for w in you.generate_white_moves(white_sum) {
        turns.push((Some(w), None));
        let mut after_white = *you;
        after_white.apply_mark(w);
        let locks = union(after_white.locked(), bot_locks);
        if n_locked(locks) >= 2 {
            continue;
        }
        after_white.lock(locks);
        for c in after_white.generate_color_moves(dice) {
            turns.push((Some(w), Some(c)));
        }
    }
    if !ends_after_bot {
        let mut no_white = *you;
        no_white.lock(base_locks);
        for c in no_white.generate_color_moves(dice) {
            turns.push((None, Some(c)));
        }
    }
    (turns, ends_after_bot)
}

/// On the bot's turn the human may mark only the white sum, or pass.
pub fn passive_options(you: &State, dice: [u8; 6]) -> Vec<Mark> {
    you.generate_white_moves(dice[0] + dice[1])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(row: usize, number: u8) -> Mark {
        Mark { row, number }
    }

    /// A state with `nums` marked in `row`, in order.
    fn with(mut s: State, row: usize, nums: &[u8]) -> State {
        for &n in nums {
            s.apply_mark(m(row, n));
        }
        s
    }

    fn has(opts: &ActiveOptions, sel: &[Mark]) -> bool {
        opts.selections
            .iter()
            .any(|x| x.len() == sel.len() && sel.iter().all(|s| x.contains(s)))
    }

    #[test]
    fn white_twelve_locking_second_row_ends_after_phase_one() {
        // Yellow is already locked; your white 12 closes red: no colour mark may follow.
        let mut you = with(State::default(), 0, &[2, 3, 4, 5, 6]);
        you.lock([false, true, false, false]);
        let mut bot = State::default();
        bot.lock([false, true, false, false]);
        let opts = active_options(&you, &bot, None, [6, 6, 1, 1, 1, 1]);
        assert!(has(&opts, &[m(0, 12)]));
        assert!(!opts.selections.iter().any(|x| x.len() == 2 && x.contains(&m(0, 12))));
        assert!(opts.strike);
        assert!(!opts.pass);
    }

    #[test]
    fn bot_reveal_locking_second_row_leaves_only_white_or_pass() {
        let mut you = with(State::default(), 0, &[2, 3]);
        you.lock([false, true, false, false]);
        let mut bot = with(State::default(), 0, &[2, 3, 4, 5, 6]);
        bot.lock([false, true, false, false]);
        let opts = active_options(&you, &bot, Some(m(0, 12)), [6, 6, 1, 1, 1, 1]);
        assert!(opts.pass);
        assert!(!opts.strike);
        assert!(!opts.selections.is_empty());
        assert!(opts.selections.iter().all(|x| x.len() == 1 && x[0].number == 12));
    }

    #[test]
    fn bot_locking_a_row_keeps_your_white_sum_but_not_colour_there() {
        let you = with(State::default(), 0, &[2, 3, 4, 5, 6]);
        let bot = with(State::default(), 0, &[2, 3, 4, 5, 6]);
        // White 6+6 = 12; red die 2 → red 8 (or 8 again).
        let opts = active_options(&you, &bot, Some(m(0, 12)), [6, 6, 2, 1, 1, 1]);
        assert!(has(&opts, &[m(0, 12)]));
        assert!(!opts.selections.iter().any(|x| x.contains(&m(0, 8))));
        assert!(opts.strike);
    }

    #[test]
    fn same_row_pair_needs_white_on_the_left() {
        // White 3+5 = 8; red colour sums 3+1 = 4 and 5+1 = 6.
        let opts = active_options(&State::default(), &State::default(), None, [3, 5, 1, 1, 1, 1]);
        assert!(has(&opts, &[m(0, 8)]));
        assert!(has(&opts, &[m(0, 4)]));
        // Red 4 + red 8 would need the colour mark (4) before the white sum (8).
        assert!(!has(&opts, &[m(0, 4), m(0, 8)]));
        // White red 8 with a colour mark in another row is fine.
        assert!(has(&opts, &[m(0, 8), m(1, 4)]));
    }

    #[test]
    fn white_mark_can_be_the_fifth_that_allows_a_colour_lock() {
        // Red has 4 marks; white 1+6 = 7 is the fifth, then colour 6+6 = 12 locks.
        let you = with(State::default(), 0, &[2, 3, 4, 5]);
        let opts = active_options(&you, &State::default(), None, [1, 6, 6, 1, 1, 1]);
        assert!(has(&opts, &[m(0, 7), m(0, 12)]));
        assert!(!has(&opts, &[m(0, 12)]));
    }

    #[test]
    fn square_that_is_both_white_and_colour_appears_once() {
        // White 3+4 = 7; red colour 4+3 = 7 too.
        let opts = active_options(&State::default(), &State::default(), None, [3, 4, 3, 1, 1, 1]);
        let n = opts
            .selections
            .iter()
            .filter(|x| x.len() == 1 && x[0] == m(0, 7))
            .count();
        assert_eq!(n, 1);
    }

    #[test]
    fn no_legal_marks_leaves_only_the_strike() {
        // Every row sits on its terminal number with 4 marks: nothing is markable.
        let you = State::from_parts(0, [(4, Some(12)), (4, Some(12)), (4, Some(2)), (4, Some(2))]);
        let opts = active_options(&you, &State::default(), None, [3, 4, 2, 2, 2, 2]);
        assert!(opts.selections.is_empty());
        assert!(opts.strike);
        assert!(!opts.pass);
    }

    #[test]
    fn passive_options_are_the_white_sum_marks() {
        let opts = passive_options(&State::default(), [3, 4, 1, 1, 1, 1]);
        assert_eq!(opts, vec![m(0, 7), m(1, 7), m(2, 7), m(3, 7)]);
    }
}
