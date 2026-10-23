//! One human-vs-bot game: the turn flow of the web UI spec over the engine.
//! Follows `src/game.rs`: both phase-1 marks are decided on the pre-roll
//! boards and applied together, locks propagate, the game may end, then the
//! active player's phase 2. The human is player 0, the bot player 1; the bot
//! starts. Display marks are the source of truth; `State` is derived.

use crate::bots::BotKind;
use crate::explorer::Turn;
use crate::gamecode::{self, Header};
use crate::selection::{active_options, n_locked, passive_options, ActiveOptions};
use qwixxer::state::{Mark, State};
use qwixxer::strategy::Strategy;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

/// Column of number `n` in row `row` (red/yellow 2→12, green/blue 12→2).
pub fn cell_index(row: usize, n: u8) -> usize {
    if row < 2 {
        (n - 2) as usize
    } else {
        (12 - n) as usize
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Sq {
    pub row: usize,
    pub n: u8,
}

impl From<Mark> for Sq {
    fn from(m: Mark) -> Self {
        Sq {
            row: m.row,
            n: m.number,
        }
    }
}

impl From<Sq> for Mark {
    fn from(s: Sq) -> Self {
        Mark {
            row: s.row,
            number: s.n,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    BotToRoll,
    Passive,
    YouToRoll,
    Active,
    Over,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Who {
    You,
    Bot,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OverReason {
    Locks,
    YouStrikes,
    BotStrikes,
}

/// One resolved roll, for the log and for staging the bot's marks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RollRecord {
    pub roller: Who,
    pub you: Vec<Sq>,
    pub you_strike: bool,
    pub bot_white: Option<Sq>,
    pub bot_colour: Option<Sq>,
    pub bot_strike: bool,
    /// Rows that became locked during this roll.
    pub locked: Vec<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SheetSnap {
    pub marks: [[bool; 11]; 4],
    pub strikes: u8,
}

impl SheetSnap {
    fn empty() -> Self {
        SheetSnap {
            marks: [[false; 11]; 4],
            strikes: 0,
        }
    }

    pub(crate) fn mark(&mut self, s: Sq) {
        self.marks[s.row][cell_index(s.row, s.n)] = true;
    }

    fn state(&self, locked: [bool; 4]) -> State {
        let mut st = State::from_marks(&self.marks, self.strikes);
        st.lock(locked);
        st
    }

    fn view(&self, locked: [bool; 4]) -> SheetView {
        let row_points: [isize; 4] = core::array::from_fn(|r| {
            // The lock symbol counts as an extra mark for whoever locked the row.
            let n = self.marks[r].iter().filter(|&&m| m).count() as isize + self.marks[r][10] as isize;
            n * (n + 1) / 2
        });
        SheetView {
            marks: self.marks,
            strikes: self.strikes,
            score: self.state(locked).count_points(),
            row_points,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Snapshot {
    pub you: SheetSnap,
    pub bot: SheetSnap,
    pub locked: [bool; 4],
    pub phase: Phase,
    pub dice: Option<[u8; 6]>,
    /// The bot's phase-1 mark for the current roll (hidden on its own turn).
    pub bot_white: Option<Sq>,
    pub over: Option<OverReason>,
    /// Each resolved roll as its game-code value (`gamecode::pack`), in
    /// order. `None` for games saved before rolls were recorded: they have
    /// no code.
    #[serde(default)]
    pub history: Option<Vec<u32>>,
}

impl Snapshot {
    /// Checks what the session indexes or unwraps: square rows and numbers,
    /// dice faces, strikes, and that dice are present exactly when a roll is
    /// being resolved.
    pub fn validate(&self) -> Result<(), String> {
        if let Some(s) = self.bot_white {
            if s.row >= 4 || !(2..=12).contains(&s.n) {
                return Err(format!("bad square {s:?}"));
            }
        }
        if let Some(d) = self.dice {
            if d.iter().any(|f| !(1..=6).contains(f)) {
                return Err(format!("bad dice {d:?}"));
            }
        }
        if self.you.strikes > 4 || self.bot.strikes > 4 {
            return Err("too many strikes".into());
        }
        let rolling = matches!(self.phase, Phase::Passive | Phase::Active);
        if rolling && self.dice.is_none() {
            return Err(format!("phase {:?} needs dice", self.phase));
        }
        if !rolling && self.bot_white.is_some() {
            return Err(format!("phase {:?} has a pending bot mark", self.phase));
        }
        if (self.phase == Phase::Over) != self.over.is_some() {
            return Err("phase and game-over reason disagree".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct SheetView {
    pub marks: [[bool; 11]; 4],
    pub strikes: u8,
    pub score: isize,
    pub row_points: [isize; 4],
}

#[derive(Debug, Clone, Serialize)]
pub struct OptionsView {
    pub selections: Vec<Vec<Sq>>,
    pub strike: bool,
    pub pass: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct View {
    pub phase: Phase,
    pub dice: Option<[u8; 6]>,
    pub you: SheetView,
    pub bot: SheetView,
    pub locked: [bool; 4],
    /// The bot's revealed white-sum mark (only during your turn).
    pub bot_white: Option<Sq>,
    /// Your legal selections (only during your turn).
    pub options: Option<OptionsView>,
    /// Your legal white-sum squares (only during the bot's turn).
    pub passive: Vec<Sq>,
    pub over: Option<OverReason>,
}

pub struct Session {
    g: Snapshot,
    strategy: Box<dyn Strategy>,
    rng: SmallRng,
}

impl Session {
    pub fn new(strategy: Box<dyn Strategy>, seed: u64) -> Self {
        let g = Snapshot {
            you: SheetSnap::empty(),
            bot: SheetSnap::empty(),
            locked: [false; 4],
            phase: Phase::BotToRoll,
            dice: None,
            bot_white: None,
            over: None,
            history: Some(vec![]),
        };
        Session {
            g,
            strategy,
            rng: SmallRng::seed_from_u64(seed),
        }
    }

    /// Resume a saved game. Snapshots come from browser storage, so a
    /// malformed one is rejected here rather than panicking later.
    pub fn restore(snap: Snapshot, strategy: Box<dyn Strategy>, seed: u64) -> Result<Self, String> {
        snap.validate()?;
        Ok(Session {
            g: snap,
            strategy,
            rng: SmallRng::seed_from_u64(seed),
        })
    }

    pub fn snapshot(&self) -> Snapshot {
        self.g.clone()
    }

    /// The game code (spec "The game code"); `Err` for a game saved before
    /// rolls were recorded. Every game starts with the bot's roll.
    pub fn code(&self, bot: BotKind) -> Result<String, String> {
        let h = self
            .g
            .history
            .as_ref()
            .ok_or("this game was saved before rolls were recorded")?;
        Ok(gamecode::encode(&Header { bot_first: true, bot }, h))
    }

    /// Appends a resolved roll to the history, from the sheets before it.
    fn record(&mut self, pre: (State, State), dice: [u8; 6], rec: &RollRecord) {
        let (you, bot) = pre;
        let (r, o, turn, other) = match rec.roller {
            Who::You => (
                you,
                bot,
                Turn {
                    marks: rec.you.iter().map(|&q| q.into()).collect(),
                    strike: rec.you_strike,
                },
                rec.bot_white.map(Mark::from),
            ),
            Who::Bot => (
                bot,
                you,
                Turn {
                    marks: rec
                        .bot_white
                        .iter()
                        .chain(rec.bot_colour.iter())
                        .map(|&q| q.into())
                        .collect(),
                    strike: rec.bot_strike,
                },
                rec.you.first().map(|&q| q.into()),
            ),
        };
        match gamecode::canonical_action(&r, &o, dice, &turn, other) {
            Some(a) => {
                if let Some(h) = self.g.history.as_mut() {
                    h.push(gamecode::pack(dice, a));
                }
            }
            // Can't happen for a legal roll; a code missing a roll would be
            // wrong, so the game gets none.
            None => self.g.history = None,
        }
    }

    fn states(&self) -> (State, State) {
        (self.g.you.state(self.g.locked), self.g.bot.state(self.g.locked))
    }

    fn expect(&self, phase: Phase) -> Result<(), String> {
        if self.g.phase == phase {
            Ok(())
        } else {
            Err(format!("expected phase {phase:?}, game is in {:?}", self.g.phase))
        }
    }

    fn roll(&mut self) -> [u8; 6] {
        core::array::from_fn(|_| self.rng.gen_range(1..=6))
    }

    /// Lock every row whose terminal number got marked; returns the new ones.
    fn settle_locks(&mut self) -> Vec<usize> {
        let mut newly = Vec::new();
        for r in 0..4 {
            if !self.g.locked[r] && (self.g.you.marks[r][10] || self.g.bot.marks[r][10]) {
                self.g.locked[r] = true;
                newly.push(r);
            }
        }
        newly
    }

    /// Ends the game if a player has 4 strikes or two rows are locked.
    fn check_over(&mut self) -> bool {
        let reason = if self.g.you.strikes >= 4 {
            Some(OverReason::YouStrikes)
        } else if self.g.bot.strikes >= 4 {
            Some(OverReason::BotStrikes)
        } else if n_locked(self.g.locked) >= 2 {
            Some(OverReason::Locks)
        } else {
            None
        };
        if reason.is_some() {
            self.g.over = reason;
            self.g.phase = Phase::Over;
            self.g.bot_white = None;
        }
        reason.is_some()
    }

    fn active_opts(&self) -> Option<ActiveOptions> {
        if self.g.phase != Phase::Active {
            return None;
        }
        let (you, bot) = self.states();
        Some(active_options(
            &you,
            &bot,
            self.g.bot_white.map(Mark::from),
            self.g.dice?,
        ))
    }

    pub fn view(&self) -> View {
        let (you, _) = self.states();
        let passive = match (self.g.phase, self.g.dice) {
            (Phase::Passive, Some(d)) => passive_options(&you, d).into_iter().map(Sq::from).collect(),
            _ => Vec::new(),
        };
        View {
            phase: self.g.phase,
            dice: self.g.dice,
            you: self.g.you.view(self.g.locked),
            bot: self.g.bot.view(self.g.locked),
            locked: self.g.locked,
            bot_white: if self.g.phase == Phase::Active {
                self.g.bot_white
            } else {
                None
            },
            options: self.active_opts().map(|o| OptionsView {
                selections: o
                    .selections
                    .iter()
                    .map(|s| s.iter().map(|&m| Sq::from(m)).collect())
                    .collect(),
                strike: o.strike,
                pass: o.pass,
            }),
            passive,
            over: self.g.over,
        }
    }

    /// Bot's turn: roll and decide its white sum (kept hidden until you confirm).
    pub fn bot_roll(&mut self) -> Result<(), String> {
        self.expect(Phase::BotToRoll)?;
        let dice = self.roll();
        let (you, bot) = self.states();
        self.g.bot_white = self.strategy.active_phase1(&bot, &[you], dice).map(Sq::from);
        self.g.dice = Some(dice);
        self.g.phase = Phase::Passive;
        Ok(())
    }

    /// Bot's turn: your white-sum mark (or pass), then the bot's phase 2.
    pub fn submit_passive(&mut self, pick: Option<Sq>) -> Result<RollRecord, String> {
        self.expect(Phase::Passive)?;
        let dice = self.g.dice.ok_or("no dice")?;
        let pre = self.states();
        let (you, _) = pre;
        if let Some(p) = pick {
            if !passive_options(&you, dice).contains(&Mark::from(p)) {
                return Err(format!("{p:?} is not a legal white-sum mark"));
            }
        }
        let bot_white = self.g.bot_white.take();
        if let Some(p) = pick {
            self.g.you.mark(p);
        }
        if let Some(w) = bot_white {
            self.g.bot.mark(w);
        }
        let mut rec = RollRecord {
            roller: Who::Bot,
            you: pick.into_iter().collect(),
            you_strike: false,
            bot_white,
            bot_colour: None,
            bot_strike: false,
            locked: self.settle_locks(),
        };
        if self.check_over() {
            self.record(pre, dice, &rec);
            return Ok(rec);
        }
        let (you, bot) = self.states();
        match self.strategy.active_phase2(&bot, &[you], dice, bot_white.is_some()) {
            Some(m) => {
                self.g.bot.mark(m.into());
                rec.bot_colour = Some(m.into());
            }
            None if bot_white.is_none() => {
                self.g.bot.strikes += 1;
                rec.bot_strike = true;
            }
            None => {}
        }
        rec.locked.extend(self.settle_locks());
        if !self.check_over() {
            self.g.phase = Phase::YouToRoll;
        }
        self.record(pre, dice, &rec);
        Ok(rec)
    }

    /// Your turn: roll; the bot decides (and reveals) its white sum at once.
    pub fn you_roll(&mut self) -> Result<(), String> {
        self.expect(Phase::YouToRoll)?;
        let dice = self.roll();
        let (you, bot) = self.states();
        self.g.bot_white = self.strategy.passive_phase1(&bot, &[you], dice, 0).map(Sq::from);
        self.g.dice = Some(dice);
        self.g.phase = Phase::Active;
        Ok(())
    }

    /// Your turn: confirm a selection (as a set), the strike box, or a pass.
    pub fn submit_active(&mut self, picks: &[Sq], strike: bool) -> Result<RollRecord, String> {
        self.expect(Phase::Active)?;
        let opts = self.active_opts().ok_or("no options")?;
        let pre = self.states();
        let dice = self.g.dice.ok_or("no dice")?;
        let chosen: Vec<Mark> = if strike {
            if !picks.is_empty() || !opts.strike {
                return Err("a strike is not allowed here".into());
            }
            Vec::new()
        } else if picks.is_empty() {
            if !opts.pass {
                return Err("select a square, or the strike box".into());
            }
            Vec::new()
        } else {
            let want: Vec<Mark> = picks.iter().map(|&p| p.into()).collect();
            opts.selections
                .iter()
                .find(|s| s.len() == want.len() && want.iter().all(|w| s.contains(w)))
                .cloned()
                .ok_or_else(|| format!("{picks:?} is not a legal selection"))?
        };
        for &m in &chosen {
            self.g.you.mark(m.into());
        }
        if strike {
            self.g.you.strikes += 1;
        }
        let bot_white = self.g.bot_white.take();
        if let Some(w) = bot_white {
            self.g.bot.mark(w);
        }
        let rec = RollRecord {
            roller: Who::You,
            you: chosen.iter().map(|&m| m.into()).collect(),
            you_strike: strike,
            bot_white,
            bot_colour: None,
            bot_strike: false,
            locked: self.settle_locks(),
        };
        if !self.check_over() {
            self.g.phase = Phase::BotToRoll;
        }
        self.record(pre, dice, &rec);
        Ok(rec)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bots::{make_bot, BotKind};
    use std::cell::RefCell;
    use std::rc::Rc;

    /// Bot that plays fixed marks and records the state it saw in phase 2.
    #[derive(Debug, Default)]
    struct Scripted {
        white: Option<Mark>,
        colour: Option<Mark>,
        passive: Option<Mark>,
        seen_phase2: Rc<RefCell<Option<State>>>,
    }

    impl Strategy for Scripted {
        fn active_phase1(&mut self, _: &State, _: &[State], _: [u8; 6]) -> Option<Mark> {
            self.white
        }
        fn active_phase2(&mut self, s: &State, _: &[State], _: [u8; 6], _: bool) -> Option<Mark> {
            *self.seen_phase2.borrow_mut() = Some(*s);
            self.colour
        }
        fn passive_phase1(&mut self, _: &State, _: &[State], _: [u8; 6], _: usize) -> Option<Mark> {
            self.passive
        }
    }

    fn sq(row: usize, n: u8) -> Sq {
        Sq { row, n }
    }

    fn sheet(rows: &[(usize, &[u8])], strikes: u8) -> SheetSnap {
        let mut marks = [[false; 11]; 4];
        for &(r, nums) in rows {
            for &n in nums {
                marks[r][cell_index(r, n)] = true;
            }
        }
        SheetSnap { marks, strikes }
    }

    fn snap(
        you: SheetSnap,
        bot: SheetSnap,
        locked: [bool; 4],
        phase: Phase,
        dice: [u8; 6],
        bot_white: Option<Sq>,
    ) -> Snapshot {
        Snapshot {
            you,
            bot,
            locked,
            phase,
            dice: Some(dice),
            bot_white,
            over: None,
            history: None,
        }
    }

    #[test]
    fn bot_starts_and_wrong_phase_calls_fail() {
        let mut s = Session::new(make_bot(BotKind::Student), 1);
        assert_eq!(s.view().phase, Phase::BotToRoll);
        assert!(s.you_roll().is_err());
        assert!(s.submit_active(&[], true).is_err());
        s.bot_roll().unwrap();
        assert_eq!(s.view().phase, Phase::Passive);
        assert!(s.view().bot_white.is_none(), "the bot's own white sum stays hidden");
        assert!(s.bot_roll().is_err());
    }

    #[test]
    fn random_games_finish_with_consistent_scores() {
        use rand::seq::SliceRandom;
        let mut pick_rng = SmallRng::seed_from_u64(99);
        for seed in 0..60u64 {
            let mut s = Session::new(make_bot(BotKind::Student), seed);
            let mut rolls = 0;
            while s.view().phase != Phase::Over {
                rolls += 1;
                assert!(rolls < 400, "game {seed} did not end");
                match s.view().phase {
                    Phase::BotToRoll => s.bot_roll().unwrap(),
                    Phase::Passive => {
                        let opts = s.view().passive;
                        let pick = if pick_rng.gen_bool(0.5) {
                            opts.choose(&mut pick_rng).copied()
                        } else {
                            None
                        };
                        s.submit_passive(pick).unwrap();
                    }
                    Phase::YouToRoll => s.you_roll().unwrap(),
                    Phase::Active => {
                        let o = s.view().options.unwrap();
                        if let Some(sel) = o.selections.choose(&mut pick_rng) {
                            s.submit_active(sel, false).unwrap();
                        } else if o.strike {
                            s.submit_active(&[], true).unwrap();
                        } else {
                            s.submit_active(&[], false).unwrap();
                        }
                    }
                    Phase::Over => unreachable!(),
                }
                let v = s.view();
                for p in [&v.you, &v.bot] {
                    assert_eq!(p.score, p.row_points.iter().sum::<isize>() - 5 * p.strikes as isize);
                    assert!(p.strikes <= 4);
                }
            }
            let v = s.view();
            assert!(v.over.is_some());
            assert!(v.you.strikes == 4 || v.bot.strikes == 4 || v.locked.iter().filter(|&&l| l).count() >= 2);
        }
    }

    #[test]
    fn your_white_lock_on_the_bots_turn_removes_that_colour_for_its_phase_two() {
        let seen = Rc::new(RefCell::new(None));
        let bot = Scripted {
            seen_phase2: seen.clone(),
            ..Default::default()
        };
        let you = sheet(&[(0, &[2, 3, 4, 5, 6])], 0);
        let s0 = snap(you, sheet(&[], 0), [false; 4], Phase::Passive, [6, 6, 1, 1, 1, 1], None);
        let mut s = Session::restore(s0, Box::new(bot), 1).unwrap();
        let rec = s.submit_passive(Some(sq(0, 12))).unwrap();
        assert_eq!(rec.locked, vec![0]);
        assert_eq!(
            seen.borrow().unwrap().locked()[0],
            true,
            "bot's phase 2 sees red locked"
        );
        // The bot marked nothing in either phase: it takes a strike.
        assert!(rec.bot_strike);
        assert_eq!(s.view().bot.strikes, 1);
        assert_eq!(s.view().phase, Phase::YouToRoll);
    }

    #[test]
    fn game_ending_at_phase_one_on_the_bots_turn_skips_its_phase_two() {
        let seen = Rc::new(RefCell::new(None));
        let bot = Scripted {
            seen_phase2: seen.clone(),
            ..Default::default()
        };
        let you = sheet(&[(0, &[2, 3, 4, 5, 6])], 0);
        let s0 = snap(
            you,
            sheet(&[], 0),
            [false, true, false, false],
            Phase::Passive,
            [6, 6, 1, 1, 1, 1],
            None,
        );
        let mut s = Session::restore(s0, Box::new(bot), 1).unwrap();
        let rec = s.submit_passive(Some(sq(0, 12))).unwrap();
        assert!(seen.borrow().is_none(), "no phase 2 after the game ended");
        assert!(!rec.bot_strike);
        let v = s.view();
        assert_eq!(v.phase, Phase::Over);
        assert_eq!(v.over, Some(OverReason::Locks));
        assert_eq!(v.bot.strikes, 0);
    }

    #[test]
    fn bot_reveal_ending_the_game_allows_only_white_or_pass_and_no_strike() {
        let bot = Scripted {
            passive: Some(Mark { row: 0, number: 12 }),
            ..Default::default()
        };
        let bot_sheet = sheet(&[(0, &[2, 3, 4, 5, 6])], 0);
        let s0 = snap(
            sheet(&[], 0),
            bot_sheet,
            [false, true, false, false],
            Phase::YouToRoll,
            [6, 6, 1, 1, 1, 1],
            None,
        );
        let mut s = Session::restore(s0, Box::new(bot), 1).unwrap();
        s.you_roll().unwrap();
        let v = s.view();
        assert_eq!(v.bot_white.map(|w| (w.row, w.n)), Some((0, 12)));
        let o = v.options.unwrap();
        assert!(o.pass && !o.strike);
        assert!(
            s.submit_active(&[], true).is_err(),
            "no strike when the game ends after the white sum"
        );
        let rec = s.submit_active(&[], false).unwrap();
        assert_eq!(rec.bot_white, Some(sq(0, 12)));
        assert_eq!(s.view().phase, Phase::Over);
        assert_eq!(s.view().you.strikes, 0);
    }

    #[test]
    fn illegal_selection_is_rejected_and_changes_nothing() {
        let mut s = Session::restore(
            snap(
                sheet(&[], 0),
                sheet(&[], 0),
                [false; 4],
                Phase::YouToRoll,
                [1, 1, 1, 1, 1, 1],
                None,
            ),
            Box::new(Scripted::default()),
            3,
        )
        .unwrap();
        s.you_roll().unwrap();
        let before = serde_json::to_string(&s.view()).unwrap();
        assert!(s.submit_active(&[sq(0, 11)], false).is_err());
        assert!(
            s.submit_active(&[], false).is_err(),
            "empty selection is not a pass here"
        );
        assert_eq!(serde_json::to_string(&s.view()).unwrap(), before);
    }

    #[test]
    fn fourth_strike_ends_the_game() {
        let s0 = snap(
            sheet(&[], 3),
            sheet(&[], 0),
            [false; 4],
            Phase::YouToRoll,
            [1, 1, 1, 1, 1, 1],
            None,
        );
        let mut s = Session::restore(s0, Box::new(Scripted::default()), 1).unwrap();
        s.you_roll().unwrap();
        let rec = s.submit_active(&[], true).unwrap();
        assert!(rec.you_strike);
        assert_eq!(s.view().over, Some(OverReason::YouStrikes));
    }

    #[test]
    fn snapshot_round_trips_in_every_phase() {
        let mut s = Session::new(make_bot(BotKind::Student), 5);
        for _ in 0..12 {
            let json = serde_json::to_string(&s.snapshot()).unwrap();
            let back: Snapshot = serde_json::from_str(&json).unwrap();
            let r = Session::restore(back, make_bot(BotKind::Student), 77).unwrap();
            assert_eq!(
                serde_json::to_string(&r.view()).unwrap(),
                serde_json::to_string(&s.view()).unwrap()
            );
            match s.view().phase {
                Phase::BotToRoll => s.bot_roll().unwrap(),
                Phase::Passive => {
                    s.submit_passive(None).unwrap();
                }
                Phase::YouToRoll => s.you_roll().unwrap(),
                Phase::Active => {
                    let o = s.view().options.unwrap();
                    match o.selections.first() {
                        Some(sel) => s.submit_active(&sel.clone(), false).unwrap(),
                        None => s.submit_active(&[], o.strike).unwrap(),
                    };
                }
                Phase::Over => break,
            }
        }
    }

    #[test]
    fn restore_rejects_malformed_snapshots() {
        let ok = snap(
            sheet(&[], 0),
            sheet(&[], 0),
            [false; 4],
            Phase::Active,
            [1; 6],
            Some(sq(0, 2)),
        );
        assert!(Session::restore(ok.clone(), make_bot(BotKind::Student), 1).is_ok());
        let bad: Vec<(&str, Snapshot)> = vec![
            (
                "row 4",
                Snapshot {
                    bot_white: Some(sq(4, 7)),
                    ..ok.clone()
                },
            ),
            (
                "n 1",
                Snapshot {
                    bot_white: Some(sq(0, 1)),
                    ..ok.clone()
                },
            ),
            (
                "n 13",
                Snapshot {
                    bot_white: Some(sq(2, 13)),
                    ..ok.clone()
                },
            ),
            (
                "die 0",
                Snapshot {
                    dice: Some([0, 1, 1, 1, 1, 1]),
                    ..ok.clone()
                },
            ),
            (
                "die 7",
                Snapshot {
                    dice: Some([1, 1, 1, 1, 1, 7]),
                    ..ok.clone()
                },
            ),
            (
                "active without dice",
                Snapshot {
                    dice: None,
                    ..ok.clone()
                },
            ),
            (
                "passive without dice",
                Snapshot {
                    phase: Phase::Passive,
                    dice: None,
                    ..ok.clone()
                },
            ),
            (
                "strikes",
                Snapshot {
                    you: sheet(&[], 9),
                    ..ok.clone()
                },
            ),
            (
                "over without reason",
                Snapshot {
                    phase: Phase::Over,
                    bot_white: None,
                    ..ok.clone()
                },
            ),
        ];
        for (what, s) in bad {
            assert!(Session::restore(s, make_bot(BotKind::Student), 1).is_err(), "{what}");
        }
        // Same through JSON, as the browser sends it: parse errors and bad values are errors, not panics.
        let json = serde_json::to_string(&ok).unwrap().replace("\"n\":2", "\"n\":200");
        let snap: Snapshot = serde_json::from_str(&json).unwrap();
        assert!(Session::restore(snap, make_bot(BotKind::Student), 1).is_err());
    }

    /// Plays a game to the end with random legal choices of yours.
    fn play_out(s: &mut Session, rng: &mut SmallRng) {
        use rand::seq::SliceRandom;
        loop {
            match s.view().phase {
                Phase::BotToRoll => s.bot_roll().unwrap(),
                Phase::YouToRoll => s.you_roll().unwrap(),
                Phase::Passive => {
                    let pick = s.view().passive.choose(rng).copied();
                    s.submit_passive(pick).unwrap();
                }
                Phase::Active => {
                    let o = s.view().options.unwrap();
                    match o.selections.choose(rng) {
                        Some(sel) => s.submit_active(&sel.clone(), false).unwrap(),
                        None => s.submit_active(&[], o.strike).unwrap(),
                    };
                }
                Phase::Over => return,
            }
        }
    }

    #[test]
    fn history_replays_to_the_same_game() {
        for seed in 0..10 {
            let mut rng = SmallRng::seed_from_u64(seed);
            let mut s = Session::new(make_bot(BotKind::Master), seed);
            play_out(&mut s, &mut rng);
            let g = crate::gamecode::replay(&s.code(BotKind::Master).unwrap()).unwrap();
            assert_eq!(g.bot, "master");
            assert!(g.bot_first);
            assert_eq!(g.rolls.len(), s.snapshot().history.unwrap().len());
            let last = &g.rolls.last().unwrap().after;
            assert_eq!(
                (last.you, last.opponent),
                (s.snapshot().you, s.snapshot().bot),
                "seed {seed}"
            );
            assert!(g.over.is_some());
        }
    }

    #[test]
    fn old_snapshots_have_no_code() {
        let mut s = Session::new(make_bot(BotKind::Master), 1);
        s.bot_roll().unwrap();
        let mut json: serde_json::Value = serde_json::to_value(s.snapshot()).unwrap();
        json.as_object_mut().unwrap().remove("history");
        let old: Snapshot = serde_json::from_value(json).unwrap();
        assert_eq!(old.history, None);
        let mut r = Session::restore(old, make_bot(BotKind::Master), 1).unwrap();
        play_out(&mut r, &mut SmallRng::seed_from_u64(1));
        assert!(r.code(BotKind::Master).is_err());
    }

    #[test]
    fn history_survives_a_snapshot_round_trip() {
        let mut rng = SmallRng::seed_from_u64(4);
        let mut s = Session::new(make_bot(BotKind::Master), 4);
        for _ in 0..9 {
            match s.view().phase {
                Phase::BotToRoll => s.bot_roll().unwrap(),
                Phase::YouToRoll => s.you_roll().unwrap(),
                Phase::Passive => drop(s.submit_passive(None).unwrap()),
                Phase::Active => {
                    let o = s.view().options.unwrap();
                    drop(s.submit_active(&[], o.strike).unwrap())
                }
                Phase::Over => break,
            }
        }
        let json = serde_json::to_string(&s.snapshot()).unwrap();
        let mut r = Session::restore(serde_json::from_str(&json).unwrap(), make_bot(BotKind::Master), 4).unwrap();
        play_out(&mut r, &mut rng);
        let g = crate::gamecode::replay(&r.code(BotKind::Master).unwrap()).unwrap();
        assert_eq!(g.rolls.last().unwrap().after.you, r.snapshot().you);
    }

    /// A seeded finished game's code, for screenshots of the review page.
    #[test]
    #[ignore]
    fn sample_code() {
        let mut s = Session::new(make_bot(BotKind::MasterPlus), 23);
        play_out(&mut s, &mut SmallRng::seed_from_u64(23));
        println!("{}", s.code(BotKind::MasterPlus).unwrap());
    }
}
