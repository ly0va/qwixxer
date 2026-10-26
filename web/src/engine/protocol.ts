// Mirrors web/src/session.rs (serde JSON). Keep in sync.
export interface Sq { row: number; n: number }
export type Phase = 'bot_to_roll' | 'passive' | 'you_to_roll' | 'active' | 'over';
export type OverReason = 'locks' | 'you_strikes' | 'bot_strikes';
export type BotKind = 'master_plus' | 'master' | 'student' | 'newbie';
export interface SheetView { marks: boolean[][]; strikes: number; score: number; row_points: number[] }
export interface OptionsView { selections: Sq[][]; strike: boolean; pass: boolean }
export interface View {
    phase: Phase;
    dice: number[] | null;
    you: SheetView;
    bot: SheetView;
    locked: boolean[];
    bot_white: Sq | null;
    options: OptionsView | null;
    passive: Sq[];
    over: OverReason | null;
}
export interface RollRecord {
    roller: 'you' | 'bot';
    you: Sq[];
    you_strike: boolean;
    bot_white: Sq | null;
    bot_colour: Sq | null;
    bot_strike: boolean;
    locked: number[];
}
export interface Step { view: View; record: RollRecord }
export type Roller = 'you' | 'opponent';
export interface SheetSnap { marks: boolean[][]; strikes: number }
/** Mirrors explorer.rs `Position`. `locked` = rows closed by tapping their lock. */
export interface Position { you: SheetSnap; opponent: SheetSnap; locked: boolean[]; dice: number[] | null; roller: Roller }
/** One side's expected final marks per row (red, yellow, green, blue) and strikes. */
export interface Ends { marks: number[]; strikes: number }
/** The win net's view of the position: P(you win) for whoever rolls next, the expected final difference, and each side's expected final marks and strikes. */
export interface Eval { p_win: number; margin: number; expected: { you: Ends; opponent: Ends } }
/** Mirrors explorer.rs `Played`: the position after your move and the opponent's predicted part (roll handed over, dice cleared). */
export interface Played { position: Position; opponent: Sq[]; opponent_strike: boolean }
/** `note` (only on Master+'s move when it isn't ranked first): why Master+ still plays it. */
export interface MoveOut { marks: Sq[]; strike: boolean; pass: boolean; p_win: number; bots: BotKind[]; note: 'rolled_out' | 'rule' | null }
/** The rolling opponent's predicted turn: white-sum mark, then colour mark or a strike (neither if the game ends first). */
export interface OpponentTurn { white: Sq | null; colour: Sq | null; strike: boolean }
/**
 * `opponent_white`: on your roll, the opponent's predicted simultaneous white-sum mark (`mark: null` = pass).
 * `opponent_turn`: on their roll, their predicted turn after your top-ranked move.
 */
export interface Answer {
    over: { you: number; opponent: number } | null;
    eval: Eval;
    moves: MoveOut[] | null;
    opponent_white: { mark: Sq | null } | null;
    opponent_turn: OpponentTurn | null;
}
/** Mirrors explorer.rs `Luck`: your P(win) after the roller's best turn for this roll, and a chunk of white pairs' share of its average over all rolls. */
export interface Luck { this_roll: number; average: number }
/** Mirrors gamecode.rs `ReviewRoll`: the position before the roll's marks (with dice and roller), what happened, the position after. */
export interface ReviewRoll { before: Position; record: RollRecord; after: Position }
/** Mirrors gamecode.rs `ReviewGame`; `over` is null for an unfinished game. */
export interface ReviewGame { bot: BotKind; bot_first: boolean; rolls: ReviewRoll[]; over: { you: number; opponent: number } | null }
/** Mirrors review.rs `BestMove`: a move of yours and the position it leaves. */
export interface BestMove { marks: Sq[]; strike: boolean; after: Position }
/** Mirrors review.rs `RollAnalysis`: your P(win) values for one roll (spec `2026-10-24-game-review-design.md`, "The numbers"). */
export interface RollAnalysis { start: number | null; start_var: number | null; this_roll: number; after_move: number; after_var: number; you_cost: number; you_best: BestMove; bot_cost: number }
/** Mirrors explorer.rs `Analysis`: play-out totals of one batch (or a running sum). */
export interface Analysis { games: number; win_sum: number; win_sq_sum: number; diff_sum: number }
/**
 * Why an engine call failed: `snapshot` when `restore` rejected the saved
 * game itself (bad JSON or values), `engine` for anything else (the wasm or
 * the worker failed to load, or a game call failed).
 */
export type EngineErrorKind = 'snapshot' | 'engine';
export class EngineError extends Error {
    kind: EngineErrorKind;
    constructor(message: string, kind: EngineErrorKind) {
        super(message);
        this.kind = kind;
    }
}
export interface Engine {
    /** Drop the current worker and start over with a fresh one (the game in it is lost). */
    reset(): void;
    /** A new game; `botFirst`: the bot rolls first, else you do. */
    start(bot: BotKind, seed: number, botFirst: boolean): Promise<View>;
    restore(bot: BotKind, snapshot: string, seed: number): Promise<View>;
    snapshot(): Promise<string>;
    botRoll(): Promise<View>;
    submitPassive(pick: Sq | null): Promise<Step>;
    youRoll(): Promise<View>;
    submitActive(picks: Sq[], strike: boolean): Promise<Step>;
    explore(position: Position): Promise<Answer>;
    /** Play-out games `start .. start + count` of the position (its dice ignored). */
    analyze(position: Position, start: number, count: number): Promise<Analysis>;
    /** Roll luck for a position with dice, white pairs `first .. first + count` of 21 (spec `2026-10-22-explorer-roll-luck-design.md`). */
    luck(position: Position, first: number, count: number): Promise<Luck>;
    /** Your move's 1-ply value on your roll: the next roll's luck average after it. */
    onePly(position: Position, marks: Sq[], strike: boolean): Promise<number>;
    /** Plays your move with the opponent's predicted part: the next position, roll handed over and dice cleared. */
    play(position: Position, marks: Sq[], strike: boolean): Promise<Played>;
    /** The finished game's code (rejects for a game saved before rolls were recorded). */
    gameCode(): Promise<string>;
    /** Replays a game code; rejects a malformed or illegal one. */
    reviewGame(code: string): Promise<ReviewGame>;
    /** Roll `k`'s review analysis. */
    reviewRoll(code: string, k: number): Promise<RollAnalysis>;
}
export const BOT_NAMES: Record<BotKind, string> = { master_plus: 'Master+', master: 'Master', student: 'Student', newbie: 'Newbie' };
