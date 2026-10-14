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
    start(bot: BotKind, seed: number): Promise<View>;
    restore(bot: BotKind, snapshot: string, seed: number): Promise<View>;
    snapshot(): Promise<string>;
    botRoll(): Promise<View>;
    submitPassive(pick: Sq | null): Promise<Step>;
    youRoll(): Promise<View>;
    submitActive(picks: Sq[], strike: boolean): Promise<Step>;
}
export const BOT_NAMES: Record<BotKind, string> = { master_plus: 'Master+', master: 'Master', student: 'Student', newbie: 'Newbie' };
