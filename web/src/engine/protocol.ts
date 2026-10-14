// Mirrors web/src/session.rs (serde JSON). Keep in sync.
export interface Sq { row: number; n: number }
export type Phase = 'bot_to_roll' | 'passive' | 'you_to_roll' | 'active' | 'over';
export type OverReason = 'locks' | 'you_strikes' | 'bot_strikes';
export type BotKind = 'master_plus' | 'master' | 'student';
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
export interface Engine {
    start(bot: BotKind, seed: number): Promise<View>;
    restore(bot: BotKind, snapshot: string, seed: number): Promise<View>;
    snapshot(): Promise<string>;
    botRoll(): Promise<View>;
    submitPassive(pick: Sq | null): Promise<Step>;
    youRoll(): Promise<View>;
    submitActive(picks: Sq[], strike: boolean): Promise<Step>;
}
export const BOT_NAMES: Record<BotKind, string> = { master_plus: 'Master+', master: 'Master', student: 'Student' };
