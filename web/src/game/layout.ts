/** Desktop layout design size (spec "Implementation notes · Window sizes"). */
export const DESK_W = 1280;
export const DESK_H = 800;

/** The least height each desktop view fits in: the game's empty space and move log give way, the explorer has little slack, the review none. */
export const DESK_MIN_H = { play: 640, explore: 760, review: 800 } as const;

/**
 * Fit for the 1280-wide desktop layout. A short window first takes height
 * from the view (down to `minHeight`, the rest being empty space and
 * scrolling lists), then scales the whole desk down so it fits without
 * scrolling. `height` is the desk's height in design pixels, at most 800.
 */
export function deskFit(innerWidth: number, innerHeight: number, minHeight: number): { scale: number; height: number } {
    const scale = Math.max(0.1, Math.min(1, innerWidth / DESK_W, innerHeight / minHeight));
    return { scale, height: Math.min(DESK_H, Math.max(minHeight, Math.floor(innerHeight / scale))) };
}

/**
 * Phone layout sizes (spec "Visual design · Phone"): 48 px squares on a
 * 390×844 phone, shrinking on short or narrow screens. Ruling (Task 9 review):
 * playability beats the spec's ~36 px floor, so squares go down to 28 px, and
 * if even that doesn't fit the page scrolls instead of clipping the sheet.
 */
export interface PhoneLayout {
    /** Sheet square size, px. */
    size: number;
    /** Die size in the compact dice row, px. */
    die: number;
    /** Square size of the bot's sheet in the drawer, px. */
    botSize: number;
    /** Tightened spacing (8 px gaps, 12 px bottom padding) to save 10 px. */
    short: boolean;
    /** Even the 28 px floor doesn't fit: let the page scroll vertically. */
    scroll: boolean;
}

/**
 * Height outside the 12 squares of a column: page padding 12 + 16, header 44,
 * dice row 48, message line 34, three 10 px gaps, sheet overhead 59 (paper
 * padding 2×6, band padding 2×4, 12 inner gaps of 3, a 3 px spacer).
 */
export const PHONE_FIXED = 243;
/** Same with 8 px gaps and 12 px bottom padding. */
export const PHONE_FIXED_SHORT = 233;
const MIN = 28;
const MAX = 48;

export function phoneLayout(innerWidth: number, innerHeight: number): PhoneLayout {
    const w = Math.min(innerWidth, 430); // .phone max-width
    // Width outside the 4 columns: page padding 2×12, sheet 4×8 + 3×6 + 2×6 = 62, body gap 10, side column ≥ 98.
    const cap = Math.max(MIN, Math.min(MAX, Math.floor((w - 194) / 4)));
    // Dice row: Roll / White slot 110 + 6 dice + 6 gaps of 6 within the page padding.
    const die = Math.max(20, Math.min(38, Math.floor((w - 24 - 110 - 36) / 6)));
    // Drawer bot sheet: 12 squares + 36 inner gaps + 2 spacer + 6 band padding + 24 panel + 28 drawer padding.
    const botSize = Math.max(16, Math.min(24, Math.floor((w - 96) / 12)));
    if (PHONE_FIXED + 12 * cap <= innerHeight) return { size: cap, die, botSize, short: false, scroll: false };
    const size = Math.max(MIN, Math.min(cap, Math.floor((innerHeight - PHONE_FIXED_SHORT) / 12)));
    return { size, die, botSize, short: true, scroll: PHONE_FIXED_SHORT + 12 * size > innerHeight };
}

/**
 * Explorer phone: two compact SheetColumns side by side. Each is 4 s + 41 px
 * wide (paper padding 2×4, band padding 4×2×3, 3 gaps of 3), plus an 8 px gap
 * and 2×12 page padding: 8 s + 114 within the (≤ 430 px) column.
 */
export function explorerPhoneSize(innerWidth: number): number {
    const w = Math.min(innerWidth, 430);
    return Math.max(24, Math.min(40, Math.floor((w - 114) / 8)));
}
