import type { Phase } from '../engine/protocol';

/** What the focused element is, as far as page-wide keys care. */
export interface KeyTarget {
    /** Focus is in a select/input/textarea: leave every key but Escape to it. */
    field: boolean;
    /** A control has keyboard focus (`:focus-visible`): Enter/Space activate it natively. */
    keyboardFocus: boolean;
    /** A modal (the phone’s bot/log drawer, or the menu) is open over the page. */
    modal?: boolean;
    /** The explorer (or the about page) is showing: no game keys at all; Escape closes the menu or die picker. */
    explore?: boolean;
    /** The game review is showing: ← → Home End step through the game, ↑ ↓ pick Played or Best. */
    review?: boolean;
}

export type KeyAction = 'confirm' | 'roll' | 'clear' | 'close' | 'swallow' | 'prev' | 'next' | 'first' | 'last' | 'played' | 'best' | null;

const STEPS: Record<string, KeyAction> = { ArrowLeft: 'prev', ArrowRight: 'next', Home: 'first', End: 'last', ArrowUp: 'played', ArrowDown: 'best' };

/**
 * Spec "Implementation notes · Keyboard": Enter confirms, Space rolls, Escape
 * clears. Mouse clicks leave buttons focused, so Enter/Space are handled
 * page-wide unless the focus came from keyboard navigation. 'swallow' means
 * preventDefault without an action (Space off your roll must not re-click a
 * mouse-focused button). null means let the browser handle the key.
 * While a modal is open, game keys never reach the game: Escape closes it,
 * Enter/Space only act natively on keyboard-focused controls or fields.
 */
export function keyAction(key: string, t: KeyTarget, phase: Phase | undefined): KeyAction {
    if (t.explore) return key === 'Escape' ? 'close' : null;
    if (t.review) return key === 'Escape' ? 'close' : t.field || t.modal ? null : (STEPS[key] ?? null);
    if (t.modal) {
        if (key === 'Escape') return 'close';
        if (key !== 'Enter' && key !== ' ') return null;
        return t.field || t.keyboardFocus ? null : 'swallow';
    }
    if (key === 'Escape') return 'clear';
    if (key !== 'Enter' && key !== ' ') return null;
    if (t.field || t.keyboardFocus) return null;
    if (key === 'Enter') return 'confirm';
    return phase === 'you_to_roll' ? 'roll' : 'swallow';
}
