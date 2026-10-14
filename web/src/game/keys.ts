import type { Phase } from '../engine/protocol';

/** What the focused element is, as far as page-wide keys care. */
export interface KeyTarget {
    /** Focus is in a select/input/textarea: leave every key but Escape to it. */
    field: boolean;
    /** A control has keyboard focus (`:focus-visible`): Enter/Space activate it natively. */
    keyboardFocus: boolean;
}

export type KeyAction = 'confirm' | 'roll' | 'clear' | 'swallow' | null;

/**
 * Spec "Implementation notes · Keyboard": Enter confirms, Space rolls, Escape
 * clears. Mouse clicks leave buttons focused, so Enter/Space are handled
 * page-wide unless the focus came from keyboard navigation. 'swallow' means
 * preventDefault without an action (Space off your roll must not re-click a
 * mouse-focused button). null means let the browser handle the key.
 */
export function keyAction(key: string, t: KeyTarget, phase: Phase | undefined): KeyAction {
    if (key === 'Escape') return 'clear';
    if (key !== 'Enter' && key !== ' ') return null;
    if (t.field || t.keyboardFocus) return null;
    if (key === 'Enter') return 'confirm';
    return phase === 'you_to_roll' ? 'roll' : 'swallow';
}
