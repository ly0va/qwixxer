import { describe, expect, it } from 'vitest';
import { keyAction } from './keys';

const mouse = { field: false, keyboardFocus: false };
const kbd = { field: false, keyboardFocus: true };
const field = { field: true, keyboardFocus: false };

describe('keyAction', () => {
    it('always clears on Escape', () => {
        expect(keyAction('Escape', mouse, 'active')).toBe('clear');
        expect(keyAction('Escape', kbd, 'active')).toBe('clear');
        expect(keyAction('Escape', field, 'passive')).toBe('clear');
    });
    it('confirms on Enter even when a button has mouse focus', () => {
        expect(keyAction('Enter', mouse, 'active')).toBe('confirm');
        expect(keyAction('Enter', mouse, 'you_to_roll')).toBe('confirm');
    });
    it('rolls on Space only on your roll, and swallows it otherwise', () => {
        expect(keyAction(' ', mouse, 'you_to_roll')).toBe('roll');
        expect(keyAction(' ', mouse, 'active')).toBe('swallow');
    });
    it('defers to native behaviour for keyboard focus and form fields', () => {
        expect(keyAction('Enter', kbd, 'active')).toBe(null);
        expect(keyAction(' ', kbd, 'you_to_roll')).toBe(null);
        expect(keyAction('Enter', field, 'active')).toBe(null);
        expect(keyAction(' ', field, 'you_to_roll')).toBe(null);
    });
    it('ignores other keys', () => {
        expect(keyAction('a', mouse, 'active')).toBe(null);
    });
});
