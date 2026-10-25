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
    it('steps through a review with the arrows, Home and End, and nothing else', () => {
        const review = { ...mouse, review: true };
        expect(keyAction('ArrowLeft', review, undefined)).toBe('prev');
        expect(keyAction('ArrowRight', review, undefined)).toBe('next');
        expect(keyAction('Home', review, undefined)).toBe('first');
        expect(keyAction('End', review, undefined)).toBe('last');
        expect(keyAction('ArrowUp', review, undefined)).toBe('played');
        expect(keyAction('ArrowDown', review, undefined)).toBe('best');
        expect(keyAction('ArrowDown', { ...field, review: true }, undefined)).toBe(null);
        // The open menu keeps its keys: the game behind it doesn't step.
        for (const k of ['ArrowLeft', 'ArrowDown', 'Home', 'End']) expect(keyAction(k, { ...review, modal: true }, undefined)).toBe(null);
        expect(keyAction('Escape', { ...review, modal: true }, undefined)).toBe('close');
        expect(keyAction('Enter', review, 'you_to_roll')).toBe(null);
        expect(keyAction(' ', review, 'you_to_roll')).toBe(null);
        expect(keyAction('ArrowLeft', { ...field, review: true }, undefined)).toBe(null);
        expect(keyAction('Escape', review, undefined)).toBe('close');
    });
    it('ignores other keys', () => {
        expect(keyAction('a', mouse, 'active')).toBe(null);
    });
    describe('with the drawer open', () => {
        const mModal = { ...mouse, modal: true };
        const kModal = { ...kbd, modal: true };
        const fModal = { ...field, modal: true };
        it('closes the drawer on Escape instead of clearing', () => {
            expect(keyAction('Escape', mModal, 'active')).toBe('close');
            expect(keyAction('Escape', fModal, 'active')).toBe('close');
        });
        it('never confirms or rolls underneath', () => {
            expect(keyAction('Enter', mModal, 'active')).toBe('swallow');
            expect(keyAction(' ', mModal, 'you_to_roll')).toBe('swallow');
        });
        it('leaves keyboard-focused drawer controls and fields native', () => {
            expect(keyAction('Enter', kModal, 'active')).toBe(null);
            expect(keyAction(' ', kModal, 'you_to_roll')).toBe(null);
            expect(keyAction('Enter', fModal, 'active')).toBe(null);
            expect(keyAction('a', mModal, 'active')).toBe(null);
        });
    });
    it('never confirms or rolls the hidden game from the explorer', () => {
        const ex = { field: false, keyboardFocus: false, explore: true };
        expect(keyAction('Enter', ex, 'active')).toBe(null);
        expect(keyAction(' ', ex, 'you_to_roll')).toBe(null);
        expect(keyAction('Escape', ex, 'active')).toBe('close');
    });
});
