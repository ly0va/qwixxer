import { describe, expect, it } from 'vitest';
import { parseRoute } from './route';

describe('parseRoute', () => {
    it('reads the game, the explorer and an explorer link', () => {
        expect(parseRoute('')).toEqual({ view: 'play' });
        expect(parseRoute('#')).toEqual({ view: 'play' });
        expect(parseRoute('#/')).toEqual({ view: 'play' });
        expect(parseRoute('#/explore')).toEqual({ view: 'explore', code: null });
        expect(parseRoute('#/explore?p=1abc')).toEqual({ view: 'explore', code: '1abc' });
        expect(parseRoute('#/explore?x=1')).toEqual({ view: 'explore', code: null });
        expect(parseRoute('#/elsewhere')).toEqual({ view: 'play' });
    });
    it('reads the review page with or without a game code', () => {
        expect(parseRoute('#/review')).toEqual({ view: 'review', code: null });
        expect(parseRoute('#/review?g=1BAAAA')).toEqual({ view: 'review', code: '1BAAAA' });
        expect(parseRoute('#/review?x=1')).toEqual({ view: 'review', code: null });
    });
});
