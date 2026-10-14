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
});
