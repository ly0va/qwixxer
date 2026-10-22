import type { Analysis, Answer, Engine, Luck, Roller, View } from '../engine/protocol';
import type { Store } from './controller.svelte';
import { decode, encode } from './link';
import * as P from './position';
import type { ExPosition, Who } from './position';

const SAVE = 'qwixxer.explorer.v1';
export const BAD_LINK = "That link doesn't hold a valid position.";
/** Spec "Controller": each analysis batch aims at about this long, within these sizes. */
export const BATCH_MS = 100;
export const MIN_BATCH = 8;
export const MAX_BATCH = 512;
/** Once ± is below this (2 points), rest as long as each batch took: half a core. */
export const REST_BELOW = 0.02;
/** A position's analysis stops here (± about 0.2–0.6 points, up to 0.9 late in the game). */
export const MAX_GAMES = 10_000;

/**
 * ± below REST_BELOW, judged with the plus-four estimate (two extra wins and
 * two extra losses) so a small one-sided sample, whose plain ± is 0, never
 * counts as precise.
 */
function precise(t: Analysis): boolean {
    return P.winChance(P.addAnalysis(t, { games: 4, win_sum: 2, win_sq_sum: 2, diff_sum: 0 })).pm < REST_BELOW;
}

/** The whole position, dice and roller included: what roll luck belongs to. */
export const luckKey = (p: ExPosition) => encode(p);

/** Spec "Web: state explorer": the position being explored and its answer. */
export class ExplorerController {
    pos = $state<ExPosition>(P.emptyPosition());
    answer = $state<Answer | null>(null);
    pending = $state(false);
    error = $state('');
    notice = $state('');
    picker = $state<number | null>(null);
    showAll = $state(false);
    /** Play-out totals and the position (dice aside) they belong to. */
    analysis = $state<{ key: string; totals: Analysis } | null>(null);
    /** A batch loop is running. */
    analysing = $state(false);
    /** Roll luck and the position (dice included) it belongs to. */
    luck = $state<{ key: string; value: Luck } | null>(null);

    private engine: Engine;
    private store: Store;
    private clipboard: (text: string) => Promise<void>;
    private later: (f: () => void, ms: number) => void;
    private inflight = false;
    private dirty = false;
    private luckInflight = false;
    private luckDirty = false;
    private noticeId = 0;
    private now: () => number;
    private active = false;
    private looping = false;
    private batch = MIN_BATCH;
    private batchKey = '';
    /** Ends the current rest between batches early (an edit or a pause). */
    private wake: (() => void) | null = null;

    constructor(
        engine: Engine,
        store: Store,
        clipboard: (text: string) => Promise<void> = (t) => navigator.clipboard.writeText(t),
        later: (f: () => void, ms: number) => void = (f, ms) => void setTimeout(f, ms),
        now: () => number = () => performance.now(),
    ) {
        this.engine = engine;
        this.store = store;
        this.clipboard = clipboard;
        this.later = later;
        this.now = now;
    }

    /**
     * The remembered position (saved as a link code), else an empty one.
     * Not evaluated yet: that waits for refresh(), when the explorer is entered.
     */
    init() {
        let saved: ExPosition | null = null;
        try {
            const raw = this.store.get(SAVE);
            saved = raw ? decode(raw) : null;
        } catch {
            saved = null;
        }
        this.pos = saved ?? P.emptyPosition();
    }

    /** Evaluate the position unless it has an answer or a call is already on it. */
    refresh() {
        if (this.luck?.key !== luckKey(this.pos) && !this.luckInflight) void this.askLuck();
        if (this.pending || (this.answer && !this.error)) return;
        void this.evaluate();
    }

    /** The Position card's numbers are current: analysed for this position, or a finished game (shown as a result). */
    get settled() {
        return P.isOver(this.pos) || this.analysis?.key === P.analysisKey(this.pos);
    }

    /** The explorer is the visible view (route and tab): analysis runs only then. */
    setActive(on: boolean) {
        this.active = on;
        this.wake?.();
        void this.analyse();
    }

    loadCode(code: string): boolean {
        const p = decode(code);
        if (!p) {
            this.flash(BAD_LINK);
            return false;
        }
        this.set(p);
        return true;
    }

    loadView(v: View) { this.set(P.fromView(v)); }

    tapSquare(who: Who, row: number, i: number) { this.set(P.toggleSquare(this.pos, who, row, i)); }
    tapStrike(who: Who, k: number) { this.set(P.tapStrike(this.pos, who, k)); }
    tapLock(row: number) { this.set(P.toggleLock(this.pos, row)); }
    clear(who: Who) { this.set(P.clearSheet(this.pos, who)); }
    clearAll() { this.set(P.clearSheets(this.pos)); }
    swap() { this.set(P.swapSides(this.pos)); }
    setRoller(r: Roller) { this.set(P.setRoller(this.pos, r)); }
    openPicker(i: number) { this.picker = this.picker === i ? null : i; }
    pickFace(v: number) {
        if (this.picker === null) return;
        const p = P.setDie(this.pos, this.picker, v);
        this.picker = P.nextUnsetDie(p, this.picker);
        this.set(p);
    }
    random() { this.picker = null; this.set(P.randomDice(this.pos)); }
    clearDice() { this.picker = null; this.set(P.clearDice(this.pos)); }

    linkFor(base: string) { return `${base}#/explore?p=${encode(this.pos)}`; }

    async copyLink(base: string) {
        try {
            await this.clipboard(this.linkFor(base));
            this.flash('Link copied');
        } catch {
            this.flash('Couldn’t copy the link');
        }
    }

    private flash(msg: string) {
        const id = ++this.noticeId;
        this.notice = msg;
        this.later(() => { if (this.noticeId === id) this.notice = ''; }, 3000);
    }

    private set(p: ExPosition) {
        this.pos = p;
        this.showAll = false;
        try { this.store.set(SAVE, encode(p)); } catch { /* storage unavailable */ }
        void this.evaluate();
        void this.askLuck();
        this.wake?.();
        void this.analyse();
    }

    /**
     * Spec "Engine": one call in flight; edits made meanwhile coalesce into
     * one call for the latest position once it returns. An answer for an
     * outdated position never lands.
     */
    private async evaluate() {
        this.pending = true;
        if (this.inflight) {
            this.dirty = true;
            return;
        }
        this.inflight = true;
        try {
            do {
                this.dirty = false;
                try {
                    const a = await this.engine.explore(P.toEngine(this.pos));
                    if (!this.dirty) {
                        this.answer = a;
                        this.error = '';
                    }
                } catch (err) {
                    // Keep the last answer: the error shows under it.
                    if (!this.dirty) this.error = err instanceof Error ? err.message : String(err);
                }
            } while (this.dirty);
        } finally {
            this.inflight = false;
            this.pending = false;
        }
    }

    /**
     * Spec "Worker and controller" (`2026-10-22-explorer-roll-luck-design.md`):
     * sent right after `explore`, so the move list comes first and the next
     * analysis batch after. One call in flight; edits meanwhile coalesce
     * into one for the latest position; a reply for an older one is
     * dropped. Without all six dice, or with the game over, there is none.
     */
    private async askLuck() {
        if (this.luckInflight) {
            this.luckDirty = true;
            return;
        }
        this.luckInflight = true;
        try {
            do {
                this.luckDirty = false;
                const pos = P.toEngine(this.pos);
                if (pos.dice === null || P.isOver(this.pos)) {
                    this.luck = null;
                    break;
                }
                const key = luckKey(this.pos);
                try {
                    const value = await this.engine.luck(pos);
                    if (luckKey(this.pos) === key) this.luck = { key, value };
                } catch {
                    // The move list shows engine errors.
                }
            } while (this.luckDirty);
        } finally {
            this.luckInflight = false;
        }
    }

    /**
     * Spec "Controller": one batch in flight while active; each reply adds
     * to the totals of the position it was asked for, and is dropped if
     * the position (dice aside) changed meanwhile. Errors stop the loop
     * quietly; the next edit or activation starts it again.
     */
    private async analyse() {
        if (this.looping || !this.active || P.isOver(this.pos)) return;
        this.looping = true;
        this.analysing = true;
        try {
            while (this.active && !P.isOver(this.pos) && !this.finished()) {
                const key = P.analysisKey(this.pos);
                // A new position's games may be much longer: start small again.
                if (key !== this.batchKey) {
                    this.batch = MIN_BATCH;
                    this.batchKey = key;
                }
                const prev = this.analysis?.key === key ? this.analysis.totals : P.ZERO_ANALYSIS;
                const count = Math.min(this.batch, MAX_GAMES - prev.games);
                const t0 = this.now();
                const r = await this.engine.analyze(P.toEngine(this.pos), prev.games, count);
                const ms = this.now() - t0;
                this.batch = Math.min(MAX_BATCH, Math.max(MIN_BATCH, Math.round((count * BATCH_MS) / Math.max(ms, 1))));
                const cur = this.analysis?.key === key ? this.analysis.totals : P.ZERO_ANALYSIS;
                if (P.analysisKey(this.pos) === key && cur.games === prev.games) {
                    const totals = P.addAnalysis(cur, r);
                    this.analysis = { key, totals };
                    if (precise(totals) && this.active && !this.finished()) await this.rest(ms);
                }
            }
        } catch {
            // Stop quietly: the move list shows engine errors.
        } finally {
            this.looping = false;
            this.analysing = false;
        }
    }

    /** The current position has all MAX_GAMES play-outs. */
    private finished() {
        return this.analysis?.key === P.analysisKey(this.pos) && this.analysis.totals.games >= MAX_GAMES;
    }

    private rest(ms: number) {
        return new Promise<void>((done) => {
            const end = () => {
                if (this.wake === end) this.wake = null;
                done();
            };
            this.wake = end;
            this.later(end, ms);
        });
    }
}
