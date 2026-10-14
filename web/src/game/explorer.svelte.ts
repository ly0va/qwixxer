import type { Answer, Engine, Roller, View } from '../engine/protocol';
import type { Store } from './controller.svelte';
import { decode, encode } from './link';
import * as P from './position';
import type { ExPosition, Who } from './position';

const SAVE = 'qwixxer.explorer.v1';
export const BAD_LINK = "That link doesn't hold a valid position.";

/** Spec "Web: state explorer": the position being explored and its answer. */
export class ExplorerController {
    pos = $state<ExPosition>(P.emptyPosition());
    answer = $state<Answer | null>(null);
    pending = $state(false);
    error = $state('');
    notice = $state('');
    picker = $state<number | null>(null);
    showAll = $state(false);

    private engine: Engine;
    private store: Store;
    private clipboard: (text: string) => Promise<void>;
    private later: (f: () => void, ms: number) => void;
    private inflight = false;
    private dirty = false;
    private noticeId = 0;

    constructor(
        engine: Engine,
        store: Store,
        clipboard: (text: string) => Promise<void> = (t) => navigator.clipboard.writeText(t),
        later: (f: () => void, ms: number) => void = (f, ms) => void setTimeout(f, ms),
    ) {
        this.engine = engine;
        this.store = store;
        this.clipboard = clipboard;
        this.later = later;
    }

    /** The remembered position (saved as a link code), else an empty one. */
    init() {
        let saved: ExPosition | null = null;
        try {
            const raw = this.store.get(SAVE);
            saved = raw ? decode(raw) : null;
        } catch {
            saved = null;
        }
        this.set(saved ?? P.emptyPosition());
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
}
