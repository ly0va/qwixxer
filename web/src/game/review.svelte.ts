import type { Engine, Luck, ReviewGame, RollAnalysis } from '../engine/protocol';
import type { Store } from './controller.svelte';
import { LUCK_CHUNK, WHITE_PAIRS } from './explorer.svelte';
import { gameHref, type Analyses, type GraphKind, type Side } from './review';

/** The last finished game's code (spec "The page": the Review tab opens it). */
export const LAST_GAME = 'qwixxer.lastGame';
/** Whether the selected roll's luck histogram is open: '1' or ''. */
const LUCK_OPEN = 'qwixxer.review.luck-open';

/**
 * Spec `2026-10-24-game-review-design.md`: the game being reviewed, its
 * per-roll analyses (filled in roll by roll while the page is on screen),
 * and the selected point.
 */
export class ReviewController {
    code = $state<string | null>(null);
    game = $state<ReviewGame | null>(null);
    an = $state<Analyses>([]);
    /** The selected point: 0 = start, 2k+1 after roll k's dice, 2k+2 after its move. */
    sel = $state(0);
    side = $state<Side>('played');
    /** The graph card's view (spec "The graph"): win chance until you pick another. */
    graph = $state<GraphKind>('win');
    error = $state<string | null>(null);
    /** "Share game"'s result, shown for two seconds. */
    notice = $state('');
    /** The luck chip's histogram is open (remembered across pages and reloads). */
    luckOpen = $state(false);
    /** The last roll whose every-roll luck came in (the histogram's data), by roll index. */
    rollLuck = $state<{ k: number; value: Luck } | null>(null);

    private engine: Engine;
    private store: Store;
    private active = false;
    /** Bumped per loaded game: replies for an older one are dropped. */
    private gen = 0;
    /** The generation whose analysis loop is running, if any. */
    private running: number | null = null;
    private noticeTimer: ReturnType<typeof setTimeout> | undefined;
    /** This game's every-roll luck by roll index, kept while it is shown. */
    private luckCache = new Map<number, Luck>();
    private luckInflight = false;
    private luckDirty = false;

    constructor(engine: Engine, store: Store) {
        this.engine = engine;
        this.store = store;
        try { this.luckOpen = store.get(LUCK_OPEN) === '1'; } catch { /* storage unavailable */ }
    }

    /** The selected roll's index, -1 at the start. */
    get roll() {
        return this.sel === 0 ? -1 : Math.floor((this.sel - 1) / 2);
    }

    get points() {
        return this.game ? 2 * this.game.rolls.length + 1 : 1;
    }

    /** Shows the game `code` (null: the last finished game, if any). The same code again changes nothing. */
    async load(code: string | null) {
        const c = code ?? (this.store.get(LAST_GAME) || null);
        if (c === null) {
            this.gen++;
            this.code = null;
            this.game = null;
            this.error = null;
            return;
        }
        if (c === this.code && (this.game || this.error)) return;
        const gen = ++this.gen;
        this.code = c;
        this.game = null;
        this.an = [];
        this.sel = 0;
        this.side = 'played';
        this.error = null;
        this.luckCache = new Map();
        this.rollLuck = null;
        try {
            const g = await this.engine.reviewGame(c);
            if (gen !== this.gen) return;
            this.game = g;
            void this.askLuck();
        } catch (err) {
            if (gen === this.gen) this.error = err instanceof Error ? err.message : String(err);
            return;
        }
        void this.analyse();
    }

    /** The review page is the visible view: analysis runs only then. */
    setActive(on: boolean) {
        this.active = on;
        if (on) void this.analyse();
    }

    /** After an error: reopen the game if it never opened (the engine may have failed to load), else resume the analysis. */
    retry() {
        if (!this.game && this.code) {
            const c = this.code;
            this.code = null;
            void this.load(c);
            return;
        }
        this.error = null;
        void this.analyse();
    }

    /** Copies the game's link (`base`: the app's address). */
    async share(base = location.origin + location.pathname, copy = (t: string) => navigator.clipboard.writeText(t)) {
        if (!this.code) return;
        let text: string;
        try {
            await copy(base + gameHref(this.code));
            text = 'Link copied';
        } catch {
            text = 'Couldn’t copy the link';
        }
        clearTimeout(this.noticeTimer);
        this.notice = text;
        this.noticeTimer = setTimeout(() => (this.notice = ''), 2000);
    }

    select(j: number) {
        this.sel = Math.max(0, Math.min(this.points - 1, j));
        this.side = 'played';
        void this.askLuck();
    }
    step(d: 1 | -1) { this.select(this.sel + d); }
    first() { this.select(0); }
    last() { this.select(this.points - 1); }
    setSide(s: Side) { this.side = s; }
    /** Show the played or the best outcome of the selected roll's move (from its dice point, move there first). */
    show(s: Side) {
        if (this.sel === 0) return;
        if (this.sel % 2 === 1) this.select(this.sel + 1);
        this.side = s;
    }

    /** The luck chip: open or close the selected roll's histogram. */
    toggleLuck() {
        this.luckOpen = !this.luckOpen;
        try { this.store.set(LUCK_OPEN, this.luckOpen ? '1' : ''); } catch { /* storage unavailable */ }
        void this.askLuck();
    }

    /**
     * The histogram's data for the selected roll, while it is open: the
     * explorer's roll luck of the position before the roll, in the same
     * chunks, kept per roll. Stepping to another roll mid-way starts over
     * for that one.
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
                const { roll: k, game, gen } = this;
                if (!this.luckOpen || !game || k < 0) break;
                const kept = this.luckCache.get(k);
                if (kept) {
                    this.rollLuck = { k, value: kept };
                    break;
                }
                const pos = game.rolls[k].before;
                const value: Luck = { this_roll: 0, average: 0, values: [] };
                try {
                    for (let first = 0; first < WHITE_PAIRS && !this.luckDirty; first += LUCK_CHUNK) {
                        const part = await this.engine.luck(pos, first, Math.min(LUCK_CHUNK, WHITE_PAIRS - first));
                        value.this_roll = part.this_roll;
                        value.average += part.average;
                        value.values = value.values.concat(part.values);
                    }
                    if (!this.luckDirty && gen === this.gen) {
                        this.luckCache.set(k, value);
                        this.rollLuck = { k, value };
                    }
                } catch {
                    // The page shows engine errors from the analysis.
                }
            } while (this.luckDirty);
        } finally {
            this.luckInflight = false;
        }
    }

    private async analyse() {
        const gen = this.gen;
        if (this.running === gen) return;
        this.running = gen;
        try {
            while (this.active && gen === this.gen && this.game && !this.error) {
                const k = this.game.rolls.findIndex((_, i) => !this.an[i]);
                if (k < 0) break;
                const code = this.code!;
                let a: RollAnalysis;
                try {
                    a = await this.engine.reviewRoll(code, k);
                } catch (err) {
                    if (gen === this.gen) this.error = err instanceof Error ? err.message : String(err);
                    break;
                }
                if (gen !== this.gen) break;
                this.an[k] = a;
            }
        } finally {
            if (this.running === gen) this.running = null;
        }
    }
}
