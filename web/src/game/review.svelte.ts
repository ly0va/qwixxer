import type { Engine, ReviewGame, RollAnalysis } from '../engine/protocol';
import type { Store } from './controller.svelte';
import type { Analyses, Side } from './review';

/** The last finished game's code (spec "The page": the Review tab opens it). */
export const LAST_GAME = 'qwixxer.lastGame';

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
    error = $state<string | null>(null);
    /** Opened from someone's link: "Player" instead of "You". */
    shared = $state(false);

    private engine: Engine;
    private store: Store;
    private active = false;
    /** Bumped per loaded game: replies for an older one are dropped. */
    private gen = 0;
    /** The generation whose analysis loop is running, if any. */
    private running: number | null = null;

    constructor(engine: Engine, store: Store) {
        this.engine = engine;
        this.store = store;
    }

    get points() {
        return this.game ? 2 * this.game.rolls.length + 1 : 1;
    }

    /** Shows the game `code` (null: the last finished game, if any). The same code again changes nothing. */
    async load(code: string | null, shared: boolean) {
        const c = code ?? this.store.get(LAST_GAME);
        if (c === null) {
            this.gen++;
            this.code = null;
            this.game = null;
            this.error = null;
            this.shared = false;
            return;
        }
        if (c === this.code && (this.game || this.error)) return;
        const gen = ++this.gen;
        this.code = c;
        this.shared = shared;
        this.game = null;
        this.an = [];
        this.sel = 0;
        this.side = 'played';
        this.error = null;
        try {
            const g = await this.engine.reviewGame(c);
            if (gen !== this.gen) return;
            this.game = g;
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
            void this.load(c, this.shared);
            return;
        }
        this.error = null;
        void this.analyse();
    }

    select(j: number) {
        this.sel = Math.max(0, Math.min(this.points - 1, j));
        this.side = 'played';
    }
    step(d: 1 | -1) { this.select(this.sel + d); }
    first() { this.select(0); }
    last() { this.select(this.points - 1); }
    setSide(s: Side) { this.side = s; }

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
