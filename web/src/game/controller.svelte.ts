import { EngineError, type BotKind, type Engine, type Sq, type Step, type View } from '../engine/protocol';
import { LAST_GAME } from './review.svelte';
import { clickable, confirmState, has, isTerminal, logEntry, NAME, sqKey, strikeOpen, type LogEntry } from './rules';

/** Spec "Pacing": minimum durations in ms. */
export const PACE = { tumble: 600, tick: 75, think: 500, stagger: 450, handoff: 900 };

export interface Store { get(key: string): string | null; set(key: string, value: string): void }

const SAVE = 'qwixxer.save.v1';
const PREFS = 'qwixxer.prefs.v1';

interface Saved { snapshot: string; log: LogEntry[]; rollNo: number; bot: BotKind; nextBot: BotKind; botFirst?: boolean }

/** Who rolls first in a new game: take turns, or always the same side. */
export type FirstRoll = 'alternate' | 'you' | 'bot';
const NEXT_FIRST_ROLL: Record<FirstRoll, FirstRoll> = { alternate: 'you', you: 'bot', bot: 'alternate' };

const randomDice = () => Array.from({ length: 6 }, () => 1 + Math.floor(Math.random() * 6));

export class GameController {
    view = $state<View | null>(null);
    sel = $state<Sq[]>([]);
    strikeSel = $state(false);
    tip = $state('');
    error = $state('');
    rolling = $state(false);
    shownDice = $state<number[] | null>(null);
    /** Your rolled dice, shown during the bot's think before the view lands (spec "Pacing"). */
    pendingDice = $state<number[] | null>(null);
    thinking = $state(false);
    busy = $state(false);
    committing = $state(false);
    log = $state<LogEntry[]>([]);
    fresh = $state<Record<string, 'draw' | 'ink'>>({});
    hidden = $state<string[]>([]);
    rollNo = $state(0);
    theme = $state<'light' | 'dark'>('light');
    bot = $state<BotKind>('master_plus');
    nextBot = $state<BotKind>('master_plus');
    firstRoll = $state<FirstRoll>('alternate');
    /** Whether the bot rolled first in the current game (before any game: false, so the first has the bot first). */
    botFirst = $state(false);
    /** The header "⋯" menu is open (UI state: page-wide keys treat it as a modal). */
    menuOpen = $state(false);
    /** The last finished game's code, for review (null: none, or a game saved before rolls were recorded). */
    lastCode = $state<string | null>(null);

    // Plain fields assigned in the constructor: `.svelte.ts` files do not
    // support TypeScript parameter properties.
    private engine: Engine;
    private store: Store;
    private sleep: (ms: number) => Promise<void>;
    private seed: () => number;
    /** An engine call failed: New game starts over with a fresh worker. */
    private stale = false;

    constructor(
        engine: Engine,
        store: Store,
        sleep: (ms: number) => Promise<void> = (ms) => new Promise((r) => setTimeout(r, ms)),
        seed: () => number = () => Math.floor(Math.random() * 2 ** 32),
    ) {
        this.engine = engine;
        this.store = store;
        this.sleep = sleep;
        this.seed = seed;
    }

    /** Escape key: drop the current selection and strike. */
    clearSelection() {
        if (this.busy) return;
        this.sel = [];
        this.strikeSel = false;
        this.tip = '';
    }

    async init() {
        this.lastCode = this.store.get(LAST_GAME) || null;
        const prefs = this.read<{ theme: 'light' | 'dark'; nextBot: BotKind; firstRoll: FirstRoll }>(PREFS);
        if (prefs) {
            this.theme = prefs.theme ?? 'light';
            this.nextBot = prefs.nextBot ?? 'master_plus';
            this.firstRoll = prefs.firstRoll in NEXT_FIRST_ROLL ? prefs.firstRoll : 'alternate';
        }
        const saved = this.read<Saved>(SAVE);
        if (saved && (await this.resume(saved, prefs?.nextBot))) return;
        return this.newGame();
    }

    /**
     * Restore a saved game. If the engine rejects the snapshot itself
     * (corrupt or old format), discard the save and report false so a new
     * game starts; otherwise a bad save would break every page load. If the
     * engine fails to load, retry once on a fresh worker, then keep the save
     * and show the error (true: no new game overwrites the save).
     */
    private async resume(saved: Saved, prefBot: BotKind | undefined): Promise<boolean> {
        let view: View | null = null;
        this.busy = true; // block newGame/confirm while the engine restores
        try {
            for (let attempt = 0; !view; attempt++) {
                try {
                    view = await this.engine.restore(saved.bot, saved.snapshot, this.seed());
                } catch (e) {
                    if (e instanceof EngineError && e.kind === 'snapshot') {
                        this.store.set(SAVE, '');
                        return false;
                    }
                    if (attempt > 0) {
                        this.error = `${e instanceof Error ? e.message : String(e)}. Your saved game is kept: reload the page to continue it.`;
                        this.stale = true;
                        return true;
                    }
                    this.engine.reset();
                }
            }
        } finally {
            this.busy = false;
        }
        const restored = view;
        await this.guard(async () => {
            this.bot = saved.bot;
            this.nextBot = prefBot ?? saved.nextBot;
            this.botFirst = saved.botFirst ?? true;
            this.log = saved.log;
            this.rollNo = saved.rollNo;
            this.view = restored;
            // A reload can land before the code of the game it ended was kept.
            if (restored.phase === 'over') void this.keepCode();
            if (restored.phase === 'bot_to_roll') await this.botTurn();
        });
        return true;
    }

    async newGame() {
        if (this.busy) return;
        await this.guard(async () => {
            if (this.stale) {
                this.engine.reset();
                this.stale = false;
            }
            this.bot = this.nextBot;
            this.thinking = false;
            this.pendingDice = null;
            this.log = [];
            this.rollNo = 0;
            this.sel = [];
            this.strikeSel = false;
            this.tip = '';
            this.fresh = {};
            this.hidden = [];
            const botFirst = this.nextBotFirst;
            this.view = await this.engine.start(this.bot, this.seed(), botFirst);
            this.botFirst = botFirst;
            if (botFirst) await this.botTurn();
        });
    }

    /** Your Roll button. */
    async roll() {
        if (this.busy || this.view?.phase !== 'you_to_roll') return;
        await this.guard(async () => {
            this.fresh = {};
            const rollNo = this.rollNo + 1;
            const result = this.engine.youRoll();
            let failed = false;
            // The engine has rolled: save now, so a reload during the tumble
            // or the bot's think restores these dice instead of a fresh roll.
            result.then(() => this.save(rollNo), () => (failed = true));
            // Tumble, then the bot thinks for at least PACE.think while the
            // engine computes; the last tumble faces stay up until the real
            // dice arrive, so the tray never falls back to blank.
            const anim = async () => {
                await this.tumble(true);
                if (failed) return;
                this.thinking = true;
                result.then((r) => (this.pendingDice = r.dice), () => {});
                await Promise.all([result.catch(() => {}), this.sleep(PACE.think)]);
            };
            const v = await this.settled(result, anim());
            this.thinking = false;
            this.view = v;
            this.pendingDice = null;
            this.shownDice = null;
            this.rollNo = rollNo;
            this.save();
        });
    }

    tapSquare(s: Sq) {
        const v = this.view;
        if (!v) return;
        if (v.phase !== 'active' && v.phase !== 'passive') {
            this.tip = v.phase === 'you_to_roll' ? 'Press Roll to start your turn.' : '';
            return;
        }
        if (this.busy) return;
        if (has(this.sel, s)) {
            this.sel = this.sel.filter((x) => x.row !== s.row || x.n !== s.n);
            this.tip = '';
            return;
        }
        if (clickable(v, this.sel, this.strikeSel, s)) {
            this.sel = [...this.sel, s];
            this.strikeSel = false;
            this.tip = '';
            return;
        }
        const name = NAME[s.row].toLowerCase();
        const alone = v.phase === 'passive' ? has(v.passive, s) : !!v.options?.selections.some((x) => has(x, s));
        if (this.strikeSel) this.tip = 'Clear the strike first to mark a square.';
        else if (this.sel.length === 1 && alone) this.tip = 'Tap your selected square to unselect it first.';
        else if (v.phase === 'active' && v.bot_white && isTerminal(v.bot_white) && v.bot_white.row === s.row)
            this.tip = `${NAME[s.row]} is locking: the bot marked ${name} ${v.bot_white.n}. Your white sum may still go in ${name}; a colour mark can’t.`;
        else if (v.phase === 'passive')
            this.tip = `On the bot’s turn you may only mark the white sum (${(v.dice?.[0] ?? 0) + (v.dice?.[1] ?? 0)}), or pass.`;
        else if (this.sel.length === 2) this.tip = 'You already picked two squares. Tap one to unselect it.';
        else this.tip = 'That square isn’t available with this roll.';
    }

    tapStrike() {
        const v = this.view;
        if (!v || this.busy) return;
        if (v.phase !== 'active') {
            this.tip = v.phase === 'passive' ? 'You can’t take a strike on the bot’s turn.' : '';
            return;
        }
        if (this.sel.length > 0) this.tip = 'Unselect your squares first to take a strike.';
        else if (!strikeOpen(v, this.sel)) this.tip = 'The game ends after the white sum, so no strike this turn.';
        else {
            this.strikeSel = !this.strikeSel;
            this.tip = '';
        }
    }

    async confirm() {
        const v = this.view;
        if (!v || this.busy) return;
        if (v.phase === 'over') return this.newGame();
        if (v.phase === 'passive') return this.resolvePassive();
        if (v.phase !== 'active') {
            this.tip = v.phase === 'you_to_roll' ? 'Press Roll to start your turn.' : '';
            return;
        }
        const c = confirmState(v, this.sel, this.strikeSel);
        if (!c.ok) {
            this.tip = this.sel.length === 0 ? 'Select a square to mark, or the strike box to take a strike.' : 'Those squares don’t work together with this roll.';
            return;
        }
        await this.guard(async () => {
            const picks = $state.snapshot(this.sel); // plain data for the worker
            const strike = this.strikeSel;
            this.committing = true;
            const step = await this.engine.submitActive(picks, strike);
            const fresh: Record<string, 'draw' | 'ink'> = {};
            step.record.you.forEach((s) => (fresh[`you:${sqKey(s)}`] = 'ink'));
            if (step.record.you_strike) fresh[`you:strike:${step.view.you.strikes - 1}`] = 'ink';
            if (step.record.bot_white) fresh[`bot:${sqKey(step.record.bot_white)}`] = 'ink';
            this.apply(step, fresh);
            if (step.view.phase === 'over') return;
            await this.sleep(PACE.handoff);
            await this.botTurn();
        });
    }

    setBot(k: BotKind) {
        this.nextBot = k;
        this.savePrefs();
        // Mid-sequence the engine snapshot can be ahead of rollNo/log; the
        // next regular save picks up nextBot.
        if (!this.busy) this.save();
    }

    /** Whether the bot rolls first in the next new game. */
    get nextBotFirst(): boolean {
        return this.firstRoll === 'alternate' ? !this.botFirst : this.firstRoll === 'bot';
    }

    /** The menu's First roll row: alternate → you → bot → alternate. */
    cycleFirstRoll() {
        this.firstRoll = NEXT_FIRST_ROLL[this.firstRoll];
        this.savePrefs();
    }

    toggleTheme() {
        this.theme = this.theme === 'dark' ? 'light' : 'dark';
        this.savePrefs();
    }

    // ---- internals ----

    private async botTurn() {
        this.fresh = {};
        this.thinking = false;
        const result = this.engine.botRoll();
        const v = await this.settled(result, this.tumble());
        this.view = v;
        this.rollNo += 1;
        this.save();
    }

    /** Your white-sum mark inks at once; the bot thinks, draws its white sum, then its colour mark. */
    private async resolvePassive() {
        await this.guard(async () => {
            const pick = $state.snapshot(this.sel[0]) ?? null; // plain data for the worker
            this.committing = true;
            this.thinking = true;
            const [step] = await Promise.all([this.engine.submitPassive(pick), this.sleep(PACE.think)]);
            this.thinking = false;
            const r = step.record;
            const fresh: Record<string, 'draw' | 'ink'> = {};
            r.you.forEach((s) => (fresh[`you:${sqKey(s)}`] = 'ink'));
            if (r.bot_white) fresh[`bot:${sqKey(r.bot_white)}`] = 'draw';
            const staged = [...(r.bot_colour ? [`bot:${sqKey(r.bot_colour)}`] : []), ...(r.bot_strike ? ['bot:strike'] : [])];
            this.hidden = staged;
            this.apply(step, fresh);
            if (staged.length) {
                await this.sleep(PACE.stagger);
                this.fresh = r.bot_colour ? { [`bot:${sqKey(r.bot_colour)}`]: 'draw' } : {};
                this.hidden = [];
            }
        });
    }

    private apply(step: Step, fresh: Record<string, 'draw' | 'ink'>) {
        this.view = step.view;
        this.sel = [];
        this.strikeSel = false;
        this.committing = false;
        this.tip = '';
        this.fresh = fresh;
        this.log = [logEntry(this.rollNo, step.record), ...this.log];
        this.save();
        if (step.view.phase === 'over') void this.keepCode();
    }

    /** Spec "Entry points": the finished game's code, kept for the review page. */
    private async keepCode() {
        try {
            const code = await this.engine.gameCode();
            this.lastCode = code;
            this.store.set(LAST_GAME, code);
        } catch {
            // No code: "the last game" is none, not the one before it.
            this.lastCode = null;
            this.store.set(LAST_GAME, '');
        }
    }

    /**
     * Wait for both the engine call and its animation, so a failed call never
     * leaves a tumble or think running past the error handling.
     */
    private async settled<T>(result: Promise<T>, anim: Promise<unknown>): Promise<T> {
        const [r, a] = await Promise.allSettled([result, anim]);
        if (r.status === 'rejected') throw r.reason;
        if (a.status === 'rejected') throw a.reason;
        return r.value;
    }

    /** `keepFaces` leaves the last tumble faces in `shownDice` for the caller to clear. */
    private async tumble(keepFaces = false) {
        this.rolling = true;
        this.shownDice = randomDice();
        const t = setInterval(() => (this.shownDice = randomDice()), PACE.tick);
        try {
            await this.sleep(PACE.tumble);
        } finally {
            clearInterval(t);
            this.rolling = false;
            if (!keepFaces) this.shownDice = null;
        }
    }

    private async guard(work: () => Promise<void>) {
        if (this.busy) return;
        this.busy = true;
        this.error = '';
        try {
            await work();
        } catch (e) {
            this.error = e instanceof Error ? e.message : String(e);
            this.stale = true;
            this.thinking = false;
            this.rolling = false;
            this.pendingDice = null;
            this.shownDice = null;
            this.committing = false;
        } finally {
            this.busy = false;
        }
    }

    private save(rollNo = this.rollNo) {
        if (!this.view) return;
        const meta = { log: this.log, rollNo, bot: this.bot, nextBot: this.nextBot, botFirst: this.botFirst };
        this.engine
            .snapshot()
            .then((snapshot) => this.store.set(SAVE, JSON.stringify({ snapshot, ...meta })))
            .catch(() => {});
    }

    private savePrefs() {
        this.store.set(PREFS, JSON.stringify({ theme: this.theme, nextBot: this.nextBot, firstRoll: this.firstRoll }));
    }

    private read<T>(key: string): T | null {
        try {
            const raw = this.store.get(key);
            return raw ? (JSON.parse(raw) as T) : null;
        } catch {
            return null;
        }
    }
}
