import type { BotKind, Engine, Sq, Step, View } from '../engine/protocol';
import { clickable, confirmState, has, isTerminal, logEntry, NAME, sqKey, strikeOpen, type LogEntry } from './rules';

/** Spec "Pacing": minimum durations in ms. */
export const PACE = { tumble: 600, tick: 75, think: 500, stagger: 450, handoff: 900 };

export interface Store { get(key: string): string | null; set(key: string, value: string): void }

const SAVE = 'qwixxer.save.v1';
const PREFS = 'qwixxer.prefs.v1';

interface Saved { snapshot: string; log: LogEntry[]; rollNo: number; bot: BotKind; nextBot: BotKind }

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

    // Plain fields assigned in the constructor: `.svelte.ts` files do not
    // support TypeScript parameter properties.
    private engine: Engine;
    private store: Store;
    private sleep: (ms: number) => Promise<void>;
    private seed: () => number;

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
        const prefs = this.read<{ theme: 'light' | 'dark'; nextBot: BotKind }>(PREFS);
        if (prefs) {
            this.theme = prefs.theme ?? 'light';
            this.nextBot = prefs.nextBot ?? 'master_plus';
        }
        const saved = this.read<Saved>(SAVE);
        if (saved && (await this.resume(saved))) return;
        return this.newGame();
    }

    /**
     * Restore a saved game. If the engine rejects the snapshot (corrupt or
     * old format), discard the save and report false so a new game starts;
     * otherwise a bad save would break every page load.
     */
    private async resume(saved: Saved): Promise<boolean> {
        let view: View;
        this.busy = true; // block newGame/confirm while the engine restores
        try {
            view = await this.engine.restore(saved.bot, saved.snapshot, this.seed());
        } catch {
            this.store.set(SAVE, '');
            return false;
        } finally {
            this.busy = false;
        }
        await this.guard(async () => {
            this.bot = saved.bot;
            this.nextBot = saved.nextBot;
            this.log = saved.log;
            this.rollNo = saved.rollNo;
            this.view = view;
            if (view.phase === 'bot_to_roll') await this.botTurn();
        });
        return true;
    }

    async newGame() {
        if (this.busy) return;
        await this.guard(async () => {
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
            this.view = await this.engine.start(this.bot, this.seed());
            await this.botTurn();
        });
    }

    /** Your Roll button. */
    async roll() {
        if (this.busy || this.view?.phase !== 'you_to_roll') return;
        await this.guard(async () => {
            this.fresh = {};
            const result = this.engine.youRoll();
            let failed = false;
            result.catch(() => (failed = true));
            const reveal = async () => {
                if (failed) return;
                const rolled = await result.catch(() => null);
                if (!rolled) return;
                this.pendingDice = rolled.dice;
                await this.think();
            };
            const v = await this.settled(result, this.tumble().then(reveal));
            this.thinking = false;
            this.view = v;
            this.pendingDice = null;
            this.rollNo += 1;
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
        if (this.strikeSel) this.tip = 'Clear the strike first to mark a square.';
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
            const picks = this.sel;
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
            const pick = this.sel[0] ?? null;
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

    private async tumble() {
        this.rolling = true;
        this.shownDice = randomDice();
        const t = setInterval(() => (this.shownDice = randomDice()), PACE.tick);
        try {
            await this.sleep(PACE.tumble);
        } finally {
            clearInterval(t);
            this.rolling = false;
            this.shownDice = null;
        }
    }

    private async think() {
        this.thinking = true;
        await this.sleep(PACE.think);
    }

    private async guard(work: () => Promise<void>) {
        if (this.busy) return;
        this.busy = true;
        this.error = '';
        try {
            await work();
        } catch (e) {
            this.error = e instanceof Error ? e.message : String(e);
            this.thinking = false;
            this.rolling = false;
            this.pendingDice = null;
            this.committing = false;
        } finally {
            this.busy = false;
        }
    }

    private save() {
        if (!this.view) return;
        const meta = { log: this.log, rollNo: this.rollNo, bot: this.bot, nextBot: this.nextBot };
        this.engine
            .snapshot()
            .then((snapshot) => this.store.set(SAVE, JSON.stringify({ snapshot, ...meta })))
            .catch(() => {});
    }

    private savePrefs() {
        this.store.set(PREFS, JSON.stringify({ theme: this.theme, nextBot: this.nextBot }));
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
