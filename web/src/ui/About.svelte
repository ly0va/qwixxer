<script lang="ts">
    import type { GameController } from '../game/controller.svelte';
    import Menu from './Menu.svelte';
    import type { Nav } from './nav';
    import Tabs from './Tabs.svelte';
    // The about/FAQ page (draft): one scrolling column on every layout.
    // Desktop keeps the Play / Explore / Review tabs; the phone reaches them from the ⋯ menu.
    let { c, nav, phone }: { c: GameController; nav: Nav; phone: boolean } = $props();
</script>

<div class="about" class:phone>
    <header class="head">
        <div class="left">
            <div class="word">Qwixxer</div>
            {#if !phone}<Tabs current="about" onPlay={nav.play} onExplore={nav.explorer} onReview={nav.review} />{/if}
        </div>
        <Menu {c} {nav} tabs={!phone} />
    </header>

    <article>
        <h1>About Qwixxer</h1>
        <p class="lede">Play Qwixx against strong bots, then find out where your games were won and lost.</p>

        <section>
            <h2>What this is</h2>
            <details>
                <summary>What is Qwixxer?</summary>
                <p>A place to play the dice game Qwixx against a computer opponent, and to look closely at how you play. You can pick an opponent from beginner to very strong, test any position in the explorer, and replay a finished game move by move in the review.</p>
            </details>
            <details>
                <summary>Is this the official Qwixx?</summary>
                <p>No. It's an unofficial fan project, not affiliated with or endorsed by Nürnberger-Spielkarten-Verlag or Gamewright. If you enjoy the game, buy the real thing: it's even better at a table with friends.</p>
            </details>
            <details>
                <summary>Can I play with friends online?</summary>
                <p>No. Qwixxer is always you against one bot. There are no rooms, accounts or online play.</p>
            </details>
            <details>
                <summary>Is it free? Are there ads or accounts?</summary>
                <p>It's free, with no ads, no sign-up and no tracking. The code is open source under the AGPL; see <a href="https://github.com/ly0va/qwixxer" target="_blank" rel="noopener noreferrer">the repository</a>.</p>
            </details>
            <details>
                <summary>I don't know the rules.</summary>
                <p>Read the <a href="https://gamewright.com/pdfs/Rules/QWIXX-01_rules.pdf" target="_blank" rel="noopener noreferrer">official rules</a> (two pages), or just start a game against Newbie: you can only make legal moves, so you'll pick it up as you go.</p>
            </details>
            <details>
                <summary>Who made it? How do I report a bug?</summary>
                <p>Qwixxer is made by Lev Potomkin. Bug reports and ideas are welcome on <a href="https://github.com/ly0va/qwixxer/issues" target="_blank" rel="noopener noreferrer">GitHub</a>.</p>
            </details>
        </section>

        <section>
            <h2>Who it's for</h2>
            <details>
                <summary>I've never played Qwixx. Is this for me?</summary>
                <p>Yes. Newbie is a gentle opponent, and the explorer shows you which moves are good and why, so you can learn without anyone watching.</p>
            </details>
            <details>
                <summary>I play a lot. What's in it for me?</summary>
                <p>A sparring partner that's tougher than most people you'll meet, and a review that shows exactly which decisions cost you. Want to dig deeper? The explorer shows the chance to win, luck and expected marks for every move.</p>
            </details>
        </section>

        <section>
            <h2>What you can do</h2>
            <details>
                <summary>Which bot should I play?</summary>
                <dl class="bots">
                    <dt>Newbie</dt><dd>Plays by a few simple, careful rules. A good first opponent.</dd>
                    <dt>Student</dt><dd>A solid player that has learned what usually works. Beatable once you know the game.</dd>
                    <dt>Master</dt><dd>Very strong. It judges every move by how likely it is to win the game.</dd>
                    <dt>Master+</dt><dd>Master, but it also looks a few rolls ahead. The strongest, and it takes a moment longer to move.</dd>
                </dl>
            </details>
            <details>
                <summary>What is the explorer?</summary>
                <p>A sandbox. Set up any score sheets and dice, and see every sensible move ranked by its chance to win, with the marks and strikes each one should lead to and how lucky the roll was.</p>
            </details>
            <details>
                <summary>What does the game review show?</summary>
                <p>After a game, press Analyze. You'll see a graph of your chance to win over the whole game, and every roll with the move you played next to the best one. A move that was as good as the best (or nearly) is marked Best or Excellent; otherwise it shows how many points of winning chance it cost you.</p>
            </details>
            <details>
                <summary>Did I lose because of luck or because of my moves?</summary>
                <p>The review answers both. The Dice luck tile adds up how much the dice helped or hurt each side, and each roll shows how it compared with every other roll that could have come up ("luckier than 80% of rolls"). The Your mistakes tile adds up what your moves cost. Compare the two.</p>
            </details>
            <details>
                <summary>What do the percentages and "+3.2" mean?</summary>
                <p>Percentages are your chance to win the game from that point. A signed number like +3.2 is a change in that chance, in percentage points: a move that costs 3.2 points took you from, say, 55% to 51.8%.</p>
            </details>
        </section>

        <section>
            <h2>Limitations and quirks</h2>
            <details>
                <summary>Is Master perfect?</summary>
                <p>No. It's very strong but it can still misjudge positions, especially close endgames. Treat its verdicts as a strong player's opinion, not the truth.</p>
            </details>
            <details>
                <summary>Isn't luck judged by the same bot that judges my moves?</summary>
                <p>Yes. Both luck and mistakes are measured with Master's sense of who's winning, so if Master misreads a position, both numbers inherit that. Over a whole game the errors mostly wash out, but don't argue over a single roll.</p>
            </details>
            <details>
                <summary>Why does Master+ sometimes play a move that isn't ranked first?</summary>
                <p>The ranking is a quick estimate. Master+ may have played the roll out many times and found another move better, or a fixed rule (like locking a row when it's safe) may decide. The move list explains which.</p>
            </details>
            <details>
                <summary>Why is some analysis approximate?</summary>
                <p>To stay fast on a phone, the review checks only the most promising moves of each roll in depth, and the explorer stops after a fixed number of simulated games. The results are close, not exact.</p>
            </details>
            <details>
                <summary>Where are my games saved?</summary>
                <p>Only in your browser: the game in progress, your last finished game and your settings. Clearing your browser's site data erases them. To keep a game, copy its link.</p>
            </details>
            <details>
                <summary>Does it work on a phone, or offline?</summary>
                <p>Yes to both. It has its own phone layout, and once loaded it keeps working without a connection. You can also add it to your home screen.</p>
            </details>
        </section>

        <section>
            <h2>How it works</h2>
            <details>
                <summary>How do the bots think?</summary>
                <p>Master learned by playing millions of games against itself. For each possible move it estimates how likely it is to win from the position that follows, and plays the best one. Master+ also looks a few rolls ahead before deciding. It's much stronger than a casual player, but not perfect.</p>
                <p class="todo">[Link to the technical write-up goes here.]</p>
            </details>
            <details>
                <summary>Does my game go to a server?</summary>
                <p>No. The bots run entirely in your browser, so nothing you play leaves your device unless you share a link.</p>
            </details>
        </section>
    </article>
</div>

<style>
    .about { max-width: 720px; margin: 0 auto; box-sizing: border-box; padding: 12px 16px 64px; }
    .head { display: flex; align-items: center; justify-content: space-between; height: 44px; }
    .left { display: flex; align-items: center; gap: 22px; }
    .word { font-family: var(--cond); font-weight: 800; font-size: 30px; letter-spacing: 0.5px; }
    .phone .word { font-size: 26px; }
    article { margin-top: 28px; line-height: 1.55; font-size: 16px; }
    h1 { font-family: var(--cond); font-weight: 800; font-size: 40px; line-height: 1.1; margin: 0 0 8px; }
    .lede { font-size: 18px; color: var(--soft); margin: 0 0 8px; }
    section { margin-top: 28px; padding: 6px 20px 14px; border-radius: 14px; background: var(--paper); box-shadow: 0 1px 0 var(--edge); }
    .phone section { padding: 4px 16px 12px; }
    h2 { font-family: var(--cond); font-weight: 800; font-size: 26px; margin: 14px 0 6px; }
    details { border-top: 1px solid var(--log-bg); }
    h2 + details { border-top: none; }
    summary { display: flex; align-items: center; justify-content: space-between; gap: 12px; min-height: 48px; padding: 6px 0; font-size: 17px; font-weight: 700; cursor: pointer; list-style: none; }
    summary::-webkit-details-marker { display: none; }
    summary::after { content: ''; flex: none; width: 8px; height: 8px; margin-right: 4px; border-right: 2px solid var(--muted); border-bottom: 2px solid var(--muted); transform: translateY(-2px) rotate(45deg); transition: transform 0.15s; }
    details[open] > summary::after { transform: translateY(2px) rotate(-135deg); }
    details > :last-child { margin-bottom: 14px; }
    p { margin: 0 0 8px; color: var(--soft); }
    a { color: var(--blue-ink); }
    .bots { display: grid; grid-template-columns: max-content 1fr; gap: 6px 16px; margin: 8px 0 8px; }
    .bots dt { font-weight: 700; }
    .bots dd { margin: 0; color: var(--soft); }
    .todo { font-style: italic; color: var(--faint); }
</style>
