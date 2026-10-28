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
            <h3>What is Qwixxer?</h3>
            <p>A place to play the dice game Qwixx against a computer opponent, and to look closely at how you play. You can pick an opponent from beginner to very strong, test any position in the explorer, and replay a finished game move by move in the review.</p>
            <h3>Is this the official Qwixx?</h3>
            <p>No. It's an unofficial fan project, not affiliated with or endorsed by Nürnberger-Spielkarten-Verlag or Gamewright. If you enjoy the game, buy the real thing: it's even better at a table with friends.</p>
            <h3>Can I play with friends online?</h3>
            <p>No. Qwixxer is always you against one bot. There are no rooms, accounts or online play.</p>
            <h3>Is it free? Are there ads or accounts?</h3>
            <p>It's free, with no ads, no sign-up and no tracking. The code is open source under the AGPL; see <a href="https://github.com/ly0va/qwixxer" target="_blank" rel="noopener noreferrer">the repository</a>.</p>
            <h3>I don't know the rules.</h3>
            <p>Read the <a href="https://gamewright.com/pdfs/Rules/QWIXX-01_rules.pdf" target="_blank" rel="noopener noreferrer">official rules</a> (two pages), or just start a game against Newbie: you can only make legal moves, so you'll pick it up as you go.</p>
        </section>

        <section>
            <h2>Who it's for</h2>
            <h3>I've never played Qwixx. Is this for me?</h3>
            <p>Yes. Newbie is a gentle opponent, and the explorer shows you which moves are good and why, so you can learn without anyone watching.</p>
            <h3>I play a lot. What's in it for me?</h3>
            <p>A sparring partner that's tougher than most people you'll meet, and a review that shows exactly which decisions cost you. Want to dig deeper? The explorer shows the chance to win, luck and expected marks for every move.</p>
        </section>

        <section>
            <h2>What you can do</h2>
            <h3>Which bot should I play?</h3>
            <dl class="bots">
                <dt>Newbie</dt><dd>Plays by a few simple, careful rules. A good first opponent.</dd>
                <dt>Student</dt><dd>A solid player that has learned what usually works. Beatable once you know the game.</dd>
                <dt>Master</dt><dd>Very strong. It judges every move by how likely it is to win the game.</dd>
                <dt>Master+</dt><dd>Master, but it also looks a few rolls ahead. The strongest, and it takes a moment longer to move.</dd>
            </dl>
            <h3>Who rolls first?</h3>
            <p>By default you take turns: whoever didn't roll first last game rolls first this time. You can change this under First roll in the ⋯ menu; it applies from your next game.</p>
            <h3>What is the explorer?</h3>
            <p>A sandbox. Set up any score sheets and dice, and see every sensible move ranked by its chance to win, with the marks and strikes each one should lead to and how lucky the roll was.</p>
            <h3>What does the game review show?</h3>
            <p>After a game, press Analyze. You'll see a graph of your chance to win over the whole game, and every roll with the move you played next to the best one. A move that was as good as the best (or nearly) is marked Best or Excellent; otherwise it shows how many points of winning chance it cost you.</p>
            <h3>Did I lose because of luck or because of my moves?</h3>
            <p>The review answers both. The Dice luck tile adds up how much the dice helped or hurt each side, and each roll shows how it compared with every other roll that could have come up ("luckier than 80% of rolls"). The Your mistakes tile adds up what your moves cost. Compare the two.</p>
            <h3>What do the percentages and "+3.2" mean?</h3>
            <p>Percentages are your chance to win the game from that point. A signed number like +3.2 is a change in that chance, in percentage points: a move that costs 3.2 points took you from, say, 55% to 51.8%.</p>
            <h3>How do I share a game or a position?</h3>
            <p>Use the link button on the review page, at the end of a game, or in the explorer. The link holds the whole game (every roll and every choice) so anyone who opens it sees exactly what you saw.</p>
        </section>

        <section>
            <h2>Limitations and quirks</h2>
            <h3>Is Master perfect?</h3>
            <p>No. It's very strong but it can still misjudge positions, especially close endgames. Treat its verdicts as a strong player's opinion, not the truth.</p>
            <h3>Isn't luck judged by the same bot that judges my moves?</h3>
            <p>Yes. Both luck and mistakes are measured with Master's sense of who's winning, so if Master misreads a position, both numbers inherit that. Over a whole game the errors mostly wash out, but don't argue over a single roll.</p>
            <h3>Why does Master+ sometimes play a move that isn't ranked first?</h3>
            <p>The ranking is a quick estimate. Master+ may have played the roll out many times and found another move better, or a fixed rule (like locking a row when it's safe) may decide. The move list explains which.</p>
            <h3>Why is some analysis approximate?</h3>
            <p>To stay fast on a phone, the review checks only the most promising moves of each roll in depth, and the explorer stops after a fixed number of simulated games. The results are close, not exact.</p>
            <h3>Where are my games saved?</h3>
            <p>Only in your browser: the game in progress, your last finished game and your settings. Clearing your browser's site data erases them. To keep a game, copy its link.</p>
            <h3>Does it work on a phone, or offline?</h3>
            <p>Yes to both. It has its own phone layout, and once loaded it keeps working without a connection. You can also add it to your home screen.</p>
        </section>

        <section>
            <h2>How it works</h2>
            <h3>How do the bots think?</h3>
            <p>Master learned by playing millions of games against itself. For each possible move it estimates how likely it is to win from the position that follows, and plays the best one. Master+ also looks a few rolls ahead before deciding. It's much stronger than a casual player, but not perfect.</p>
            <p class="todo">[Link to the technical write-up goes here.]</p>
            <h3>Does my game go to a server?</h3>
            <p>No. The bots run entirely in your browser, so nothing you play leaves your device unless you share a link.</p>
            <h3>Who made it? How do I report a bug?</h3>
            <p>Qwixxer is made by Lev Potomkin. Bug reports and ideas are welcome on <a href="https://github.com/ly0va/qwixxer/issues" target="_blank" rel="noopener noreferrer">GitHub</a>.</p>
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
    h2 { font-family: var(--cond); font-weight: 800; font-size: 26px; margin: 14px 0 4px; }
    h3 { font-size: 17px; font-weight: 700; margin: 20px 0 4px; }
    p { margin: 0 0 8px; color: var(--soft); }
    a { color: var(--blue-ink); }
    .bots { display: grid; grid-template-columns: max-content 1fr; gap: 6px 16px; margin: 8px 0 8px; }
    .bots dt { font-weight: 700; }
    .bots dd { margin: 0; color: var(--soft); }
    .todo { font-style: italic; color: var(--faint); }
</style>
