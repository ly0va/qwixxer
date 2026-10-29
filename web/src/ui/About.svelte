<script lang="ts">
    // The about/FAQ page: a static page of its own (about/index.html), rendered
    // to HTML at build time so search engines index it. No game state here; the
    // wordmark leads back to the app.
    const home = import.meta.env.BASE_URL;
</script>

<div class="about">
    <header class="head">
        <a class="word" href={home}>Qwixxer</a>
    </header>

    <article>
        <h1 class="lede">The strongest Qwixx bot on the internet.</h1>

        <section>
            <h2>What this is</h2>
            <details>
                <summary>What is Qwixxer?</summary>
                <p>A place to play the dice game Qwixx against a computer opponent, analyze your games and explore strategies. Its top bot is the strongest Qwixx bot on the internet that we know of.</p>
            </details>
            <details>
                <summary>Is this the official Qwixx?</summary>
                <p>No. It's an unofficial fan project, not affiliated with or endorsed by Nürnberger-Spielkarten-Verlag or Gamewright. If you enjoy the game, buy the real thing: it's even better at a table with friends.</p>
            </details>
            <details>
                <summary>Can I play with friends online?</summary>
                <p>No. Qwixxer is always you against one bot. However, online play and more opponents might become available in the future.</p>
            </details>
            <details>
                <summary>Is it free? Are there ads or accounts?</summary>
                <p>It's free, with no ads, no sign-up and no tracking. The code is open source: see <a href="https://github.com/ly0va/qwixxer" target="_blank" rel="noopener noreferrer">the repository</a>.</p>
            </details>
            <details>
                <summary>Who made it? How do I report a bug?</summary>
                <p>Qwixxer is made by <a href="https://github.com/ly0va" target="_blank" rel="noopener noreferrer">this guy</a>. Bug reports and ideas are welcome on <a href="https://github.com/ly0va/qwixxer/issues" target="_blank" rel="noopener noreferrer">GitHub</a>.</p>
            </details>
        </section>

        <section>
            <h2>Who it's for</h2>
            <details>
                <summary>I don't know the rules.</summary>
                <p>Read the <a href="https://gamewright.com/pdfs/Rules/QWIXX-01_rules.pdf" target="_blank" rel="noopener noreferrer">official rules</a>, or just start a game against Newbie: you can only make legal moves, so you'll pick it up as you go.</p>
            </details>
            <details>
                <summary>I've never played Qwixx. Is this for me?</summary>
                <p>Yes. Newbie is a gentle opponent, and the explorer shows you which moves are good and why, so you can learn without anyone watching.</p>
            </details>
            <details>
                <summary>I play a lot. What's in it for me?</summary>
                <p>Master+ is a sparring partner that's tougher than most people you'll meet, and a review that shows exactly which decisions cost you. Want to dig deeper? The explorer shows the chance to win, luck and expected marks for every move.</p>
            </details>
            <details>
                <summary>Which bot should I play?</summary>
                <dl class="bots">
                    <dt>Newbie</dt><dd>Plays by a few simple, careful rules. A good first opponent.</dd>
                    <dt>Student</dt><dd>A solid but impatient player that has learned what usually works. Beatable once you know the game.</dd>
                    <dt>Master</dt><dd>Very strong. It judges every move by how likely it is to win the game.</dd>
                    <dt>Master+</dt><dd>Master, but it also looks a few rolls ahead. The strongest, and it takes a moment longer to move.</dd>
                </dl>
            </details>
        </section>

        <section>
            <h2>What you can do</h2>
            <details>
                <summary>What is the explorer?</summary>
                <p>A sandbox. Set up any score sheets and dice, and see every sensible move ranked by its chance to win, with the marks and strikes each one should lead to and how lucky the roll was.</p>
            </details>
            <details>
                <summary>What does the game review show?</summary>
                <p>After a game, press Analyze. You'll see a graph of your chance to win over the whole game, and every roll with the move you played next to the best one. A move that was as good as the best is marked Best or Excellent; otherwise it shows how many points of winning chance it cost you.</p>
            </details>
            <details>
                <summary>Does it work on a phone?</summary>
                <p>Yes. It has its own phone layout, and you can add it to your home screen to open it like an app.</p>
            </details>
            <details>
                <summary>Can I use it offline?</summary>
                <p>Yes. Open Qwixxer once with a connection and it keeps working without one: the bots run locally on your device, so there's nothing to download mid-game.</p>
            </details>
            <details>
                <summary>Which rules does it use? Are variants supported?</summary>
                <p>The standard rules of the original Qwixx, for two players: the usual scoring, a row locks after at least five marks, and the game ends when two rows are locked or someone takes a fourth strike. Variants such as Qwixx Mixx, Qwixx Big Points or house rules aren't supported.</p>
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
                <p>The ranking is a quick estimate. Master+ may have played the roll out many times and found another move better.</p>
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
                <summary>Why don't turns have separate phases?</summary>
                <p>At the table, everyone marks the white sum simultaneously, then the roller marks a white+colour sum. Against a single bot, two separate steps per roll would only mean more clicks, so on your roll you pick up to two boxes (or the strike box) and confirm once. Qwixxer works out which mark is the white sum and which is the colour; the order never changes the result.</p>
                <p>On the bot's roll you mark the white sum (or pass) first, then the bot plays the rest of its turn, so a row you lock is already closed to it on the second phase. Even though you mark first, the bot does not see your move until the second phase begins.</p>
            </details>
        </section>

        <section>
            <h2>How it works</h2>
            <details>
                <summary>How do the bots think?</summary>
                <p>Newbie plays by a few simple, hand-written rules. Student and Master both learned by playing against themselves, but Student has just 5 parameters to tune, while Master has about 30,000 and played millions of games. For each possible move Master estimates how likely it is to win from the position that follows, and plays the best one. Master+ also looks a few rolls ahead before deciding. Both are much stronger than a casual player, but not perfect.</p>
            </details>
            <details>
                <summary>What do the percentages and "−3.2" mean?</summary>
                <p>Percentages are your chance to win the game from that point. A signed number like −3.2 is a change in that chance, in percentage points: a suboptimal move that costs 3.2 points took you from, say, 55% to 51.8%.</p>
            </details>
            <details>
                <summary>How is luck calculated?</summary>
                <p>Each roll is compared with every roll that could have come up instead. For each of the 46,656 ways the six dice can land, Qwixxer works out your chance to win if whoever rolled makes their best move with them. A roll's luck is how far its own result is from the average of all of them: +4 means this roll left you 4 points of winning chance better off than a typical roll would have. The bot's good luck is your bad luck, and the other way round.</p>
                <p>In the review, each roll also shows how it ranks ("luckier than 80% of rolls"), and the Dice luck tile adds up the whole game. Its z score divides that total by how much the dice could have swung it: around 0 is an ordinary game, beyond ±2 is rare.</p>
            </details>
            <details>
                <summary>Are the dice fair? Does the bot cheat?</summary>
                <p>The dice are fair: each die is rolled at random, with the same odds on your rolls and the bot's. The bot doesn't know future rolls, and it sees only what you see on the sheets. On its own roll it doesn't see your white-sum mark before it decides.</p>
            </details>
        </section>
    </article>
</div>

<style>
    .about { max-width: 720px; margin: 0 auto; box-sizing: border-box; padding: 12px 16px 64px; }
    .head { display: flex; align-items: center; justify-content: space-between; height: 44px; }
    .word { font-family: var(--cond); font-weight: 800; font-size: 30px; letter-spacing: 0.5px; color: var(--ink); text-decoration: none; }
    article { margin-top: 28px; line-height: 1.55; font-size: 16px; }
    .lede { font-size: 18px; font-weight: 400; color: var(--soft); margin: 0 0 8px; }
    section { margin-top: 28px; padding: 6px 20px 14px; border-radius: 14px; background: var(--paper); box-shadow: 0 1px 0 var(--edge); }
    h2 { font-family: var(--cond); font-weight: 800; font-size: 26px; margin: 14px 0 6px; }
    details { border-top: 1px solid var(--log-bg); }
    h2 + details { border-top: none; }
    summary { display: flex; align-items: center; justify-content: space-between; gap: 12px; min-height: 48px; padding: 6px 0; font-size: 17px; font-weight: 700; cursor: pointer; list-style: none; }
    summary::-webkit-details-marker { display: none; }
    summary::after { content: ''; flex: none; width: 8px; height: 8px; margin-right: 4px; border-right: 2px solid var(--muted); border-bottom: 2px solid var(--muted); transform: translateY(-2px) rotate(45deg); transition: transform 0.15s; }
    details[open] > summary::after { transform: translateY(2px) rotate(-135deg); }
    details > :last-child { margin-bottom: 14px; }
    p { margin: 0 0 8px; color: var(--soft); }
    article a { color: var(--blue-ink); }
    .bots { display: grid; grid-template-columns: max-content 1fr; gap: 6px 16px; margin: 8px 0 8px; }
    .bots dt { font-weight: 700; }
    .bots dd { margin: 0; color: var(--soft); }
    @media (max-width: 599px) {
        .word { font-size: 26px; }
        section { padding: 4px 16px 12px; }
    }
</style>
