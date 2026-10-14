<script lang="ts">
    import { createEngine } from '../engine/client';
    const engine = createEngine();
    let text = $state('loading engine…');
    engine
        .start('student', 1)
        .then((v) => engine.botRoll().then((b) => (text = `phase ${v.phase} → ${b.phase}, dice ${b.dice}`)))
        .catch((e) => (text = `error: ${e.message}`));
</script>

<p>{text}</p>
