# Template: the generated style skill (Phase A, the Package step)

Write this to `~/.claude/skills/style-<slug>/SKILL.md`, filled from the board, the breakdown and
the library. Keep every rule numbered and measurable; a plan or a revision note must be able to
say "rule 7" and mean one thing. Replace every <…>.

```markdown
---
name: style-<slug>
description: The <Style Name> editing style (Hebrew name: <name as the user calls it>; registry slug <slug>) for <format, e.g. 9:16 talking-head reels>, trained on <N> reference videos of <creator/source>. <One sentence on the look: grounds, type, accent, motion.> Use whenever the user wants to edit, plan, caption or design a clip "in the <Style Name> style", "like <creator>", or names this style. Editing runs through the style-lab skill (phase B); this skill holds the style's rules, tokens and reference library.
---

# <Style Name>

Registry: `<slug>` in ~/.claude/style-lab/styles.json · fonts: <brand|reference> · colours: <brand|reference>

Trained <date> from <N> reference videos (<creator>, list in <pack>/refs/sources.md) and the style
board <board artifact URL>. Measured: <pack>/fingerprint.json (per-layer numbers with spreads),
<pack>/laws.json (when X then Y), <pack>/creator-mind.json (how the creator turns a line into a
picture), <pack>/recipes.json (how each element is made again). Proof: <pack>/proof.json (hold-out
F1 <x>, timing error <y> s; recreation moments <ids>). To edit a clip in this style, follow
style-lab phase B (B1–B6) with this skill loaded, and read those files at B2.

## What makes it work
<3–5 sentences: the feeling, the discipline, what is never done.>

## Tokens
| Token | Value | Notes |
|---|---|---|
| Main font | <brand default: Heebo, weights> | <reference's original, e.g. Inter> |
| Special font | <brand default: Karantina 700> | <reference's original; when used> |
| Mono font | <brand default: Consolas / Courier New> | <when used> |
| Colours | <brand tokens from style-lab references/brand-defaults.md> | <reference's original hex; where each is allowed> |
| Grounds / textures | <…> | <strength, e.g. paper grain 12 % opacity> |
| Line work / shapes | <…> | <thickness px at 1080×1920> |
| Sizing | <ratios> | <e.g. bold word 2.5× light line> |
| Layout / safe zones | <…> | <caption band y-range, margins> |

## Rules (numbered)
1. <Caption rule …> Why: <the principle: metaphor, reveal, tension, contrast, rhythm>.
2. …
(Every rule: what, measured value with its spread (median, p10-p90 from fingerprint.json), when it
applies, and why it works. A rule from laws.json names the law (`when` → `then`, lag, support).
Cite the board callout or the law it came from.)

## Categories
### <n>. <Category name>  (<scene type: a-roll / overlay / cutaway>)
- Design: <ground, texture, type pair, colour, element style, position>
- Motion: <in/out durations in frames at <fps>, easing, camera moves>
- Placement: <which kinds of script lines get this scene; how long after the trigger word>
- Refs: <V##-A## — one-line description> (2–3)

## Planning patterns (from the references)
- <e.g. Announce a new idea on the face; describe it with a visual.>
- <Opening: …> <Ending: …> <Rhythm: a cut-away every ~N s; never two of the same category back to back …>

## Way of thinking (how a beat is designed in this style)
<3-6 short principles, e.g. "the visual is a metaphor for the point, not the noun";
"show the real tool working"; "one idea per beat, never repeated". See style-lab
references/craft-rules.md for the general ones; write only what is specific here.>

## Look (from look.json)
- Grain <std, size px>; vignette <falloff>; tone <black/white points, contrast>; split toning
  <shadow and highlight tint>; halation <yes/no>; text finish <stroke px, shadow, glow>.

## Sound profile
- Voice: <LUFS, words per second, pause median, spectral tilt, crest dB (compression), room rt60>
- Music: <feel, level under voice in LUFS or dB, BPM, ducking dB, sections per minute>
- Effects: <text appear → click; line/graph draw → marker; transition → whoosh; …>, density ≈ <n per minute>

## Review-round rules (added after each round)
<!-- style-lab:learned-rules:start <slug> -->
<!-- style-lab:learned-rules:end <slug> -->

## Known gaps
<what the references did not show enough of; what to ask the user>
```

**Review-round rules.** Keep that heading and the marker pair exactly as above, with `<slug>`
replaced by the style's slug, empty when the style is packaged, in both STYLE.md and the style skill. `lessons.py
approve` writes the style's approved rules between its markers (numbered, measured, with
evidence count, date and lesson id; cite them by lesson id) and keeps everything outside them
as written, other styles' blocks included (`references/learning-loop.md` §6). Never type rules
between the markers by hand; rules there apply to every video in this style.

After writing it, list it to the user with its name and tell them a new session will pick it up
(skills load at session start).
