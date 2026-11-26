# Reading a video as its creator (creator.json, creator-mind.json)

The measurements say WHAT the creator did: a cut at 4.2 s, a 1.3x push-in, a hit at -18 dB. This guide
is about WHY. Above all it is about how the creator came up with each visual idea and what they wanted
the viewer to get from it. With that, a new video gets new visuals built in the creator's way of thinking,
not copies of their moves.

You write one `analysis/<id>/creator.json` per training video. Do it after you have read the keyframe
sheets, the transcript and `measure.json`, `motion.json`, `sound.json` and the look. Then
`creator.py check <pack>` validates the files, and `creator.py synth <pack>` builds `creator-mind.json`.
Every visual beat of a new plan then runs through `creator.py ideate` and the checklist.

The five design questions of `craft-rules.md` §1 are the backbone. Every visual you read here should
answer them: the point, not the noun (1); the real thing working (2); the invisible made visible (3); a
real reference for the move (4); something new (5). Reading a creator means finding how *they* answer
those five questions.

Order of work for one video:
1. Visual ideas (§1). This is the main part and takes most of the time.
2. Story and hook (§2).
3. Decisions and the problems they solve (§3).
4. Retention devices (§4).
5. Toolchain, from evidence only (§5).
6. Rules of thumb, with habit kept apart from one-off (§6).

---

## 1. Reading the visual ideas (the centre)

Write one `visual_ideas` entry for every visual beat: every time the picture stops being "the person
talking" and starts *saying* something. That covers a B-roll, a prop, a graphic, a text slam, a change
of framing that carries meaning, and a cutaway. For each one, ask these questions in this order:

1. **What line does it sit on?** Copy the spoken words under the visual into `line`, and set `t0`/`t1`
   to the visual's own span (from the sheets and `measure.json` cuts), not the sentence's span.
2. **What did the creator want the viewer to get?** Write it in `message`, as an understanding or a
   feeling, not as a noun: "this is much cheaper than they think", not "a price". Put the feeling in
   `emotion` (surprise, relief, unease, pride, curiosity, ...).
   Then tag `message_type` with one of these:
   - `fact`: a plain piece of information;
   - `number`: a quantity, price, time or ratio;
   - `claim`: an opinion or a promise the viewer must believe;
   - `emotion`: a feeling to share;
   - `contrast`: this versus that;
   - `process`: steps or how something works;
   - `reveal`: hidden, then shown;
   - `cta`: what to do next.
3. **What is the idea, concretely?** In `idea`, describe what is on screen so that someone could rebuild
   it without seeing the video: the object, the action, the framing, the text. "A paper receipt crumples
   into a ball and drops into a bin, top-down, on white", not "a nice metaphor about money".
4. **How did they translate the message into a picture?** Choose one `translation`:

   | method | what it does | generic example |
   |---|---|---|
   | `literal` | shows the thing itself | "open the app": the app opening |
   | `metaphor` | shows something else that works the same way | "it unlocks the next level": a key turning in a door |
   | `symbol` | a sign the viewer already reads | "approved": a green tick stamped on the frame |
   | `contrast` | two opposite states side by side | "loud vs calm": a shaking cup next to a still one |
   | `exaggeration` | pushes the real thing past real life | "a mountain of emails": inbox counter spins to 99,999 |
   | `visual_pun` | the words taken literally | "break the ice": a hammer cracks an ice cube with a word inside |
   | `data_viz` | the number drawn as a chart, bar or counter | "40 % more": a bar fills to 40 % |
   | `pov` | the viewer's own eyes in the situation | "your first day": first-person walk through a door |
   | `before_after` | the same thing in two moments | "it fixes the sound": waveform jagged, then clean |
   | `juxtaposition` | two unrelated images that make a third meaning | "time vs money": an hourglass on a pile of coins |
   | `personification` | a thing gets a face, a voice or intent | "the bug hides": a cartoon bug ducks behind a line of code |
   | `scale_shift` | size carries the meaning | "ten times bigger": a coin grows taller than a house |
   | `reveal` | the answer is hidden, then shown | "guess the price": a card flips to show the number |
   | `other` | anything else (name it in `idea`) | |

5. **Why this idea, and not a literal shot?** In `why_this`, say what the figure does that a literal
   shot could not: "size makes the number felt", "the viewer sees the gap before they hear it", "a
   familiar object makes an abstract tool concrete". When the creator chose `literal`, say why literal
   was right here; often it is proof (§3).
6. **What else could they have done?** Write two `alternatives`: real, buildable ideas that use other
   methods. They show you understood the space of choices, and they seed the ideas bank.
7. **How does the craft support the idea?** Fill `craft` with:
   - `framing`: where the idea sits and how big it is;
   - `motion`: how it enters, moves and leaves, and why that motion fits;
   - `text_role`: none, a label, the keyword, or the whole message;
   - `sound_role`: none, a hit on the reveal, a riser, or silence for weight.

   Read these from the measurements, not from memory.
8. **Optional `pattern`:** a short, reusable name for the idea, such as "everyday object at giant
   scale", "split screen old vs new", or "counter spins past the real number". Synth counts patterns
   across videos. Without one, the idea text itself is counted, and it rarely repeats.

### Three worked examples (invented content)

**A. A number.** Line: "it renders three times faster". The visual is a stopwatch on a desk. On "three
times" it shrinks to a third of its size while a second stopwatch beside it keeps running.
```json
{"t0": 12.4, "t1": 14.6, "line": "it renders three times faster",
 "message": "the wait you hate is now a third as long", "message_type": "number", "emotion": "relief",
 "idea": "two stopwatches side by side on a desk; on 'three times' the right one shrinks to a third and stops while the left one keeps running",
 "translation": "scale_shift",
 "why_this": "the ratio is seen as size before it is heard; a speed chart would need reading",
 "alternatives": ["data_viz: two progress bars racing, one done at a third of the way", "exaggeration: a coffee cup that never gets a sip before the render is done"],
 "craft": {"framing": "objects centred, speaker cut away", "motion": "shrink eased in over 10 frames, small settle", "text_role": "'3x' slams on the shrink", "sound_role": "a tick stops dead on the shrink"},
 "pattern": "object resized to the ratio"}
```

**B. A claim.** Line: "most people do this step wrong". The visual is a top-down shot of ten identical
paper cups. Nine tip over at once, and one stays standing in the brand colour.
- `message`: "you are probably in the nine, and you want to be the one".
- `message_type`: `claim`; `translation`: `symbol` (the one upright cup stands for "the right way").
- `why_this`: the viewer counts themselves into the group without being told; a talking head would only
  assert it.
- `alternatives`:
  - a `data_viz` of a 90 % pie;
  - a `pov` of the viewer's own hand making the mistake.
- `craft`:
  - framing: a top-down grid;
  - motion: the nine fall with staggered physics, not in sync;
  - text_role: none, the image carries it;
  - sound_role: one clatter, then silence.

**C. A contrast that the creator kept literal.** Line: "here is the file before, and after". The visual
is a real screen recording of the same file. The left half is noisy and the right half is clean, with
a wipe on "after".
- `translation`: `before_after`.
- `why_this`: this is proof; any metaphor would weaken the claim, so the creator shows the real result
  (craft-rules §1 question 2).
- `alternatives`:
  - a `metaphor` of a muddy window wiped clean;
  - a `data_viz` of a noise meter dropping.
- Note it in the decisions too: the creator chose *not* to use a metaphor (§3).

### Reading the visual ideas well
- Read the line and the picture together. Keep asking what changed in the viewer's head at that moment.
- One beat can carry two methods (a `scale_shift` that is also a `reveal`). Pick the one that carries
  the message, and name the other one in `why_this`.
- Do not flatter the creator. If a visual is decoration with no message, write that in `message`
  ("none, it fills a gap"), choose `other`, and say so in `why_this`. Synth must not learn decoration
  as an idea.
- Do not write the style's name, the creator's name or a brand into the free text. Describe the move.

---

## 2. Story and hook

`story.structure` divides the video into parts. Every second belongs to one part, and the parts are in
time order:

| part | its job |
|---|---|
| `hook` | stop the scroll |
| `setup` | who, what, why this matters |
| `context` | background the payoff needs |
| `turn` | the twist or the "but" |
| `escalation` | raises the stakes |
| `payoff` | the answer or the result |
| `recap` | the point again, short |
| `cta` | the next action |
| `other` | none of the above |

`words` holds the first few words of the part.

`story.hook`:
- `type`: paradox, question, claim, number, visual, story or other.
- `first_words_s`: when the first word is spoken. A visual hook often starts before any word.
- `visual_device`: what the picture does in the first second.

`open_loops`: every question the video raises and answers later ("I'll show you the trick at the end").
Record `opened_t`, `closed_t` (null when it is never closed) and `text`.

`audience_promise`: what the viewer is promised for staying, in one sentence.

## 3. Decisions: why each change, and what it solves

Write a `decisions` entry for every notable change: a cut, a zoom, a layout switch, music in or out, a
graphic, or a held silence. The `what` is visible in the measurements. The work is the `why`. Watch the
second before the change and ask what was going wrong, or about to go wrong.

Name the problem it solves in `problem_solved`:
- **`attention_dip`**: a long static stretch, a slow sentence, or a repeated framing. The change
  re-grabs the eye. Look for long shot lengths just before it in `measure.json`.
- **`clarity`**: the words alone would be hard to follow. The change shows the thing, labels it, or
  isolates it, for example a zoom on the button being named.
- **`proof`**: a claim needs to be believed. The change shows real evidence, such as a real screen,
  a real result, or a number on screen.
- **`emotion`**: the feeling must land. The change adds weight, such as a slow push, silence, a close-up,
  or music dropping out.
- **`rhythm`**: the cut lands on the beat or the breath so the edit feels musical. Check it against the
  BPM and beats in `sound.json`.
- **`other`**: say which.

Then name the `principle`, the general reason that would apply to any video. Examples:
- "cut to a new framing before a shot passes 4 s"
- "prove a claim with the real thing, never a metaphor"
- "silence before the key line gives it weight"

Write principles so they can be counted: short and general. Synth ranks principles by how many
decisions they explain.

**What the creator chose NOT to do** goes in `alternative_rejected`: the obvious move they skipped, and
why skipping it was better. For example: "no zoom on the punchline, because a zoom would undercut the
deadpan" or "no music under the confession, so it feels unproduced". What a creator refuses to do is as
much a part of the style as what they do. Leave out what is not there for a reason; record only an
absence you can explain.

`evidence` names where you saw it: measure, motion, sound, look, transcript or sheet, alone or in a
list.

## 4. Retention devices

Write a `retention` entry for every moment built to keep the viewer watching. The devices are:
- `pattern_interrupt`: a sudden change of framing, colour or layout;
- `reveal`;
- `open_loop`: opening a question;
- `payoff`: closing one;
- `tempo_change`;
- `sound_hit`;
- `text_slam`;
- `other`.

`gap_s_since_last` is the time since the previous device, or since 0 for the first. Synth reads the
largest gap per video and turns its p90 into "the creator never lets more than N s pass without a
device". The planner checks every new beat against that number.

## 5. Toolchain: inferred from evidence, never guessed

For each kind of element (broll, title cards, 3D objects, captions, transitions, ...), say how it was
probably made. `likely_tool` uses short, plain names:
- `ai_image+i2v`
- `ai_video`
- `after_effects`
- `capcut_template`
- `stock`
- `3d_render`
- `screen_recording`
- `live_action`
- `html_motion`
- `manual_animation`

`how` gives the steps, for example "still generated, then animated with a slow push".

Every entry needs `evidence` that you actually saw, and a `confidence` from 0 to 1:

| evidence seen | suggests | typical confidence |
|---|---|---|
| hands or text morphing, objects melting at the edges, physics that drifts | `ai_image+i2v` or `ai_video` | 0.6-0.8 |
| the same caption animation and font as a well-known template, pixel-identical across videos | `capcut_template` | 0.7 |
| perfectly smooth keyframed curves with motion blur, layered depth, expression wiggle | `after_effects` | 0.5-0.7 |
| a watermark ghost, a mismatched grade or resolution, a generic subject | `stock` | 0.5 |
| real reflections and shadows on a rotating object, a clean turntable | `3d_render` | 0.6 |

The rules:
- **No evidence, no entry.** "It looks AI" is not evidence; "the fingers merge at 3.1 s" is.
- Keep the confidence honest. Below 0.4, write the entry only if a recipe needs it, and name the doubt
  in `evidence`.
- The toolchain feeds the recipes, which say how a pass can build each element locally. It never
  names a paid service to buy.

## 6. Habit versus one-off; rules of thumb

Before writing anything down as the style, ask whether the creator would do it in a video about
something else.
- **A habit (style)** appears whatever the topic: the way numbers become sizes, a device every ~3 s,
  silence before the payoff. It goes into `rules_of_thumb` and is counted across videos.
- **A content need (one-off)** is there because this topic demanded it, for example a screen recording
  because the video is a tutorial for one app. Keep it in `visual_ideas` and `decisions`, but not in
  `rules_of_thumb`.

Write `rules_of_thumb` as short imperatives in plain English: "turn every number into a size", "never
hold a framing past 4 s", "prove claims with the real screen". Synth normalises them (lowercase,
punctuation stripped, whitespace collapsed) and counts them with `seen_in`, so reuse the same wording
across videos when you mean the same rule. A rule seen in 30 % or more of the videos is a rule
candidate; below that it is an option.

---

## 7. From creator.json to the style

1. `creator.py check <pack>`. Fix every reported `<id>/creator.json: <field>` line until it exits 0.
2. `creator.py synth <pack> --train <ids> --exclude <hold-out ids>`. This writes `creator-mind.json`
   with:
   - `ideation`: the method distribution, the playbook per message type (for example "number →
     scale_shift 60 %, data_viz 30 %") with reference examples, recurring patterns, the
     literal-vs-figurative share and the craft notes per method;
   - `story_template`;
   - `hooks`;
   - `retention`, including `never_more_than_s`;
   - ranked `principles` and `problems`;
   - `toolchain` consensus per element;
   - `rules_of_thumb` with `seen_in`;
   - `checklist`.
3. Fill STYLE.md, section **"How the creator thinks"**, from `creator-mind.json`. Write it in the
   order below and copy the numbers, never round them into vague words:
   - **How they find visual ideas**: the playbook, one line per message type with its shares, plus
     the literal share. Add one example per top method, written fresh in generic words (not the
     reference's own content).
   - **What they want the viewer to feel**: the top emotions and the audience promise in one line.
   - **The story they tell**: the template order with the median duration of each part, and the
     common hook types.
   - **How they hold attention**: devices per 10 s, and "never more than N s without a device".
   - **Their principles**: the top 5, each with the problem it usually solves.
   - **What they never do**: the most common `alternative_rejected` themes.
   - **How it is made**: the toolchain consensus with confidence (only entries of 0.6 or more as
     rules; the rest as "probably").
   - **Rules of thumb**: rule candidates first, then options, each marked with `seen_in`.
4. In a new plan (phase B), run `creator.py ideate <pack> --line "<line>" --type <message_type>` for
   every visual beat. It returns the creator's top methods for that message type with reference
   examples, their why-patterns, the craft notes, and the rule: write 3 ideas, 2 in the creator's top
   methods and 1 wild card in a method they rarely use. Then answer the `checklist` questions for the
   beat and pick one idea. Take the method, never the reference idea itself: a new video must pass
   the style's checks AND feel new next to the references.
5. Hold-out: `creator.score_structure(pred, actual)` scores a predicted part structure against a
   held-out video. A part boundary matches within 1.0 s, and the result is the F1 and the timing error.
   `creator.retention_events()` turns predicted devices into events for `prove.py holdout`.
