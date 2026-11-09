# The source workflow, step by step

Source: Joseph | Video Editing, "How to Edit VIRAL Talking Head Videos Using ONLY Claude! (New
Minimal Style)", YouTube `g9xoKm5DEeU`, uploaded 2026-10-31, 20:57. Analysed 2026-11-02:
all 37,701 frames decoded, 40 contact sheets read, prompts read off full-resolution frames,
speech transcribed locally (faster-whisper small.en). The 6:15–7:50 and 11:06–11:44 and
19:40–20:57 segments are an advert for the creator's course and are not part of the workflow.

Tools shown: Claude desktop app in **Cowork** mode, model **Opus 5.5**, effort **Extra**; a Max
plan; Figma (the board); Remotion (the build); Chrome (assets); Google Fonts; Wikipedia and
Pexels (assets). No Higgsfield, no Seedance, no paid plugins.
**This skill swaps two of them by the user's choice:** the board is a Claude artifact (not
Figma) and the build is HyperFrames (not Remotion). Everything below records what the video
did; where it says Remotion, style-lab uses HyperFrames.

Contents: [Structure](#structure) · [Chapter 1](#chapter-1--the-style-0461554) ·
[Chapter 2](#chapter-2--training-claude-7501144) · [Chapter 3](#chapter-3--the-full-edit-11441930) ·
[Time and cost](#time-and-cost-19001935)

## Structure
| Chapter | Time | What happens |
|---|---|---|
| 1 | 0:46–5:54 | The style broken into 10 principles (`example-style-minimal-premium.md`) |
| 2 | 7:50–11:06 | Training Claude: 20 refs → board → training prompt → style skill + 225-animation library |
| 3 | 11:44–19:00 | One full edit: prompt → plan document → human plan → build → 6 revisions → sound |
| — | 19:00–19:35 | Time and cost breakdown |

Why chapter 1 comes first (6:06): "If you simply don't know how to create good videos, you're
only going to be using AI to create bad videos. You need to fundamentally understand what
makes these designs good."

## Chapter 1 — the style (0:46–5:54)
See `example-style-minimal-premium.md` for all ten principles with rules.

## Chapter 2 — training Claude (7:50–11:44)

**Step 1 — references (8:05).** Go to the creator's Instagram, find 20 of the most premium
edited videos in the style, download all of them.

**Step 2 — editor's breakdown (8:11).** Watch all 20 and break them down from an editor's
perspective. Every time a scene looks good ask: why this scene, why this design, why these
colours, why these effects, why these animations. "By doing this for every scene, you're
going to understand how this style is composed from a video editor's perspective."

**Step 3 — the board (8:29).** New Figma board ("Clean Educational Style"). Watch every edit
again, screenshot every scene you like, place it on the board. About **130 inspiration
pieces**. Then (8:44, "the important part") categorise them into the principles: all the text
scenes in one section, zoom into one text scene and break it down — **what makes it up? the
fonts, the colours, the sizes, the patterns** — then the same for every other principle,
"until we end up with a full board that breaks down how this entire editing style is made
from scratch" (9:08). "This board is the most important part of the whole training" (9:18).

Board as seen (8:31–9:10, 10:23): pages list "Full Inspiration Board", "On-Screen Text
(Captions)", "B-roll + Text + Overlay", "Imagery / Characters + Text", "Line + Text
Animations", "Line + Illustrations", "Text only scenes", "Center Illustration + Text", "Side
Illustration + Text", "Objects / Elements + Text", "Graphs + Visual Representations". Each
section: a grey panel with a title, a grid of screenshots (left), and zoomed frames with red
callout lines to short rule labels (right). Example callouts (8:59): "Multiple sans light small
text" / "Singular or double bold sans text" / "Distance of the light sans text is 75% of the big
bold text". Questions on screen: "What makes it up?" "What are the fonts?" "The colors";
"Why this design?" "Why these colors?" "Why these effects?" "Why these animations?"

**Step 4 — the training prompt (9:27).** New Claude chat, attach the folder "20 reference
videos" and the Figma board link "Clean Educational Style", send (verbatim, from the frame at
10:00 and the narration at 9:36):

> I want to teach you a new editing style. Analyze this clean premium minimal editing style
> using the 20 reference videos. Start by creating a word-by-word transcript and a
> frame-by-frame breakdown of every single video to study three main things. First, where the
> animations are placed in correlation with the script. Second, how the design is composed and
> what elements make it up. And third, the animation logic. How long the animations are and
> what they feel like so you can recreate them. Then use the Figma board as a reference that
> breaks down the elements of the style and master the style using both so that I could
> prompt you to recreate it.

On-screen labels for the three studies (10:12): "Study the animation design" · "Study the
animation logic" (a clip with a 1.4 s scrubber) · "Where it sits in the script" (the transcript
with the line highlighted).

**What Claude does (10:03–10:53).** Breaks the 20 videos down frame by frame, picks out every
frame with an animation and studies its design, its logic and where it sits in the script.
"By doing this, Claude also builds a huge library of references that it can use later to
create better designs." Then goes through the board and understands the logic behind every
design and animation. Then packages it all into a skill ("arthur-brooks-minimal-animation")
that "breaks down every detail that makes the style up, from the sizing, the layout, the
textures, the shapes, the colors and everything in between" (10:36), and extracts **225
animations** from the 20 videos into folders "with its designs as a reference and a sheet of
the keyframes that make it up" (10:43).

Library folder as shown (10:51), "References / Arthur Brooks Study / V01-A06 Box wipe, line
leads to brain", 4 items: `still.png`, `strip.jpg` (every frame tiled, each labelled), `clip.mp4`,
`notes.md`:
```
V01-A06 — Box wipe, line leads to brain
• Category: 5. Line + Illustrations
• Source: V01 frames 665–720 (22.2–24.0 s, 56 frames ≈ 1867 ms)
• Words: "neurobiological benefits to that."
• Keyframe: f715
'neurobiological' white sans gets an azure box wiping L→R letter by letter (15 f) while
shrinking 17%; the camera then pans down ~770 px (60% H) in 26 f slow-fast-slow, following a
thin 10 px azure line to a b&w engraving brain; 'benefits / to that' under it.
```
Summary line (10:54): "The reference videos are how Claude learns the logic behind the
animations and the Figma board is how Claude learns the design. And together, that's how
Claude mastered the style. And this is how you can train Claude on any style."

## Chapter 3 — the full edit (11:44–19:30)

**Setup (11:54).** Claude, model Opus 5.5, effort Extra. Raw footage: a clip cut from a
podcast on YouTube (57 s, "Four career patterns"). Drag the raw clip into Claude.

**The prompt (12:08–12:51).** From the frames at 12:36–12:53 and the narration:

> I want you to use the Arthur Brooks minimal animation skill to edit this short-form clip in
> this premium editing style.
> Start by creating a word-by-word transcript at the millisecond level and cut out all
> silences leaving 30 milliseconds of padding.
> On top of that, hand me a document with the script of my raw video on the right side, and
> text boxes on the left with recommended designs pulled from the Arthur Brooks reference
> folder, so that I can plan out and decide what needs to be built for each scene.

Why the extra lines (12:16): "Lazy inputs with AI get you lazy results." The clip has silences
baked in; and "if you let AI do the planning on its own, it's actually not creative enough to
do a great job right off the bat."

**Claude's task list (12:53):** Load the Arthur Brooks minimal animation skill · Transcribe the
clip word by word (ms timings) · Cut the silences, keep 30 ms of padding · Build the plan
document: script right, designs left · Recommend designs from the Arthur Brooks references.
Done message (15:45): "Done in 5 steps … Transcribed the clip word by word (237 words) · Cut
every silence, leaving 30 ms of padding · Matched every line to Arthur's reference designs ·
Built your edit plan document. Your edit plan is ready: 15 beats, each with three recommended
designs from Arthur's reels, the script and the clip on the right. Pick a design for every beat
or write your own notes."

**The plan document (13:11–15:45).** Title "Four career patterns *edit plan*"; meta: Clip "G -
Four career patterns - edit-ready cut 9x16 (1080x1920).mp4", Length "57.4 s · 15 beats",
Source, Style; "Plan saves automatically"; "Copy Plan". "BEAT TIMELINE — MY SUGGESTED PICKS":
a bar of 15 beats coloured by scene type, legend A-roll + captions / Overlay on A-roll / Cut-away
scene, ticks 0 s · 14 s · 29 s …. Each beat: "Beat 1 0:00.0 – 0:01.9", "Chosen: A · Opening
zoom-out + justified number", a text box "Write your plan for this beat — or pick a design below
and edit it.", DESIGN OPTIONS ×3, each with thumbnail, "A · title", category · scene type in
blue, a description, "Ref V14-A01", buttons "My pick" / "Use this" / "✓ Chosen". Right side:
SCRIPT, the clip playing that beat, SEEN IN THE REELS (e.g. "A framework announced in the
first line, like V04 ('four profiles', its quadrant graph previewed up front) and V08 (numbered
chapter cards). … opened on A-roll with the zoom-out and cut away at 1.7 s."), and a blue camera
note ("Camera: Medium shot 0–31.8 s."). Descriptions cite rules ("(rule 13)", "style guide §10").
Example option texts: "Justified block 'There's / four / career patterns,' (rule 13), 'four' big
and centred; 'psychologically' as a small line after it." · "Halftone paper; four faint shapes
side by side (steps, zig-zag, flat line, spiral) under 'four / career patterns'. The same map
comes back for each pattern." · "A No.-style card: big '4' with 'career patterns' under it and a
small 'psychologically' tag."

**How the plan was made (13:20–15:38)** — "the most important step of the whole edit":
| Script | Choice | Reason given |
|---|---|---|
| "There's four career patterns, psychologically." | A-roll + captions | "helps the viewer connect with Arthur before anything happens on the screen" |
| "There's linears who … go up and up … in their careers." | Graph: a line that climbs one step every time he says "up" | "the first visual the viewer sees … a great one to get him hooked" |
| "There are transitories who … skip around all over the place." | Back to A-roll + captions | "he's transitioning into a new type and the viewer needs to see his face again" |
| "I'm gonna be a barista and I fell in love with a girl in San Diego." | Painting animation | "show the viewer exactly what he's talking about" |
| "There are what's called expert, which is like slow and steady." | A-roll first half, then a long flat-line graph | user note: "Keep him on A-roll for the first half." |
| "My dad had the same job for 42 years." | Character call-out | "since he's describing a person" |
| "…are spirals for every seven to 12 years." | Centre illustration of a spiral | |
| "Take their career down to the studs and start again." | B-roll | "so it's easy for the viewer to feel it" |
| "First turn is hardest … leaving the French horn and becoming a scientist" | Object call-out with images of him | wraps his point |
| "Maybe in 10 years, I'll be a circus clown or firefighter" | Centred illustration of him with a circle cycling his options | user note: "Centred illustration of Arthur with a circle cycling" |
| "…pursuing your calling because you have the agility and the courage." | Line + text through calling, agility, courage | |
| "…the business of life." | Text-centric scene with quotation marks | ending |
Pattern (14:13): **"Whenever Arthur announces a new type, we stay on his face. Whenever he
describes it, we actually show it."** Goal (15:30): use your own experience as an editor to
choose scenes that make the video most fun to watch, so every scene is balanced and it flows.

**Build prompt (15:40–15:53).** Copy Plan → paste (shows as "PASTED: # Four career patterns edit
plan · Beat 1 (0:00.0–0:01.9) Script: There's …"), then:
> Build all the animations and create the edit.
> If it needs any assets, use the Chrome browser or files, and prompt me if there's anything
> you can't find.
(Second message read off frames 15:52 and the narration at 15:47.)

**What Claude does (16:00–16:50).** Reads the plan; "Found 10 references in your Arthur folder"
(a grid of the chosen library stills, each labelled with its V##-A## id); "Preparing the exact designs to copy";
"Designs ready to copy"; "Asset list ready — 6 assets": A Lady Taking Tea — Chardin (Wikipedia),
The Kiss — Hayez (Wikipedia), B-roll door opening into sunlight (Pexels), B-roll man working at a
laptop (Pexels), Arthur Brooks — portrait (Wikipedia), Arthur Brooks — speaking (Wikipedia);
"Fetching the assets…" (Chrome on Pexels "cinematic sunlight", Google search); "All 6 assets
collected"; "Creating the edit in Remotion"; "Starting with the A-roll captions…"; "Installing the
fonts from Google Fonts…"; "Captioned the full video · 236 words"; "Installed Inter + Playfair
Display"; "Applied the cinematic text design"; "Building the edit scene by scene in Remotion…":
Painting scene · Character scene · B-roll + text scene · Element callout · Graph scenes · Line +
text scene · Spiral illustration · Final Arthur illustration; "Rendering the full video and
sending it… 100%"; "Rendered and sent the full video — Final Version MP4 · 1080 × 1920 · 0:57".

**First draft (16:49):** "The first draft is going to be 60 to 70% of the way there. It's our job
to watch the video and give it detailed revisions."

**Revisions (17:00), verbatim from 17:23:**
1. The text design in the first scene isn't good. Keep the word "psychologically" up on the same scene.
2. Rough cut between the graph and the next shot. Hold the graph text until 0:06, then cut.
3. The paint texture is too heavy and doesn't move with the painting. Half the strength, and let it move with it.
4. Bad cut at 0:24. Remove the tiny "But" clip.
5. Keep the spiral on screen longer, through "for every 7 to 12 years", with the slow zoom out.
6. Add small quotation marks to the text scene at the end.
Claude: "Applied all 6 revisions — Updated Version". Then a question card: "Is the video complete
and ready for sound design, or are there any more revisions?" (1. Ready for sound design — Add
the music and the sound effects · 2. More revisions). User: "It looks perfect. Go ahead with the
sound design."

**Sound design (17:33–17:48).** "Matched the sound to your 20 Arthur references" (it used the
reference videos "to figure out which sound effects go where and what the music sounds like"):
Adding a cinematic music track · Clicks for text · Highlighters for graphs · Whooshes for
transitions · "74 sound effects placed". Renders the complete video.

## Time and cost (19:00–19:35)
Only cost: the Claude plan (Pro $20/month or Max $100/month; the creator uses Max).
| Step | Time |
|---|---|
| Cuts + planning document | 15 min |
| Planning the edit (human) | 15 min |
| Creating the entire edit | 60 min |
| Revisions | 10 min |
| Sound design + music | 15 min |
Summary card: raw video → 15 min of planning → ≈2 hours Claude creates the edit. "Next time:
half the time."
