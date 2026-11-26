# Stage C: create or update a client in chat

A **client** is a brand that any style can wear: the style gives layout, motion, sizes and
rhythm; the client gives colours, fonts, logo, tone and its dos and don'ts. One client can be
linked to several styles (one default), and one style to many clients. Profiles live in the
global store (`references/store-schema.md`), never in the skill folder.

Use this when the user says "לקוח חדש", "תוסיף לקוח", names a brand that is not in
`clients.py list`, sends a logo, a brand colour or a website for a brand, or asks for a palette
(פלטת צבעים). Name the step in Hebrew ("שלב C: פרופיל לקוח"), one question at a time, and say
why each answer matters.

## Commands
```
python scripts/clients.py list [--json]
python scripts/clients.py show <slug, name or alias>
python scripts/clients.py add --name "<name>" [--slug <slug>] [--industry ".."] [--tone ".."]
                              [--color "#RRGGBB" ...] [--theme dark|light|both]
python scripts/clients.py set <client> field=value [field=value ...]
python scripts/clients.py palette <client> [--pick monochrome|analogous|accent]
python scripts/clients.py link <client> <style> [--default]
python scripts/clients.py unlink <client> <style>
python scripts/clients.py remove <client>       # registry entry only, the folder is kept
python scripts/clients.py restore <slug>        # re-register a kept folder
```
`set` fields: `name industry audience tone notes theme`, the lists
`aliases languages logo brand_colors dos donts assets` (items split on ","; so write each do or
don't without a comma), and `fonts.heading`, `fonts.body`, `fonts.mono`. Every write prints one
line; read it, and on an error fix the input instead of retrying blindly.

## The steps
**C1. Name.** Ask for the client's name as they write it (Hebrew is fine). The slug is made
from it (Hebrew is transliterated); pass `--slug` only when the user wants a specific one. If
`add` says a kept folder exists, the client was removed earlier: offer `restore`. Also ask the
industry, the audience and the tone in a few words, and which languages the videos use.
```
python scripts/clients.py add --name "<name>" --industry "<industry>" --tone "<tone>"
python scripts/clients.py set <slug> audience="<audience>" languages=he
```

**C2. Logo.** Ask for the logo files (SVG or a transparent PNG is best; a JPG on white works
only on white). Copy them (never link) into `~/.claude/style-lab/clients/<slug>/logo/` and
record the copies:
`clients.py set <slug> logo=clients/<slug>/logo/logo.svg,clients/<slug>/logo/logo-white.svg`.
No logo yet: continue, and say videos will carry the name in the heading font until one comes.

**C3. Brand colour: one colour, or the website.**
- One colour: take the hex the user gives (or read it from the logo file and confirm it with
  the user). `clients.py set <slug> brand_colors=#RRGGBB` (more than one: the main one first,
  it feeds the palette).
- A website: only a link the user gave. Fetch the page and its CSS, count the colours used on
  buttons, headings and links, and read `font-family`. Show what you found (the top 3-5
  colours with where they appear, the fonts) and ask which is the brand colour. Never save a
  guess without the user's yes. Nothing is downloaded beyond the page itself.

**C4. Palette.** `python scripts/clients.py palette <slug>` prints three proposals
(monochrome, analogous, accent) with their checks; `references/palette.md` explains them. Show
the three on real frames (a caption on its pill, a headline with one emphasis word, a CTA),
one marked **ההמלצה שלי**, with one plain line each on how it feels. A proposal that fails a
check is shown with what fails, never as the recommendation. Save the pick:
`python scripts/clients.py palette <slug> --pick accent`.
Then set the theme the client's videos use: `clients.py set <slug> theme=dark|light|both`.

**C5. Fonts.** Default: the course fonts (Karantina 700 headings, Heebo body, Consolas mono).
If the client has its own, ask for the font names (Google Fonts) or the files and their licence;
copy files into `clients/<slug>/fonts/`. Hebrew videos need fonts with Hebrew glyphs: check, and
if the brand font has none, keep it for Latin words and pair it with a Hebrew font.
`clients.py set <slug> fonts.heading="<font> 700" fonts.body="<font>"`.

**C6. Dos and don'ts.** Ask what the brand never shows and what it always wants ("no red",
"never show money falling", "always the product in hand"). They become rules for every video of
this client, above the style's taste rules.
`clients.py set <slug> dos="logo top right,product shot in every video" donts="no red,no stock people"`.
Product shots or UI captures they give go in `assets=...`, and anything else in `notes=...`.

**C7. Link styles.** `styles.py list` shows the ready styles. Ask which style(s) this client
uses and which is the default; link with `clients.py link <slug> <style> --default` (the
`--default` link clears any other default). No style yet: offer phase A, or link later.

**C8. Check and say it back.** `clients.py show <slug>` and summarise in Hebrew in a few lines:
name, logo, palette (with its three main colours), theme, fonts, dos and don'ts, linked styles
with the default. Ask for corrections and apply them with `set`.

## Using a client in an edit (B0)
- The client's `palette` roles and `fonts` replace the style's brand tokens; the style's
  layout, sizes, motion and colour discipline stay (one accent stays one accent).
- `dos` and `donts` are hard rules for that edit; quote them in the plan when a beat touches one.
- The logo comes from `logo`; never redraw or recolour a logo unless the user asks.
- A client without a palette: run C4 first, or use the course brand and say so.

## Changing or removing
- Rename or fix anything with `set`; a new brand colour means a new palette: rerun C4.
- `remove` only hides the client (the folder and its files stay); `restore <slug>` brings it
  back. Nothing in the store is deleted by these commands; ask before deleting any file by hand.
