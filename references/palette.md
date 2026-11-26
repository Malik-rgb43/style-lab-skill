# Palette from one colour

`scripts/palette.py` turns one brand colour into three full palettes. Each one fills every
colour role a video style uses, for a light and a dark theme, and carries contrast and
colour-blind checks. Use it when a client gives one colour ("הצבע שלנו הוא ..."), or when a
style needs its own colours.
```
python scripts/palette.py "#RRGGBB"                    # the three proposals as a table
python scripts/palette.py "#RRGGBB" --json             # the same as JSON (all roles, ramps, checks)
python scripts/palette.py "#RRGGBB" --save <client> --pick monochrome|analogous|accent
python scripts/clients.py palette <client> [--pick <kind>]   # same, from the client's first brand colour
```
`--save` writes the picked proposal into the client's `client.json` as `palette` (with
`source_hex` and the `picked` date). A proposal that fails a check still saves, with a warning:
say it to the user and offer another proposal or a nudged colour.

Colour maths: OKLCH / OKLab (Björn Ottosson) for every lightness and hue move, WCAG 2.x
contrast ratios, CIEDE2000 for colour differences, Machado 2009 deuteranopia (severity 1.0)
for the colour-blind check.

## The three proposals
All three share the brand, the tonal ramp, the neutrals and the role table below. They differ
only in the **accent** (highlights, emphasis words, CTA) and the **CTA fill**.

| Proposal | Accent | CTA fill | Feels like |
|---|---|---|---|
| `monochrome` | the brand's own ramp step 300 (step 800 when the brand is light, OKLCH L >= 0.66) | the brand | calm, one colour family |
| `analogous` | the neighbour hue (brand hue +-30, toward warm for cool brands) at the brand's lightness, chroma at least 0.10 | the accent | warmer or cooler, still related |
| `accent` | a complementary accent: a cool brand (hue 90-270) gets warm orange (hue 50), any other brand gets blue (hue 250), at L 0.70, C 0.16. A neutral brand (chroma under 0.02: grey, black, white) always gets blue | the accent | one strong pop for CTAs and emphasis |

Every accent is then moved (lightness first, then hue in 10 degree steps) until it is clearly
apart from the brand: deuteranopia ΔE2000 >= 10 and ΔE2000 >= 20.

## Ramps
The brand hue as an 11-step ramp. Each step is an OKLCH lightness and a chroma given as a
factor of the brand's own chroma. Steps darker than 500 drift toward blue (hue 260), up to 22
degrees at 950. Neutrals use the same lightness steps at the brand hue with chroma
min(0.008, brand chroma).

| Step | 50 | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 | 900 | 950 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| L | 0.97 | 0.94 | 0.895 | 0.82 | 0.73 | 0.64 | 0.55 | 0.465 | 0.38 | 0.30 | 0.22 |
| chroma x | 0.10 | 0.25 | 0.44 | 0.62 | 0.82 | 0.96 | 1.00 | 0.85 | 0.60 | 0.45 | 0.32 |

## Role table
Role keys: `background`, `surface`, `text`, `muted`, `brand`, `brand_deep`, `tint`, `accent`,
`highlight`, `caption_pill`, `cta`. `(L, C)` = OKLCH at the brand hue with C capped at the
brand chroma; `ramp:N` / `neutral:N` = that step; `accent:L` = the accent hue at lightness L
with 0.45 of the accent's chroma.

| Role | Light theme | Dark theme | Used for |
|---|---|---|---|
| background | neutral:50 | (0.18, 0.020) | the ground behind everything |
| surface | (0.995, 0.004) | (0.235, 0.025) | cards and panels |
| text | (0.22, 0.030) | (0.97, 0.008) | body and headline text |
| muted | (0.48, 0.020) | (0.74, 0.020) | secondary text, labels |
| brand | the brand colour | the brand colour | logo-colour shapes, lines, fills |
| brand_deep | ramp:800 | ramp:900 | deep fills, shadows in brand colour |
| tint | ramp:200 | ramp:800 | soft brand fills behind text |
| accent | the proposal accent | the proposal accent | the one pop colour (fills, marks) |
| highlight | accent:0.88 | accent:0.80 | marker behind an emphasised word |
| caption_pill | ramp:800 | (0.13, 0.020) | the caption background |
| cta | the proposal CTA fill | the proposal CTA fill | the call-to-action button or bar |

Each theme then holds its roles to the bars by moving lightness only (hue and chroma kept):
text to 7:1 on background and surface (built for AAA), muted to 4.5:1 on background, brand to
3:1 on background. The CTA, highlight and caption pill move until their best label reaches
4.5:1.

## Labels (words on a coloured fill)
Each theme stores `labels`, the colour for words on a fill:
- `caption`, `cta`, `highlight`: the best of the theme's text, background, white or black on
  `caption_pill`, `cta` and `highlight` (most contrast wins).
- `accent_text`: the accent used for large headline words on the background. **The accent role
  stays the visible fill** (shapes, marks, the CTA); `accent_text` is the same hue moved in
  lightness only, as far as 3:1 on the background needs (and still colour-blind apart from the
  brand). On a light background it is usually a little darker than the accent. Use
  `accent_text` for words, `accent` for fills; never small body text in either.

## Checks and bars
Every proposal carries `checks`. Each number is the worse of the two themes; `pass` is true only
when every number meets its bar (compared on the raw values) and `one_accent` is true.

| Check | Bar | What it measures |
|---|---|---|
| text_on_background | 4.5 | text on background |
| text_on_surface | 4.5 | text on surface |
| muted_on_background | 4.5 | muted on background |
| brand_on_background | 3.0 | brand shapes on background (WCAG non-text) |
| caption_text | 4.5 | caption label on caption_pill |
| cta_text | 4.5 | CTA label on the CTA fill |
| text_on_highlight | 4.5 | highlight label on highlight |
| accent_large_text | 3.0 | labels.accent_text on background (large text) |
| accent_vs_brand_cvd | 10 | accent vs brand ΔE2000 as a deuteranope sees them |
| one_accent | true | accent vs brand ΔE2000 >= 20, and every non-accent role with chroma >= 0.02 stays within 40 degrees of the brand hue (skipped for neutral brands) |

## Showing the proposals
Show all three side by side on the user's real material, not as bare hex codes: a caption on its
pill, a headline with one emphasis word, a CTA, the background, in both themes when the client's
theme is `both`. One line per proposal on how it feels, one marked **ההמלצה שלי**, and the check
result in plain words ("כל הטקסטים קריאים, גם לעיוורי צבעים"). The user picks; then save it.

## Worked example: a teal brand, #008754
`python scripts/palette.py "#008754"`, all three proposals pass. The roles every proposal
shares (light / dark):

| Role | Light | Dark |
|---|---|---|
| background | #F1F7F3 | #0A140E |
| surface | #FBFEFC | #14221A |
| text | #0E1F15 | #F1F7F3 |
| muted | #556159 | #A1AFA6 |
| brand | #008754 | #008754 |
| brand_deep | #004F3E | #00372D |
| tint | #BEE8CF | #004F3E |
| caption_pill | #004F3E | #020905 |
| caption label | #FFFFFF | #FFFFFF |

What changes per proposal (light / dark):

| | monochrome | analogous | accent |
|---|---|---|---|
| accent | #98D4B1 | #3D5B00 | #EB7C33 |
| highlight | #C5DFCF / #ABC5B5 | #CDDFBB / #B4C5A2 | #FFCBAE / #E5B194 |
| cta (label) | #008754 (#000000) | #3D5B00 (#FFFFFF) | #EB7C33 (#000000) |
| accent_text | #5F9979 / #98D4B1 | #3D5B00 / #4C6B19 | #DD6F23 / #EB7C33 |
| accent_vs_brand_cvd | 26.55 | 15.62 | 23.77 |
| cta_text | 4.59 | 7.79 | 7.46 |

Shared check values: text_on_background 15.78, text_on_surface 15.19, muted_on_background
5.97, brand_on_background 4.1, caption_text 9.6, accent_large_text 3.03-3.06.

Read the `accent` proposal like this: a deep teal (#004F3E) for the caption pill and deep
fills, a soft mint tint (#BEE8CF) behind text, an off-white ground (#F1F7F3), and one warm
orange (#EB7C33) for the CTA and the emphasis marks. On the light ground, orange headline words
use `accent_text` #DD6F23 so they reach 3:1; the CTA button itself stays #EB7C33 with black
words. This is an example only; every brand gets its own numbers from the script.
