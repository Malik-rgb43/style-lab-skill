# The Studio (local UI)

A local web app over the store and the project room. Read it before starting the Studio for the owner.

## Start
`python scripts/studio.py` starts it on 127.0.0.1 (a free port), opens the browser and prints the URL with
a per-run token. Claude starts it in the background when a pick step begins (`--no-browser` or
`--print-url`, then give the owner the printed link). It runs on the Python standard library only; nothing
is installed and nothing leaves the computer.

## Screens
- **סגנונות**: every style with its cover frame, status, kind, aspect, approved rules and linked clients;
  a style page with its rules and a rename.
- **לקוחות**: palette, fonts, dos and don'ts, client rules, projects.
- **רעיונות**: the ideas bank, filtered by kind; a clip plays on hover; "הסרה" archives (never deletes).
- **למידה**: lessons waiting for the owner; "לאשר כחוק" runs `lessons.py approve` (a possible conflict is
  shown first), "לדחות" runs `lessons.py reject`; the list of rules.
- **פרויקטים** and the **project room**: per beat the latest set of plan concepts, frames or motion
  clips, with the wild card and "ההמלצה שלי" marked; "לבחור בזה" (or keys 1-3), "עוד רעיונות", a note per
  beat, the motion mode, and the approved spend. Keys ↑ ↓ move between beats.

## How picks reach Claude
Every pick is written to the project's `picks.json` through `picks.py` (`source: studio`, `decided_by:
owner`), the same shape as the pick pages and the chat. The owner says "בחרתי"; Claude runs
`picks.py sync <project>` and acts on the "more" and own-design lists.

## Safety
Bound to 127.0.0.1; every request needs the per-run token (in the first URL, then a SameSite=Strict
cookie); every write also needs the Studio's own Origin. Files are served only from inside a project, a
pack, an idea or a client folder, never from any `refs/` folder (reference videos are never shown or
shared). Without the Studio, every pick step still has its exported page with "copy as text"
(`pickpage.py`).
