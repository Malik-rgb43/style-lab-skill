# Changelog

All notable changes to this skill. The format follows [Keep a Changelog](https://keepachangelog.com/).

## [2.0.0-rc.1] - 2026-11-27

### Added
- **Store, clients, palette, connections check:** one global store outside the skill (`store.py`);
  styles v2 (`styles.py`), clients with a brand profile (`clients.py`), a full palette from one colour
  (`palette.py`), the connections check per stage with install hints (`doctor.py`).
- **Learning loop:** every note interpreted, layered and turned into a rule only on the owner's yes
  (`lessons.py`); taste from every pick (`taste.py`).
- **Precise cut:** forced alignment and cuts only in the quiet between words (`cutlang.py`, `align.py`,
  `cut.py`, `tighten.py`, `check_cut.py`).
- **Deep training:** per-video measurement (`measure_video.py`, `track_motion.py` with value and speed
  curves, `sound_events.py` with voice timbre and music sections, `look.py`), project tokens
  (`code_tokens.py`), fingerprint and laws (`fingerprint.py`, `laws.py`), languages and adaptation
  (`lang.py`, `font_match.py`), assets, licences and reproduction recipes (`assets_log.py`), the creator's
  way of thinking (`creator.py`), proof with hold-out and recreation (`prove.py`), cover frames
  (`cover.py`), the YuNet face model (`fetch_model.py`).
- **Frames and motion:** the plan with laws, ideas and a wild card (`plan.py`), the paid-spend gate
  (`spend.py`), recipes into the generation queue (`genprep.py`), the look check (`imgmetrics.py`,
  `lookcheck.py`), three real stills per beat (`frames.py`, `hf.py`), motion-only variants
  (`motion.py`), picks and pick pages (`picks.py`, `pickpage.py`), build gates and one draft per round
  (`buildcheck.py`), the ideas bank seeded from styles and approved finals (`ideas.py`), projects
  (`project.py`).
- **Profile input:** read a profile, rank, pick, consent-gated download into `refs/`
  (`profile_fetch.py list|page|pick|download`).
- **Studio:** styles, clients, ideas, learning and the project pick room, as a claude.ai artifact whose
  clicks Claude applies (`studio_artifact.py`) or as a local app (`studio.py`); one dark look with
  Karantina headings and bundled Heebo text.
- **Install:** install, update, seed and roll back with a dated backup (`install.py`).
- **Publishing guard:** private files, machine paths and the owner's names file (`publish_guard.py`).

### Changed
- A1 takes a folder, a profile link, this conversation or a project folder.
- B1 is the precise cut pipeline; `cut_silences.py` and `verify_cut.py` are a fallback only.
- B2-B7: plan, cost gate, frames, motion, build gates, review rounds, ideas from finals.
- The review loop logs lessons and taste; the store moved out of the skill folder.

## [1.0.0] - 2026-11-09
- The first public version: train a style from references with a style board, edit with a human plan
  and review rounds.
