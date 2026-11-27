# Behaviour evals (for maintainers)

Each scenario checks that an agent reading `SKILL.md` does the right thing in an isolated sandbox: its own
store (`STYLE_LAB_HOME`), its own home folder, a fake downloader, and scripted owner replies.

1. Check the weekly usage first; a full batch is 8 agents.
2. `python evals/run_evals.py list`, then for each scenario
   `python evals/run_evals.py prepare <id> --work evals/.runs/<id>-<stamp> --real-home ~` (exit 3 = skipped,
   a need is missing; it is reported, never hidden).
3. Start one fresh agent per prepared scenario (a mid-tier model, in the background, in parallel) with only:
   "Read `<work>/PROMPT.md` and do it."
4. `python evals/run_evals.py check <id> --work <work> --real-home ~`, then
   `python evals/run_evals.py report evals/.runs`.
5. For a failed criterion, find the cause in `SKILL.md` or `references/` and fix the docs; re-run only the
   failed scenarios, at most two rounds. Change a criterion only when it is provably wrong.

Every scenario also checks that the real store and skills folder were not touched (`no_leak`), that the
agent never got stuck, and that it spoke to the owner. Fixtures are rebuilt with
`python evals/fixtures/build_fixtures.py` (generic content only).
