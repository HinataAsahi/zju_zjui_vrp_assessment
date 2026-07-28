# Project Collaboration Rules

This repository is for the ZJU VRP assessment project in ZJUI. Follow these
rules when working in this repository.

## Skill Workflow

- Do not invoke the full brainstorming skill for clear, localized changes,
  including bug fixes, algorithmic enhancements, tests, configuration changes,
  and modifications confined to one or two modules.
- For these tasks:
  1. inspect the relevant implementation;
  2. briefly state material assumptions and the intended approach;
  3. implement directly;
  4. add or update tests;
  5. run relevant verification.
- Use the full brainstorming skill only when the user explicitly requests it,
  or when the task involves a new architecture, multiple unresolved product
  decisions, or changes that are expensive to reverse.
- After a design is approved and work is ready to move into implementation,
  invoke `superpowers:writing-plans` to create an implementation plan.
- Use `superpowers:test-driven-development` when implementing new code or a
  bugfix where tests can define the expected behavior.
- Use `superpowers:systematic-debugging` before fixing failing tests,
  unexpected behavior, or runtime errors.
- Use `superpowers:verification-before-completion` before claiming work is
  complete, tests pass, a fix works, or a commit/push is ready to report.

## Current Project Flow

The current agreed sequence is:

1. Review and approve the Chinese learning-note design spec.
2. Write the formal Chinese technical learning note.
3. Review the learning note.
4. Plan and implement the feasible baseline, official `solve.py`, evaluation
   tooling, and later report/presentation materials.

Do not jump directly to implementation before the relevant design/review gate is
cleared, unless the user explicitly changes the workflow.

## Repository Hygiene

- Keep generated caches and local experiment outputs out of git.
- Do not commit the raw project data directory unless the user explicitly asks
  for it.
- New dependencies may be installed when needed for documentation generation,
  implementation, evaluation, or training. Explain why the dependency is needed,
  prefer the smallest practical dependency set, and record reproducibility
  information in the relevant docs or environment files.
- Use focused commits with conventional-style messages, such as
  `docs: ...`, `feat: ...`, `fix: ...`, or `test: ...`.
- Push completed commits to the configured GitHub remote when appropriate.
