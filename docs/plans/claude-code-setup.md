# Plan: Claude Code / AI Assistant Setup

**Status:** Complete
**Date:** 2026-06-18

---

## Goal

Make this template first-class for AI-assisted development with Claude Code, and bring the existing
documentation in line with the current codebase (notably the Celery integration added after the docs
were last written). Each phase is independently deliverable.

## Background

This repo is the template for future Django + DRF + React projects, so its documentation and AI-tooling
setup are part of the product, not an afterthought. Two gaps existed:

1. **No Claude Code configuration.** There was guidance for GitHub Copilot
   (`.github/copilot-instructions.md` + `.github/agents/`) but nothing Claude-native, and no
   `.claude/settings.json` to streamline the local permission loop.
2. **Docs drifted from code.** Celery + Redis were added (see [backend-celery.md](backend-celery.md)
   and [celery-full-implementation.md](celery-full-implementation.md)) but the README, the Copilot
   instructions, and the docs-agent file still described a Celery-free stack.

---

## Phases

### Phase 1 — Claude Code Configuration

**Goal:** Give Claude Code a project permission profile so the local dev loop runs without per-command prompts.

- [x] Add `.claude/settings.json` allowing `Bash(*)` for this project
- [x] Note in `CLAUDE.md` that the allowlist removes prompts, not the "Absolute Don'ts" judgement

### Phase 2 — Claude-Native AI Docs

**Goal:** Provide a `CLAUDE.md` counterpart to the Copilot instructions so both assistants share one set of conventions.

- [x] Create root `CLAUDE.md`: project summary, "where to look first" doc map, common `just` commands, house rules
- [x] Cross-link `CLAUDE.md` ⇄ `.github/copilot-instructions.md` and state they must stay in sync
- [x] Emphasise template hygiene (keep things generic/reusable, no hardcoded project specifics)

### Phase 3 — Sync Existing Docs With Current Code

**Goal:** Bring README and AI docs up to date with the Celery integration and the new plans.

- [x] README: add Celery to the backend blurb, add a "Async tasks" tech-stack row
- [x] README: add `celery_setup.md` and `docs/plans/` to the documentation table; add an AI-assistants note
- [x] Copilot instructions: add Celery to the backend stack line and a dedicated "Async tasks — Celery" subsection
- [x] Copilot instructions: add a header pointing to `CLAUDE.md` as the in-sync counterpart
- [x] Docs-agent file: refresh the stale "Current files" list (Celery + UI plans, this plan) and note the AI-assistant docs

---

## Testing

- **Unit tests:** none — documentation and configuration only; no application code changed.
- **Integration tests:** none.
- **Manual verification:**
  - `python -c "import json; json.load(open('.claude/settings.json'))"` parses the settings file.
  - `just --list` still resolves; commands referenced in `CLAUDE.md` (`just dev`, `be-test`, `celery-up`, `flower`) exist in the `justfile`.
  - Markdown passes the repo's markdownlint config (`.markdownlint.json`).
  - All relative doc links in `README.md` and `CLAUDE.md` resolve to existing files.

## Risks & Notes

- **Drift risk:** `CLAUDE.md` and `.github/copilot-instructions.md` now duplicate conventions by design.
  Both files state they must be kept in sync; treat a change to one as a change to both.
- **Permissions scope:** `Bash(*)` is broad. It is intentional for a single-developer template to reduce
  prompts; downstream projects with multiple contributors or CI may want to tighten the allowlist.
- `.claude/settings.json` is committed (project-level). Personal overrides belong in
  `.claude/settings.local.json`, which should stay out of version control.
