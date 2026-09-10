# AI Assistant Guardrails

How Claude Code (and any other agent) is fenced off in this repo, and why the fence is
shaped the way it is.

The short version: **the git setup matters more than the permission config.** Commits and
branches are cheap and reversible. The only operations that actually lose work are
force-pushes, `reset --hard`, `clean -f`, and pushing to a shared branch. Fence those off
and let the assistant do the rest without interruption.

---

## 1. Git-side guardrails (these are the real protection)

- **Work on a feature branch, never `main`.** Branch names follow
  [`onboarding.md`](onboarding.md) — `feat/<topic>`, `fix/<topic>`, `chore/<topic>`.
- **Enable branch protection on `main`** in GitHub so even a stray push can't land
  unreviewed. This is a repo-settings change, not something in this file:

  ```bash
  gh api -X PUT repos/:owner/:repo/branches/main/protection \
    -F required_pull_request_reviews.required_approving_review_count=0 \
    -F enforce_admins=false \
    -F required_status_checks=null \
    -F restrictions=null
  ```

  > **Status:** not yet enabled on `bthek1/model_playground` — `main` has no branch
  > protection and no rulesets. Until it is, step 2 is the only thing standing between an
  > agent and `origin/main`.
- **Nothing is lost while it's in the reflog.** `git reflog` recovers from almost any local
  mistake — a bad rebase, a lost commit, a clobbered branch. The exceptions are
  `git clean -f` (untracked files were never in the object store) and a force-push that
  drops commits nobody else fetched.
- **Commit or stash your own work before starting a session**, so there is a clean line
  between your changes and the assistant's.
- **Fully autonomous runs go in a worktree** — `git worktree add ../feature-x` — so the
  agent cannot touch your checked-out branch at all.

---

## 2. Permission config

Lives in [`.claude/settings.json`](../../.claude/settings.json) (project, committed).
Personal overrides go in `.claude/settings.local.json` (gitignored) or
`~/.claude/settings.json` (global).

**Deny wins over ask, ask wins over allow**, so the shape is a broad allow plus targeted
`ask`/`deny` rules:

| List | Contains | Effect |
|------|----------|--------|
| `allow` | `Bash(*)`, plus the read-only and local git verbs spelled out | Runs without a prompt |
| `ask` | `git push` / `rebase` / `merge`, `gh pr create\|merge`, `gh issue create\|edit\|close`, `docker compose down`, `just down-v`, `just db-reset`, `rm -rf` | Prompts every time |
| `deny` | `git push --force` / `-f`, `git reset --hard`, `git clean`, `git branch -D`, `git checkout .` | Blocked outright |

Notes on the shape:

- **`Bash(*)` stays** because the local dev loop (`just dev`, `just be-test`, `just fe-e2e`,
  `uv run …`, `npx …`) is otherwise a prompt every few seconds. The explicit git entries in
  `allow` are redundant under `Bash(*)` — they are there so the intent survives if anyone
  later narrows the wildcard. If you want to try allow-list-only for a week, delete
  `Bash(*)` and add rules back as you notice yourself approving the same prompt repeatedly.
- **Committing is unattended, pushing is not.** A commit on a feature branch is recoverable;
  a push is outward-facing. The assistant still commits because you asked it to — the
  permission only removes the prompt.
- **Use the `prefix:*` form.** `Bash(git push:*)` is the syntax that has worked in every
  version. `*` matching any text (`Bash(git *)`) works in current builds, but `:*` is the
  safer bet.
- **Chained commands need each part to match.** Claude Code parses shell operators, so
  `Bash(just *)` does not authorise `just fe-test && git push`.
- **A wrapper is a different command string.** `Bash(docker compose down:*)` does not match
  `just down-v`, even though that recipe runs exactly that. This repo's `justfile` hides two
  volume-destroying commands behind recipes (`just down-v`, `just db-reset` — both run
  `docker compose down -v`), so those recipe names are listed in `ask` in their own right. Any
  new recipe that wraps something dangerous needs the same treatment.

### The caveat that matters

**Pattern denies are bypassable in principle.** `git push origin +main` is a force-push
without the string `--force`, and `git commit -m "..." --no-verify` doesn't prefix-match a
`git commit --no-verify` rule. Rules stop accidents, not a determined command line — which
is exactly why §1 (branch protection, feature branches, reflog) is the real protection and
this section is the convenience layer. Prose rules for things a pattern cannot express are
in the "Absolute Don'ts" of [`.github/copilot-instructions.md`](../../.github/copilot-instructions.md).

### Attribution

Commits get a `Co-Authored-By: Claude …` trailer by default. To drop it, add
`"includeCoAuthoredBy": false` to a settings file (or set `"attribution": { "commit": "" }`,
the non-deprecated form).

---

## 3. Habits

- Ask for **small, logical commits** and review `git diff` (or the PR) before pushing.
- **Don't approve pushes reactively mid-session.** A push prompt is the one place where
  reading the diff first is worth the interruption — that's why it's in `ask` rather than
  `allow`.
- Plans are GitHub issues (`gh issue create --label plan`); creating, editing and closing
  them prompts, because they are the project's record.
- Add rules when you notice yourself approving the same prompt repeatedly, not in advance.
