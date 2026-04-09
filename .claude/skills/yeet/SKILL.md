---
name: yeet
description: "Stage, commit, push, and open a GitHub pull request in one seamless flow using GitHub CLI (gh). Use this skill whenever the user says 'yeet', asks to ship changes, wants to quickly push and open a PR, or requests a one-shot commit-push-PR workflow. Also trigger when the user says things like 'PR 만들어줘', '푸시하고 PR 열어줘', or 'ship it'."
---

# Yeet — One-shot commit, push & PR

This skill automates the full git workflow from staging changes to opening a draft PR. It exists because this multi-step process is tedious to do manually and easy to mess up (forgetting to push, wrong branch name, malformed PR body, etc.).

## Prerequisites

Before starting, verify both of these — if either fails, stop and tell the user what to do:

1. **GitHub CLI installed**: Run `gh --version`. If missing, ask the user to install it.
2. **Authenticated session**: Run `gh auth status`. If not authenticated, ask the user to run `gh auth login`.

## Naming conventions

| Item | Format | Example |
|------|--------|---------|
| Branch | `codex/{description}` | `codex/fix-auth-timeout` |
| Commit message | `{description}` (terse, lowercase) | `fix auth timeout on retry` |
| PR title | `[codex] {description}` | `[codex] fix auth timeout on retry` |

The `{description}` should summarize the full diff, not just echo the user's words. Look at what actually changed.

## Workflow

### 1. Branch

- If currently on `main`, `master`, or the repo's default branch → create a new branch: `git checkout -b "codex/{description}"`
- Otherwise → stay on the current branch (the user is already working on something)

### 2. Stage & commit

```bash
git status -sb              # Show the user what's about to be committed
git add -A                  # Stage everything
git commit -m "{description}"
```

### 3. Run checks

If pre-commit hooks or CI checks exist, let them run. If they fail because of missing dependencies or tools, install what's needed and retry once. If they fail for other reasons, stop and tell the user.

### 4. Push

```bash
git push -u origin $(git branch --show-current)
```

If the push fails due to workflow/auth errors, try pulling from the base branch first and retry:
```bash
git pull origin main --rebase
git push -u origin $(git branch --show-current)
```

### 5. Open draft PR

Write a proper PR description to a temp file to avoid newline escaping issues:

```bash
cat > /tmp/pr-body.md << 'EOF'
## Summary
<detailed prose: what changed, why, what it affects>

## Changes
<bullet list of key changes>

## Testing
<how this was validated>
EOF

GH_PROMPT_DISABLED=1 GIT_TERMINAL_PROMPT=0 \
  gh pr create --draft \
  --title "[codex] {description}" \
  --body-file /tmp/pr-body.md \
  --head "$(git branch --show-current)"
```

The PR description should be **detailed prose** covering:
- What the issue or feature is
- The cause and effect on users
- What was changed and why
- Any tests or checks used to validate

### 6. Report back

Show the user the PR URL so they can review it.
