# AGENTS.md — Repository Operating Rules

Rules for any agent (AI or human) working in this repository.

## Privacy Rule: No Personal Tokens in the Repo

This is a **public repository** (`github.com/ar-nim/beaglemem`). Personal data
must NEVER be committed, even in tests, examples, or commit messages.

### What counts as a personal token

- **Medications / health data**: e.g. `<MEDICATION_NAME>`, any drug name
  tied to a person, dosage regimens (that is why the test corpus says
  `Vitamin C` instead).
- **Names / email addresses**: real people's names, personal emails,
  nicknames (author identity in `git log` is exempt — it is the account owner).
- **Places tied to a person**: home address, specific street names, frequented
  locations (e.g. `<HOME_DISTRICT>`, `<EMPLOYER_BUILDING>`).
- **Services / account IDs**: `<GOV_SERVICE_ID>`, `<INSURANCE_ID>`, phone
  numbers, NIK/KTP/Passport numbers, bank accounts.
- **Company-internal terms** that could identify the owner's employer or
  medical providers.

### Enforced BEFORE first push (the cheap gate)

The banned-token list lives in a **gitignored** file so it can be updated
without re-leaking the values into the public repo. AGENTS.md itself only
names token *classes*, never real values.

```bash
# tokens live in .gitignore'd scripts/privacy_tokens.txt (one per line)
# IF that file is missing, list tokens inline with grep -E 'tok1|tok2|...'
git grep -n -I -E "$(paste -sd'|' scripts/privacy_tokens.txt)" -- . \
  ':(exclude)AGENTS.md' ':(exclude)scripts/privacy_tokens.txt'
```

Zero matches = proceed.

### If a token was already pushed (the expensive gate)

`git filter-repo` is the ONLY correct fix — a follow-up commit that removes a
token from the working tree does NOT remove it from history. The token stays
in the old commits, visible to anyone who clones.

```bash
# 1. Build the replacement map: real_token ==> neutral_replacement
#    (real values come from scripts/privacy_tokens.txt, never hardcoded here)
cat > /tmp/pii_replacements.txt <<'EOF'
REAL_TOKEN_1==>neutral_placeholder_1
real_token_1==>neutral_placeholder_1
EOF

# 2. Rewrite ALL history (blobs only — this does NOT touch commit messages)
git filter-repo --replace-text /tmp/pii_replacements.txt

# 3. Commit messages need a SEPARATE pass if the token appears in one
cat > /tmp/pii_messages.txt <<'EOF'
REAL_TOKEN_1==>neutral_placeholder_1
EOF
git filter-repo --replace-message /tmp/pii_messages.txt

# 4. filter-repo deletes origin — re-add it
git remote add origin git@github.com:ar-nim/beaglemem.git

# 5. Force-push the rewritten history
git push --force --all
git push --force --tags

# 6. VERIFY — this MUST return nothing (exclude AGENTS.md + the token list):
git log --all -p | grep -iwE "REAL_TOKEN_1" --exclude=AGENTS.md --exclude=privacy_tokens.txt
```

**Pitfall:** `--replace-text` only rewrites file blobs. If the token appears
in a commit message (e.g. "scrub `<MEDICATION_NAME>` from test"), run
`--replace-message` too — otherwise the message keeps the token forever.

**Order matters:** scrub BEFORE pushing anything new. Once a commit is on the
remote, the token is public forever to anyone who cloned/forked it — rewriting
your history only fixes your copy.

### Testing discipline (why this rule exists)

Tests and examples are the highest-risk place for personal tokens: a
"realistic-looking" fixture often imports real personal data. The beaglemem
test corpus uses neutral synthetic content on purpose. If you need realistic
data, synthesize it — never copy real values.

## TDD Rule

- Every feature/bugfix starts with a failing test (RED), then minimal
  implementation (GREEN), then refactor.
- No production code without a failing test first.
- Test files live in `tests/`, not in `scripts/` or `beaglemem/`.

## Conventional Commits

- Format: `type(scope): description`
- Types: `feat`, `fix`, `docs`, `test`, `chore`, `refactor`, `perf`
- One logical change per commit.

## Repo Layout

- `beaglemem/` — the plugin package (self-contained, numpy-only).
- `tests/` — pytest suite (run `pytest -q` from repo root).
- `scripts/` — build/verify/probe/update CLI scripts.
- `examples/` — demo corpus generator + verify config.
- `SPEC.md` — project specification; `README.md` — user docs.

## Verification before "done"

- `pytest -q` → all green.
- `scripts/verify.py --data <data> --config examples/verify.demo.json`
  → `ACCEPTANCE: PASS`.
- Privacy grep (above) → zero matches.
- `git status --short` → clean (or staged commits ready to push).
