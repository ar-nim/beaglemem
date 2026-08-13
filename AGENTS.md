# AGENTS.md — Repository Operating Rules

Rules for any agent (AI or human) working in this repository.

## Privacy Rule: No Personal Tokens in the Repo

This is a **public repository** (`github.com/ar-nim/beaglemem`). Personal data
must NEVER be committed, even in tests, examples, or commit messages.

### What counts as a personal token

- **Medications / health data**: e.g. `Vitamin C`, `MEDICATION_NAME`, any drug name
  tied to a person, dosage regimens (that is why the test corpus says
  `Vitamin C` instead).
- **Names / email addresses**: real people's names, personal emails,
  nicknames (author identity in `git log` is exempt — it is the account owner).
- **Places tied to a person**: home address, specific street names, frequented
  locations (e.g. `HOME_DISTRICT`, `EMPLOYER_BUILDING`).
- **Services / account IDs**: `GOV_SERVICE_ID`, `INSURANCE_ID`, phone numbers, NIK/KTP/Passport
  numbers, bank accounts.
- **Company-internal terms** that could identify the owner's employer or
  medical providers.

### Enforced BEFORE first push (the cheap gate)

```bash
grep -rIwE "vitamin|HOME_DISTRICT|transjakarta|GOV_SERVICE_ID|INSURANCE_ID|EMPLOYER_BUILDING" \
  --include="*.py" --include="*.md" --include="*.yaml" --include="*.toml" .
```

Add any new token you become aware of to this list. Zero matches = proceed.

### If a token was already pushed (the expensive gate)

`git filter-repo` is the ONLY correct fix — a follow-up commit that removes a
token from the working tree does NOT remove it from history. The token stays
in the old commits, visible to anyone who clones.

```bash
# 1. Build the replacement map: old_token ==> neutral_replacement
cat > /tmp/pii_replacements.txt <<'EOF'
Vitamin C==>Vitamin C
vitamin==>vitamin
EOF

# 2. Rewrite ALL history (blobs + commit messages)
git filter-repo --replace-text /tmp/pii_replacements.txt

# 3. filter-repo deletes origin — re-add it
git remote add origin git@github.com:ar-nim/beaglemem.git

# 4. Force-push the rewritten history
git push --force --all
git push --force --tags

# 5. VERIFY — this MUST return nothing:
git log --all -p | grep -iw "vitamin"
```

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
