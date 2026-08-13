"""CLI subcommands must be profile-aware (v0.3): resolve the Hermes home from
HERMES_HOME when set, falling back to ~/.hermes — never hardcode ~/.hermes.

Before this fix, `hermes -p <profile> beaglemem status|build|migrate` read
the DEFAULT profile's data dir, so a named profile's CLI operated on the
wrong store."""
import os

from beaglemem import cli


def test_hermes_home_defaults_to_user_home(monkeypatch):
    """No HERMES_HOME set → fall back to ~/.hermes (default profile)."""
    monkeypatch.delenv("HERMES_HOME", raising=False)
    assert cli._hermes_home() == os.path.expanduser("~/.hermes")


def test_hermes_home_respects_env(monkeypatch):
    """HERMES_HOME set (e.g. ~/.hermes/profiles/coder) → use it verbatim."""
    monkeypatch.setenv("HERMES_HOME", "/home/user/.hermes/profiles/coder")
    assert cli._hermes_home() == "/home/user/.hermes/profiles/coder"


def test_status_uses_profile_home(tmp_path, monkeypatch, capsys):
    """status must read the PROFILE's beaglemem-data, not ~/.hermes.

    Build a minimal store in a fake profile home, set HERMES_HOME to it, and
    assert status reports the profile's model rather than the default's."""
    from beaglemem.store import BeagleStore
    profile_home = tmp_path / "profiles" / "coder"
    data_dir = profile_home / "beaglemem-data"
    data_dir.mkdir(parents=True)
    store = BeagleStore(str(data_dir / "beaglemem.db"), create=True)
    store.persist_model(["hello", "world"], {"hello": 1, "world": 1}, {
        "dim": 64, "window": 2, "min_count": 2,
        "tokenizer_fingerprint": "fp", "regex": "r", "stemmer": None,
        "consumed_sentences": 1, "corpus_source": "state_db",
        "last_seen_id": 1, "encoder_version": "idf-v1",
    })
    store.close()

    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    cli._status(None)
    out = capsys.readouterr().out
    assert "2 words" in out
    assert "dim=64" in out


def test_status_falls_back_to_default(tmp_path, monkeypatch, capsys):
    """status with no HERMES_HOME and no default store says NOT built."""
    monkeypatch.delenv("HERMES_HOME", raising=False)
    # Point HOME at an empty temp dir so ~/.hermes resolves somewhere clean.
    monkeypatch.setenv("HOME", str(tmp_path))
    cli._status(None)
    out = capsys.readouterr().out
    assert "NOT built" in out
