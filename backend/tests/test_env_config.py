"""The env-file layout from #61, checked against the files themselves.

Development is one `.env` in the project root, shared with the frontend;
production is `.env.prod`, passed to compose with `--env-file`. Every way this
drifts is silent: a settings module reading `backend/.env` again finds nothing
and falls back to defaults, a required variable missing from an example only
fails on the next fresh clone, and a prod compose call without `--env-file`
quietly interpolates the *development* file. These read source text, because
the property is how the files are written.
"""

import re
from pathlib import Path

from django.conf import settings

REPO = Path(settings.BASE_DIR).parent
SETTINGS_DIR = Path(settings.BASE_DIR) / "core" / "settings"

# env("X"), env.list("X"), env.db(\n "X" …) — and whether a default follows.
ENV_CALL = re.compile(r'\benv(?:\.\w+)?\(\s*"([A-Z0-9_]+)"\s*(,\s*default\s*=)?')


def example_keys(name: str) -> set[str]:
    text = (REPO / name).read_text()
    return set(re.findall(r"^\s*#?\s*([A-Z][A-Z0-9_]*)=", text, re.M))


def required_keys(*modules: str) -> set[str]:
    keys: set[str] = set()
    for module in modules:
        source = (SETTINGS_DIR / f"{module}.py").read_text()
        keys |= {key for key, default in ENV_CALL.findall(source) if not default}
    return keys


def commands(path: Path) -> list[str]:
    """Every line of a justfile; only fenced code in Markdown (prose may quote the
    wrong form on purpose, to say why it is wrong)."""
    lines = path.read_text().splitlines()
    if path.suffix != ".md":
        return lines
    fenced, inside = [], False
    for line in lines:
        if line.lstrip().startswith("```"):
            inside = not inside
        elif inside:
            fenced.append(line)
    return fenced


def test_settings_read_the_root_env():
    assert settings.ENV_FILE == REPO / ".env"
    # The repo root, not some other parent: the files that live only there.
    assert (REPO / "justfile").is_file()
    assert (REPO / "docker-compose.yml").is_file()


def test_no_settings_module_reads_another_env_file():
    for path in SETTINGS_DIR.glob("*.py"):
        for call in re.findall(r"read_env\(([^)]*)\)", path.read_text()):
            assert call == "ENV_FILE", f"{path.name} reads {call!r}, not ENV_FILE"


def test_the_regex_sees_the_settings_it_checks():
    # A pattern that matches nothing makes every test below pass vacuously.
    assert {"SECRET_KEY"} <= required_keys("base")
    assert {"SECRET_KEY", "ALLOWED_HOSTS", "CSRF_TRUSTED_ORIGINS"} <= required_keys(
        "prod"
    )


def test_dev_example_lists_every_required_dev_variable():
    missing = required_keys("base", "dev") - example_keys(".env.example")
    assert not missing, f".env.example is missing {sorted(missing)}"


def test_prod_example_or_compose_supplies_every_required_prod_variable():
    compose = (REPO / "docker-compose.prod.yml").read_text()
    from_compose = set(re.findall(r"^\s+([A-Z][A-Z0-9_]*):", compose, re.M))
    missing = (
        required_keys("base", "prod") - example_keys(".env.prod.example") - from_compose
    )
    assert not missing, f".env.prod.example is missing {sorted(missing)}"


def test_examples_carry_no_cloud_credentials():
    # CI deploys through OIDC and `just infra-*` uses a local AWS profile;
    # neither half of the app reads them, so an example must not invite them.
    for name in (".env.example", ".env.prod.example"):
        assert not {k for k in example_keys(name) if k.startswith("AWS_")}, name


def test_the_per_half_env_files_are_gone_for_good():
    assert not (REPO / "backend" / ".env.example").exists()
    assert not (REPO / "frontend" / ".env.example").exists()
    config = [
        "docker-compose.yml",
        "docker-compose.prod.yml",
        "justfile",
        "backend/.envrc",
        ".vscode/launch.json",
        ".vscode/settings.json",
        "frontend/vite.config.ts",
    ]
    for name in config:
        text = (REPO / name).read_text()
        assert not re.search(r"(backend|frontend)/\.env\b(?!rc)", text), name


def test_every_prod_compose_call_passes_the_prod_env_file():
    files = [REPO / "justfile", REPO / "README.md", *(REPO / "docs").rglob("*.md")]
    for path in files:
        for line in commands(path):
            if re.search(r"docker compose\b.*-f docker-compose\.prod\.yml", line):
                assert "--env-file .env.prod" in line, (
                    f"{path.relative_to(REPO)}: {line.strip()}"
                )
    # The justfile routes every recipe through one variable, so check it too.
    justfile = (REPO / "justfile").read_text()
    prod_compose = "docker compose --env-file .env.prod -f docker-compose.prod.yml"
    assert f'prod_compose := "{prod_compose}"' in justfile
    assert justfile.count("{{ prod_compose }}") >= 8
