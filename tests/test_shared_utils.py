from __future__ import annotations

from pathlib import Path

import pytest

import shared_utils


def test_parse_simple_yaml_scalar_types() -> None:
    assert shared_utils.parse_simple_yaml_scalar("42") == 42
    assert shared_utils.parse_simple_yaml_scalar("3.14") == 3.14
    assert shared_utils.parse_simple_yaml_scalar("true") is True
    assert shared_utils.parse_simple_yaml_scalar("false") is False
    assert shared_utils.parse_simple_yaml_scalar("null") is None
    assert shared_utils.parse_simple_yaml_scalar("'hello'") == "hello"


def test_load_simple_yaml_parses_pairs_and_skips_noise(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text(
        """
# comment
units_to_add: 100
name: boston
enabled: true
no_colon_line
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    data = shared_utils.load_simple_yaml(path)

    assert data["units_to_add"] == 100
    assert data["name"] == "boston"
    assert data["enabled"] is True
    assert "no_colon_line" not in data


def test_require_existing_path_raises_for_missing(tmp_path: Path) -> None:
    missing = tmp_path / "missing.txt"
    with pytest.raises(FileNotFoundError):
        shared_utils.require_existing_path(missing, "Missing file")
