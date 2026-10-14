"""Verify that a runnable submission includes its entrypoint and local backend."""
import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/create_course_submission.py"
spec = importlib.util.spec_from_file_location("course_submission", SCRIPT)
package = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package)


def test_missing_desktop_entrypoint_stops_distribution(tmp_path):
    with pytest.raises(RuntimeError, match="visualization_test2.py"):
        package.validate_package(tmp_path)


def test_local_framework_data_module_is_preserved(tmp_path):
    assert "data" not in package.ignore(str(tmp_path), ["data", "__pycache__"])
    assert "__pycache__" in package.ignore(str(tmp_path), ["data", "__pycache__"])
    assert package.ignore(str(tmp_path), [".env", ".env.production"]) == {".env", ".env.production"}
