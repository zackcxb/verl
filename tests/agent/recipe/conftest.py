"""Shared fixtures for recipe adapter tests."""

import importlib
import pathlib
import sys

import pytest

TESTS_AGENT_ROOT = str(pathlib.Path(__file__).resolve().parents[1])


def _import_with_clean_path(module_name: str):
    """Import a module after temporarily removing tests/agent from sys.path.

    pytest prepends tests/agent to sys.path, which shadows the top-level
    ``recipe`` package. This helper removes that entry for the duration of
    the import so ``recipe.deepeyes_with_gateway.*`` resolves correctly.
    """
    original_sys_path = list(sys.path)
    try:
        sys.modules.pop("recipe", None)
        sys.path[:] = [path for path in sys.path if path != TESTS_AGENT_ROOT]
        return importlib.import_module(module_name)
    finally:
        sys.path[:] = original_sys_path


@pytest.fixture()
def adapter_module():
    """Return the trainer_adapter module, imported with clean sys.path."""
    return _import_with_clean_path("recipe.deepeyes_with_gateway.trainer_adapter")


@pytest.fixture()
def adapter_class(adapter_module):
    """Return the AgentFrameworkRolloutAdapter class."""
    return adapter_module.AgentFrameworkRolloutAdapter
