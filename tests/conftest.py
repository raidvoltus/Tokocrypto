"""
PYTEST CONFIGURATION
Description: Pytest configuration and shared fixtures for NVRA trading bot tests.
"""

import os
import sys
import pytest
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure pytest
def pytest_configure(config):
    """Configure pytest environment."""
    # Set test environment variables
    os.environ.setdefault("NVRA_ENV", "test")
    # Ensure no model is accidentally loaded from user's system
    if "NVRA_MODEL_PATH" in os.environ:
        del os.environ["NVRA_MODEL_PATH"]
    if "NVRA_MODEL_SHA256" in os.environ:
        del os.environ["NVRA_MODEL_SHA256"]


def pytest_collection_modifyitems(config, items):
    """Modify test collection."""
    for item in items:
        # Add markers for test organization
        if "test_model_loader" in str(item.fspath):
            item.add_marker(pytest.mark.model_loader)
        elif "test_inference" in str(item.fspath):
            item.add_marker(pytest.mark.inference)


# Shared fixtures
import tempfile
from unittest.mock import Mock
import pickle


@pytest.fixture(scope="session")
def temp_session_dir():
    """Session-scoped temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_model():
    """Generic mock model fixture."""
    model = Mock()
    model.predict_proba = Mock(return_value=[[0.3, 0.7]])
    return model


@pytest.fixture
def temp_model_pickle(mock_model):
    """Create a temporary pickle model file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump({"model": mock_model, "version": "test_1.0"}, f)
        yield model_path
