"""
Pytest configuration and fixtures for RLLM proxy tests.
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
proxy_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(proxy_dir))


@pytest.fixture(autouse=True)
def reset_global_state():
    """Reset global state between tests."""
    from sdk.proxy.context import set_session_context
    from sdk.proxy.tracer import set_tracer

    # Reset context
    set_session_context(None)

    # Reset tracer (will be recreated on next get_tracer call)
    set_tracer(None)

    yield

    # Cleanup after test
    set_session_context(None)
