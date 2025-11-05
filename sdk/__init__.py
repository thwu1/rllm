"""
Compatibility shim for the legacy top-level `sdk` package.

The SDK implementation now lives under `rllm.sdk`. This module forwards imports
so existing code that still does `import sdk` keeps working.
"""

import sys
from importlib import import_module

_sdk_mod = import_module("rllm.sdk")

# Replace this module with the real implementation to preserve attribute access.
sys.modules[__name__] = _sdk_mod
