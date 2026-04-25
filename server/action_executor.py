"""DEPRECATED — V3 removed the hardcoded action dispatcher.

The action executor has been replaced by the free-form search/replace
patch applier in server/patch_applier.py.

Import from server.patch_applier instead:
    from server.patch_applier import apply_search_replace_patch
"""

import warnings

warnings.warn(
    "server.action_executor is DEPRECATED in FlakeForge V3. "
    "Use server.patch_applier.apply_search_replace_patch instead. "
    "The hardcoded 7-action dispatch has been replaced by free-form "
    "search/replace patching.",
    DeprecationWarning,
    stacklevel=2,
)

from .patch_applier import apply_search_replace_patch

# Legacy alias
build_patch_spec = apply_search_replace_patch
