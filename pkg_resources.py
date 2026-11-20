# Copyright (c) 2026 Ali Nihad Qadir -- Middle Technical University / Polytechnic College.
# All Rights Reserved. Unauthorized copying, redistribution, or attribution is prohibited.
"""
Shim for pkg_resources on Python 3.12+ / 3.13 environments where pkg_resources
was removed or not installed by default with setuptools.
Provides resource_filename needed by face_recognition_models.
"""
from __future__ import annotations
import os
import importlib.util

def resource_filename(package_or_requirement: str, resource_name: str) -> str:
    spec = importlib.util.find_spec(package_or_requirement)
    if spec and spec.origin:
        base_dir = os.path.dirname(spec.origin)
        return os.path.normpath(os.path.join(base_dir, resource_name))
    # Fallback to direct path search in sys.path
    import sys
    for p in sys.path:
        candidate = os.path.join(p, package_or_requirement, resource_name)
        if os.path.exists(candidate):
            return os.path.normpath(candidate)
    raise FileNotFoundError(f"Resource '{resource_name}' not found for '{package_or_requirement}'")
