"""Namespace bootstrap shared with wisent-core and sibling packages.

Uses pkgutil.extend_path so all wisent-* packages merge at import time
even though wisent-core ships a regular (non-PEP-420) package.
"""
import os
import pkgutil

__path__ = pkgutil.extend_path(__path__, __name__)

_base = os.path.dirname(__file__)
for _entry in sorted(os.listdir(_base)):
    _path = os.path.join(_base, _entry)
    if os.path.isdir(_path) and not _entry.startswith(('.', '_')):
        __path__.append(_path)
