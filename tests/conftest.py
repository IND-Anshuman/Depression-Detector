from __future__ import annotations

import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
	"""Allow `import mmds` in tests without requiring an editable install.

	This keeps the developer workflow lightweight (just `pytest`) while still
	using the src-layout package.
	"""

	repo_root = Path(__file__).resolve().parents[1]
	src_path = repo_root / "src"
	if src_path.is_dir():
		sys.path.insert(0, str(src_path))


_ensure_src_on_path()
