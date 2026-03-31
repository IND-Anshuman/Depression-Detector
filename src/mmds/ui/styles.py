from __future__ import annotations

from pathlib import Path


def glassmorphic_css() -> str:
    repo_root = Path(__file__).resolve().parents[3]
    css_path = repo_root / "app" / "static" / "css" / "dashboard.css"
    if css_path.exists():
        return css_path.read_text(encoding="utf-8")
    return """
    .mmds-shell { background: #101820; color: white; min-height: 100vh; }
    .mmds-card { border-radius: 16px; padding: 14px; background: rgba(255,255,255,0.08); }
    """
