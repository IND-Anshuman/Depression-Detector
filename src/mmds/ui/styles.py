from __future__ import annotations


def glassmorphic_css() -> str:
    # Keep styling lightweight and compatible with Spaces.
    return """
    :root {
      --panel-bg: rgba(255, 255, 255, 0.08);
      --panel-border: rgba(255, 255, 255, 0.15);
    }

    .mmds-shell {
      background: radial-gradient(1200px circle at 10% 10%, rgba(120, 120, 255, 0.18), transparent 35%),
                  radial-gradient(900px circle at 90% 20%, rgba(255, 120, 180, 0.14), transparent 40%),
                  radial-gradient(900px circle at 60% 90%, rgba(120, 255, 210, 0.12), transparent 40%),
                  #0b1020;
      color: rgba(255,255,255,0.92);
      min-height: 100vh;
    }

    .mmds-card {
      background: var(--panel-bg);
      border: 1px solid var(--panel-border);
      border-radius: 16px;
      backdrop-filter: blur(10px);
      padding: 14px;
    }

    .mmds-kpi {
      font-size: 26px;
      font-weight: 700;
      letter-spacing: 0.2px;
      margin: 4px 0 0 0;
    }

    .mmds-sub {
      font-size: 12px;
      opacity: 0.85;
      margin-top: 6px;
    }

    .mmds-badge {
      display: inline-block;
      padding: 4px 10px;
      border-radius: 999px;
      border: 1px solid var(--panel-border);
      background: rgba(255,255,255,0.06);
      font-size: 12px;
    }
    """
