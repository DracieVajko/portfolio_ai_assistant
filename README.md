# Portfolio AI Assistant

Reliable Investment Monitoring System with local AI models (Ollama/LM Studio).

## Features

- **T212 Integration** - Read-only portfolio sync via Trading 212 API
- **Technical Analysis** - RSI, MA crossovers, momentum, trend detection
- **Signal Generation** - BUY/SELL/HOLD with probabilities
- **Per-Asset News** - 48h filtered news with sentiment analysis
- **Local AI** - Works with Ollama or LM Studio (Fin-R1, Gemma, GPT-OSS)
- **Deterministic Reports** - Markdown + JSON output

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure
```bash
cp api.env.example api.env
# Edit api.env with your T212 API credentials
```

### 3. Configure Portfolio
Edit `portfolio_config.json` with your assets and settings.

### 4. Run

**Fast deterministic report (no AI):**
```bash
python portfolio_ai_assistant.py --no-ai
```

**Single asset:**
```bash
python portfolio_ai_assistant.py --no-ai --asset AAPL
```

**Full with AI (requires LM Studio):**
```bash
python portfolio_ai_assistant.py
```

### Batch Files (Windows)
```cmd
run_automated.bat      # Daily automated run
run_full_report.bat    # Full report with AI
run_t212_analysis.bat  # T212 analysis only
```

## Configuration

### `portfolio_config.json`
- `assets` - Your portfolio assets with symbols, groups, aliases
- `settings` - Thresholds, AI backends, news settings
- `symbol_aliases` - T212 internal ticker → Yahoo Finance mappings
- `ai_backends` - LM Studio / Ollama model configuration

### `api.env`
```env
TRADING212_API_KEY=your_key
TRADING212_API_SECRET=your_secret
TRADING212_ENABLED=true
```

## Project Structure

```
portfolio_ai_assistant/
├── portfolio_ai_assistant.py   # Main entry point
├── portfolio_config.json       # Configuration
├── requirements.txt
├── api.env                     # Secrets (gitignored)
├── .gitignore
├── trading212/                 # T212 modules
│   ├── auth.py                 # Authentication
│   ├── portfolio.py            # Portfolio analysis
│   └── integration.py          # High-level integration
├── reports/                    # Generated reports (gitignored)
├── logs/                       # Runtime logs (gitignored)
├── data/                       # Cache/data
└── run_*.bat                   # Windows batch helpers
```

## AI Models (LM Studio)

| Model | Purpose |
|-------|---------|
| `fin-r1` | Alpha analysis (reasoning) |
| `gemma-4-12b` | Summary (fast) |
| `gpt-oss-20b` | Fallback |

Configure in `ai_backends` section of config.

## Scheduled operation (V4)

Exact scheduled command (Windows Task Scheduler):

```cmd
py -3.12 portfolio_ai_assistant.py --scheduled
```

(or double-click `run_full_report.bat`, which pins Python 3.12 and passes `--scheduled`).

Expected output files per run:

```text
reports/latest/portfolio_report.md    # concise daily dashboard (read this)
reports/latest/portfolio_report.json  # canonical machine-readable result
reports/latest/data_quality.md        # mapping/provider audit (technical)
reports/latest/run_manifest.json      # run metadata: exit code, timings, files
reports/archive/YYYY-MM-DD/<run_id>/  # same four files + log snapshot
reports/portfolio_analysis.md/json    # legacy compatibility copies
```

Exit codes: `0` success · `1` fatal (no usable report) · `2` config/env error ·
`3` skipped, another run active · `4` partial (usable report + warnings) ·
`5` broker sync failed (report states broker data unavailable/stale).

Task Scheduler hardening — set **both** barriers:

1. The script takes an OS-level lock (`runtime/run.lock`); an overlapping run logs
   a warning and exits with code `3` without touching reports.
2. In Task Scheduler set: *“If the task is already running: Do not start a new instance.”*

Pre-flight checks (offline, no secrets printed):

```cmd
py -3.12 portfolio_ai_assistant.py --validate-only   :: config check, exit 2 on errors
py -3.12 portfolio_ai_assistant.py --dry-run --no-ai :: full pipeline in memory, no files
py -3.12 -m pytest tests/ -q                         :: 55 offline tests
```

Secrets: `api.env` holds real keys — never commit it, never paste logs containing
`Authorization`/`Bearer`/`key=` values (the tool redacts them automatically).

## Output (legacy layout note)

- `reports/portfolio_analysis.md` - Human-readable Markdown (compat copy of the main report)
- `reports/portfolio_analysis.json` - Machine-readable JSON (compat copy)
- `reports/archive/` - Timestamped history (new layout: `YYYY-MM-DD/<run_id>/`)
- `reports/latest/` - Always the newest complete run (published atomically)

## License

MIT