# Contributing to Atlas

Thanks for your interest! Atlas is a solo project in early alpha, so contributions are welcome — just keep in mind things may move fast and break often.

## Before You Start

It's worth opening an issue before working on anything significant, just so we're not duplicating effort or heading in different directions.

## Setup

1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Optional extras: `pip install sentence-transformers PySide6 torch`
4. Run Atlas: `python Atlas.py`

## Ways to Contribute

- **Bug reports** — open a GitHub issue with steps to reproduce, your OS, and Python version
- **Bug fixes** — PRs welcome, keep them focused on one thing
- **Feature ideas** — open an issue first to discuss before implementing
- **Docs** — if something is unclear or missing, feel free to improve it

## Code Style

Nothing too strict, but please:
- Keep it readable
- Match the existing style roughly
- Add a comment if you're doing something non-obvious

## Pull Requests

- Branch off `main`
- Keep PRs small and focused where possible
- Describe what you changed and why

## What's Out of Scope (for now)

- Cloud/API-based model support — Atlas is intentionally local-first
- Major architectural changes without prior discussion

That's about it. Don't overthink it — if you've found a bug or have a useful fix, just go for it.
