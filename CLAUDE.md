# Receipt — project context for Claude Code

> Project-specific context layered on top of the global standards in `~/.claude/CLAUDE.md`.

## What it does
Your bank app tells you how much you spent. **receipt** tells you what actually happened — semantic clustering, seven behavioral detectors, and an AI narrative that thinks like a sharp friend with an accounting degree: specific dollar amounts, named merchants, non-obvious patterns, and none of the generic budget-app moralizing.

## Stack
python

## Commands
```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest
```

## Conventions
- Conventional commits, one logical change each; secrets never hardcoded; external API calls via a service layer; errors normalized before the client.
- Python: virtualenv always, `requirements.txt` pinned, deterministic logic split from I/O.

