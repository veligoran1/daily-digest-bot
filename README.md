# 📊 Daily Digest Bot

Automated daily digest of AI/ML/DS articles delivered to Telegram.

## Sources

- **Habr** — AI/ML, mathematics, physics, Python
- **HackerNoon** — science, machine learning, data science, programming
- **arXiv** — research papers (ML, AI, CV, Optimization, Data Science)
- **Kaggle** — active competitions (top 5 by team count)

## How It Works

1. Scrapes yesterday's articles from all sources
2. Fetches full article texts
3. Generates summaries in Russian via Groq API (Llama 3.1)
4. Sends the digest to Telegram

## Manual Run

```bash
pip install -r requirements.txt

export GROQ_API_KEY="..."
export TELEGRAM_TOKEN="..."
export TELEGRAM_CHAT_ID="..."
export KAGGLE_API_TOKEN="..."

python3 daily_digest.py --days-back 1
```

## Automation

Runs daily at 09:00 MSK via GitHub Actions.

Secrets are configured in Settings → Secrets → Actions:
- `GROQ_API_KEY`
- `TELEGRAM_TOKEN`
- `TELEGRAM_CHAT_ID`
- `KAGGLE_API_TOKEN`

## Tech Stack

- Python 3.11
- Groq API (Llama 3.1 8B)
- arXiv API
- Kaggle API
- python-telegram-bot
- BeautifulSoup4
- GitHub Actions
