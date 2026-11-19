# humanorbot

NLP project: classify a piece of text as **human-written** or **bot/AI-generated**.

## Project structure

```
humanorbot/
├── data/
│   ├── raw/          # original datasets (not committed)
│   └── processed/    # cleaned / split data (not committed)
├── notebooks/        # exploration and experiments
├── src/humanorbot/   # reusable code: preprocessing, features, models
├── models/           # saved model checkpoints (not committed)
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/abdullahahmad0629-glitch/humanorbot.git
cd humanorbot
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Team workflow

1. Pull the latest `main` before starting: `git pull`
2. Create a branch for your work: `git checkout -b your-name/feature`
3. Commit small, clear changes and push: `git push -u origin your-name/feature`
4. Open a pull request on GitHub and ask a teammate to review it.

Large datasets and model files stay out of git (see `.gitignore`); share them via Drive or similar and note the link here.

## Team

- _add names here_
