# humanorbot

A simple NLP project that guesses whether a piece of text was written by a **human** or a **bot/AI**.

It turns text into TF-IDF features (word and word-pair frequencies) and trains two classic classifiers, **Naive Bayes** and **Logistic Regression**. It then compares them and saves the better one.

## Project structure

```
humanorbot/
├── data/
│   └── sample.csv      # 100 short demo sentences (label 0 = human, 1 = bot)
├── src/
│   ├── preprocess.py   # loads the CSV and cleans the text
│   ├── train.py        # trains + evaluates both models, saves the best
│   └── predict.py      # classifies new text with the saved model
├── models/             # trained model is saved here (not committed)
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

## How to run

Train (prints accuracy, precision/recall and a confusion matrix for each model):

```bash
python src/train.py
```

Classify your own text:

```bash
python src/predict.py "bro i forgot the assignment lol"
# HUMAN (64% confident)
python src/predict.py "In conclusion, it is important to consider multiple perspectives."
# BOT (72% confident)
```

Run `python src/predict.py` with no text to type sentences one by one.

## Using a bigger dataset

`data/sample.csv` is a tiny demo set so the code runs straight away. Its results (about 90% accuracy) do not mean much. For real results, download a larger human-vs-AI text dataset, for example from Kaggle or Hugging Face. Save it as a CSV with a `text` column and a `label` column (0 = human, 1 = bot), put it in `data/raw/`, and run:

```bash
python src/train.py --data data/raw/your_dataset.csv
```

Files in `data/raw/` are not uploaded to GitHub because they are usually too large. Share them with the team through Google Drive instead.

## Team workflow (everyone commits to `main`)

Do this once on your computer, so your commits show your GitHub username:

```bash
git config --global user.name "YourGitHubUsername"
git config --global user.email "the email on your GitHub account"
```

Then, every time you work:

```bash
git pull                      # 1. get your teammate's latest changes first
# ...edit code...
git add .
git commit -m "Short description of what you changed"
git push                      # 2. upload to main
```

If `git push` is rejected, someone pushed before you. Run `git pull`, fix any conflicts, then push again. You can see who changed what under **Commits** on GitHub.

## Ideas to extend it

- Add hand-made features such as average sentence length, punctuation count or vocabulary richness.
- Try character n-grams: `TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))`.
- Fine-tune a small transformer model (e.g. DistilBERT) and compare it with these baselines.

## Team

- Abdullah Ahmad ([@abdullahahmad0629-glitch](https://github.com/abdullahahmad0629-glitch))
- Eshaal Yasir ([@EshaalYasir123](https://github.com/EshaalYasir123))
