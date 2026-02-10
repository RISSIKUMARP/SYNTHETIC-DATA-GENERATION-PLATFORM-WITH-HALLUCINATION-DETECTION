# Week 1 - Day 1: Setup Guide

## Step 1: Create Your Project Folder

```bash
mkdir synthetic-data-project
cd synthetic-data-project
mkdir -p data/raw scripts
```

## Step 2: Create a Virtual Environment

This keeps your project libraries separate from your system Python.

```bash
# Create it
python -m venv venv

# Activate it (run this every time you open a new terminal)
# Mac/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# You should see (venv) at the start of your terminal prompt
```

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Heads up:** `torch` and `sdv` are big downloads (1-2 GB combined). Give it 10-15 minutes on a normal connection.

If you hit errors:
- **Mac M1/M2**: PyTorch should auto-detect ARM. If not: `pip install torch --index-url https://download.pytorch.org/whl/cpu`
- **Windows**: Make sure you have Visual C++ Build Tools installed
- **Linux**: Should work out of the box

## Step 4: Download the Kaggle Dataset

1. Go to: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Sign in (free account) and click **Download**
3. Unzip it — you'll get `creditcard.csv` (~150MB)
4. Move it to: `data/raw/creditcard.csv`

**Alternative (Kaggle CLI):**
```bash
pip install kaggle
# Put your kaggle.json API key in ~/.kaggle/
kaggle datasets download -d mlg-ulb/creditcardfraud
unzip creditcardfraud.zip -d data/raw/
```

## Step 5: Verify Everything

```bash
cd synthetic-data-project
python scripts/verify_setup.py
```

You should see all green checkmarks. The script will:
- Check all libraries are installed
- Confirm the dataset loaded (284,807 transactions)
- Run a tiny GAN test (trains for 1 epoch on 500 rows)
- Generate 5 fake rows to prove it works

## What Each Column in the Dataset Means

| Column | What It Is |
|--------|-----------|
| Time | Seconds since first transaction in dataset |
| V1-V28 | PCA-transformed features (anonymized for privacy) |
| Amount | Transaction amount in dollars |
| Class | 0 = legitimate, 1 = fraud |

The V1-V28 columns are already anonymized by PCA — the original bank features (merchant name, card number, etc.) were transformed so you can't reverse-engineer them. This is actually perfect for our privacy project since the raw data is already privacy-preserving.

## Folder Structure After Setup

```
synthetic-data-project/
├── data/
│   └── raw/
│       └── creditcard.csv     ← Kaggle dataset goes here
├── scripts/
│   └── verify_setup.py        ← Run this to check everything
├── requirements.txt            ← All dependencies
├── venv/                       ← Your virtual environment
└── README.md                   ← This file
```

## Troubleshooting

**"ModuleNotFoundError: No module named 'sdv'"**
→ Make sure your venv is activated. Run `which python` — it should point to your venv folder.

**"torch not found" or CUDA errors**
→ For CPU-only (fine for this project): `pip install torch --index-url https://download.pytorch.org/whl/cpu`

**SDV version conflicts**
→ Try: `pip install sdv==1.10.0 --no-deps` then install missing deps manually

**Dataset download issues**
→ The Kaggle page sometimes requires accepting terms. Log in, go to the dataset page, and click "Download" manually.
