#!/usr/bin/env bash
set -e

echo "=== Synthetic Data Validation Platform ==="
echo ""

# Install dependencies
echo "[1/3] Installing dependencies..."
pip install -r requirements.txt --quiet

# Generate demo data (so Tab 1 has something to show immediately)
echo "[2/3] Generating demo data..."
python demo_data.py --n 500

# Launch Streamlit
echo "[3/3] Launching dashboard..."
echo "       Open http://localhost:8501 in your browser"
echo ""
streamlit run app.py
