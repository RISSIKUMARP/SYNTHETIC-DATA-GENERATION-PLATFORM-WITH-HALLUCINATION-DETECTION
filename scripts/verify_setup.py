"""
============================================================
Week 1 - Day 1: Environment Setup Verification
============================================================
Run this AFTER installing requirements.txt and downloading
the Kaggle dataset to data/raw/creditcard.csv

Usage:
    python scripts/verify_setup.py
============================================================
"""

import sys
import importlib


def check_library(name, import_name=None):
    """Try importing a library and report version."""
    import_name = import_name or name
    try:
        mod = importlib.import_module(import_name)
        version = getattr(mod, "__version__", "installed (no version attr)")
        print(f"  ✅ {name}: {version}")
        return True
    except ImportError:
        print(f"  ❌ {name}: NOT INSTALLED")
        return False


def main():
    print("=" * 60)
    print("SETUP VERIFICATION - Week 1 Day 1")
    print("=" * 60)

    # ---- Step 1: Python version ----
    print(f"\n📌 Python Version: {sys.version}")
    if sys.version_info < (3, 9):
        print("  ⚠️  Python 3.9+ recommended. You're on an older version.")

    # ---- Step 2: Core libraries ----
    print("\n📦 Checking Core Libraries...")
    all_ok = True

    # Must-haves for Week 1
    week1_libs = [
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("sdv", "sdv"),
        ("torch (PyTorch)", "torch"),
        ("scipy", "scipy"),
        ("scikit-learn", "sklearn"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
    ]

    # Needed later but good to have now
    later_libs = [
        ("openai", "openai"),
        ("pinecone-client", "pinecone"),
        ("fastapi", "fastapi"),
        ("streamlit", "streamlit"),
    ]

    print("\n  -- Week 1 essentials --")
    for display_name, import_name in week1_libs:
        if not check_library(display_name, import_name):
            all_ok = False

    print("\n  -- Needed later (good to install now) --")
    for display_name, import_name in later_libs:
        check_library(display_name, import_name)

    # ---- Step 3: CTGAN specifically ----
    print("\n🔧 Checking CTGAN (the GAN model we'll use)...")
    try:
        from sdv.single_table import CTGANSynthesizer
        print("  ✅ CTGANSynthesizer imported successfully")
    except ImportError as e:
        print(f"  ❌ CTGANSynthesizer import failed: {e}")
        all_ok = False

    # ---- Step 4: Dataset check ----
    print("\n📊 Checking Kaggle Dataset...")
    import os

    dataset_paths = [
        "data/raw/creditcard.csv",
        "../data/raw/creditcard.csv",
    ]

    dataset_found = False
    for path in dataset_paths:
        if os.path.exists(path):
            import pandas as pd

            df = pd.read_csv(path)
            print(f"  ✅ Dataset found at: {path}")
            print(f"  ✅ Rows: {len(df):,}")
            print(f"  ✅ Columns: {df.shape[1]}")
            print(f"  ✅ Columns list: {list(df.columns)}")
            print(f"  ✅ Fraud cases: {df['Class'].sum():,} "
                  f"({df['Class'].mean()*100:.2f}%)")
            print(f"  ✅ Amount range: ${df['Amount'].min():.2f} - "
                  f"${df['Amount'].max():.2f}")
            dataset_found = True
            break

    if not dataset_found:
        print("  ❌ Dataset NOT found!")
        print("     Download from: https://www.kaggle.com/mlg-ulb/creditcardfraud")
        print("     Save as: data/raw/creditcard.csv")
        all_ok = False

    # ---- Step 5: Quick GAN smoke test ----
    if dataset_found:
        print("\n🧪 Quick GAN Smoke Test (generating 5 rows)...")
        try:
            import pandas as pd
            from sdv.single_table import CTGANSynthesizer
            from sdv.metadata import SingleTableMetadata

            # Load a tiny sample
            df_sample = pd.read_csv(path).head(500)

            # SDV needs metadata about the table
            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(df_sample)

            # Create a tiny CTGAN (1 epoch just to test it runs)
            print("  ⏳ Training CTGAN on 500 rows, 1 epoch (just a test)...")
            synth = CTGANSynthesizer(
                metadata,
                epochs=1,
                verbose=False
            )
            synth.fit(df_sample)

            # Generate 5 fake rows
            fake = synth.sample(num_rows=5)
            print(f"  ✅ Generated {len(fake)} synthetic rows!")
            print(f"  ✅ Columns match: {list(fake.columns) == list(df_sample.columns)}")
            print("\n  Sample synthetic row:")
            print(f"  {fake.iloc[0].to_dict()}")

        except Exception as e:
            print(f"  ❌ GAN smoke test failed: {e}")
            print("  This might be a PyTorch/CUDA issue. We can debug this.")
            all_ok = False

    # ---- Summary ----
    print("\n" + "=" * 60)
    if all_ok:
        print("🎉 ALL CHECKS PASSED - You're ready for Week 1!")
        print("   Next step: Run the EDA notebook (Day 2)")
    else:
        print("⚠️  SOME CHECKS FAILED - Fix the issues above first")
        print("   Need help? Share the error messages and we'll sort it out.")
    print("=" * 60)


if __name__ == "__main__":
    main()
