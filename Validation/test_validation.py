"""
Testing script for validation layers.

Run layers incrementally with small samples first.

Usage:
    python test_validation.py --layer1 --sample 1000
    python test_validation.py --layer2 --full
    python test_validation.py --both --sample 10000
"""

import argparse
import pandas as pd
import json
from pathlib import Path

try:
    from Validation.layer1_rule_validator import Layer1RuleValidator
    from Validation.layer2_statistical import Layer2StatisticalValidator
    from Validation.config import config, validate_config
except ImportError:
    from layer1_rule_validator import Layer1RuleValidator
    from layer2_statistical import Layer2StatisticalValidator
    from config import config, validate_config


def test_layer1(data: pd.DataFrame, enable_gemini: bool = True):
    """Test Layer 1 rule-based validator."""

    print("TESTING LAYER 1: RULE-BASED VALIDATOR")
  
    
    validator = Layer1RuleValidator(
        enable_gemini_rules=enable_gemini,
        gemini_api_key=config.GEMINI_API_KEY if enable_gemini else None,
        real_data_path=config.REAL_DATA_PATH
    )
    
    result = validator.validate(data)
    validator.save_report(result)
    
    print("\n" + "="*80)
    print("LAYER 1 RESULTS")
    print("="*80)
    print(f"Status: {'PASSED' if result.passed else 'FAILED'}")
    print(f"Valid: {result.valid_count} / {result.total_count}")
    print(f"Pass rate: {result.pass_rate:.2f}%")
    print(f"Throughput: {result.metrics['records_per_second']:.0f} rec/s")
    
    if result.failures:
        print(f"\nTop 5 failed rules:")
        sorted_failures = sorted(
            result.failures.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )[:5]
        for rule_name, failures in sorted_failures:
            print(f"  {rule_name}: {len(failures)} failures")
    
    print("="*80)
    
    return result


def test_layer2(data: pd.DataFrame):
    """Test Layer 2 statistical validator."""
    print("\n" + "="*80)
    print("TESTING LAYER 2: STATISTICAL VALIDATOR")
    print("="*80)
    
    validator = Layer2StatisticalValidator(
        real_data_path=config.REAL_DATA_PATH
    )
    
    result = validator.validate(data)
    validator.save_report(result)
    
    print("\n" + "="*80)
    print("LAYER 2 RESULTS")
    print("="*80)
    print(f"Status: {'PASSED' if result.passed else 'FAILED'}")
    print(f"Tests passed: {result.tests_passed} / {result.total_tests}")
    print(f"Pass rate: {result.pass_rate:.2f}%")
    print(f"KS tests: {result.metrics['ks_tests_passed']}/{result.metrics['ks_tests_total']}")
    print(f"Correlation preserved: {result.metrics['correlation_preserved']}")
    
    failed_tests = [t for t in result.test_results if not t.passed]
    if failed_tests:
        print(f"\nFailed tests: {len(failed_tests)}")
        for test in failed_tests[:5]:
            print(f"  {test.test_name} ({test.feature}): {test.statistic:.4f}")
    
    print("="*80)
    
    return result


def analyze_results(l1_result=None, l2_result=None):
    """Analyze and compare results."""
    print("\n" + "="*80)
    print("COMBINED ANALYSIS")
    print("="*80)
    
    if l1_result:
        print("\nLayer 1 (Rule-Based):")
        print(f"  Pass rate: {l1_result.pass_rate:.2f}%")
        print(f"  Invalid records: {l1_result.invalid_count}")
        print(f"  Total rules: {l1_result.metrics['total_rules']}")
    
    if l2_result:
        print("\nLayer 2 (Statistical):")
        print(f"  Test pass rate: {l2_result.pass_rate:.2f}%")
        print(f"  KS tests: {l2_result.metrics['ks_tests_passed']}/31")
        print(f"  Correlation: {'✓' if l2_result.metrics['correlation_preserved'] else '✗'}")
    
    print("\nOverall Assessment:")
    if l1_result and l2_result:
        if l1_result.pass_rate >= 90 and l2_result.metrics['ks_tests_passed'] >= 26:
            print("  ✓ EXCELLENT - Ready for Layer 3")
            quality = "excellent"
        elif l1_result.pass_rate >= 85 and l2_result.metrics['ks_tests_passed'] >= 23:
            print("  ✓ GOOD - Minor issues, can proceed")
            quality = "good"
        elif l1_result.pass_rate >= 80:
            print("  ⚠ MODERATE - Review failures")
            quality = "moderate"
        else:
            print("  ✗ POOR - Investigate CTGAN quality")
            quality = "poor"
    elif l1_result:
        quality = "excellent" if l1_result.pass_rate >= 90 else "moderate"
    elif l2_result:
        quality = "excellent" if l2_result.metrics['ks_tests_passed'] >= 26 else "moderate"
    else:
        quality = "unknown"
    
    print("\nRecommendations:")
    if quality == "excellent":
        print("  → Build Layer 3 (Semantic Validator)")
        print("  → Test on 100 records first (~$0.001)")
    elif quality == "good":
        print("  → Review top failed rules")
        print("  → Can proceed to Layer 3 cautiously")
    elif quality == "moderate":
        print("  → Tune rule thresholds")
        print("  → Review KS statistics")
    else:
        print("  → Check CTGAN training quality")
        print("  → Verify fraud rate preservation")
        print("  → Consider retraining")
    
    print("\nGenerated reports:")
    if l1_result:
        print(f"  - {config.OUTPUT_DIR}/layer1_report.json")
    if l2_result:
        print(f"  - {config.OUTPUT_DIR}/layer2_report.json")
    
    print("="*80)


def main():
    """Main testing function."""
    parser = argparse.ArgumentParser(
        description="Test validation layers incrementally"
    )
    parser.add_argument(
        "--layer1",
        action="store_true",
        help="Test Layer 1 (Rule-Based Validator)"
    )
    parser.add_argument(
        "--layer2",
        action="store_true",
        help="Test Layer 2 (Statistical Validator)"
    )
    parser.add_argument(
        "--both",
        action="store_true",
        help="Test both layers"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Sample size for testing (default: full dataset)"
    )
    parser.add_argument(
        "--no-gemini",
        action="store_true",
        help="Disable Gemini rule discovery (Layer 1 only)"
    )
    
    args = parser.parse_args()
    
    # Validate configuration
    print("="*80)
    print("VALIDATION SYSTEM - TESTING SCRIPT")
    print("="*80)
    
    if not validate_config():
        print("\n✗ Configuration errors detected. Please fix before continuing.")
        return
    
    # Load data
    print(f"\nLoading synthetic data from {config.SYNTHETIC_DATA_PATH}...")
    data = pd.read_csv(config.SYNTHETIC_DATA_PATH)
    print(f"Loaded {len(data)} records")
    
    # Sample if requested
    if args.sample:
        print(f"\nSampling {args.sample} records for testing...")
        data = data.sample(n=min(args.sample, len(data)), random_state=42)
        print(f"Using {len(data)} records")
    
    # Determine what to test
    test_l1 = args.layer1 or args.both
    test_l2 = args.layer2 or args.both
    
    if not test_l1 and not test_l2:
        print("\n✗ Please specify --layer1, --layer2, or --both")
        return
    
    # Run tests
    l1_result = None
    l2_result = None
    
    if test_l2:
        # Test Layer 2 first (no API risk)
        l2_result = test_layer2(data)
    
    if test_l1:
        # Test Layer 1 with optional Gemini
        enable_gemini = not args.no_gemini
        if enable_gemini:
            print("\n⚠ Layer 1 will use Gemini API for rule discovery")
            print("  Estimated cost: ~$0.001")
            response = input("  Continue? (y/n): ")
            if response.lower() != 'y':
                print("Skipping Layer 1")
                test_l1 = False
        
        if test_l1:
            l1_result = test_layer1(data, enable_gemini=enable_gemini)
    
    # Analyze combined results
    if l1_result or l2_result:
        analyze_results(l1_result, l2_result)


if __name__ == "__main__":
    main()
