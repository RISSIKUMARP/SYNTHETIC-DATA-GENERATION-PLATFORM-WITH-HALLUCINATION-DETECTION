"""
Layer 2: Statistical Validator

Pure Python statistical tests for distribution comparison and outlier detection.
No LLM dependencies - safe to run first.

Performance target: 2 seconds per 10K records
"""

import pandas as pd
import numpy as np
import json
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.ensemble import IsolationForest
from config import config


@dataclass
class StatisticalTest:
    """Result of a single statistical test."""
    test_name: str
    feature: str
    passed: bool
    statistic: float
    threshold: float
    p_value: Optional[float] = None
    details: Optional[Dict] = None


@dataclass
class ValidationResult:
    """Results from Layer 2 statistical validation."""
    validator_name: str
    passed: bool
    tests_passed: int
    tests_failed: int
    test_results: List[StatisticalTest]
    execution_time: float
    metrics: Dict = field(default_factory=dict)
    
    @property
    def total_tests(self) -> int:
        return self.tests_passed + self.tests_failed
    
    @property
    def pass_rate(self) -> float:
        if self.total_tests == 0:
            return 0.0
        return (self.tests_passed / self.total_tests) * 100
    
    def to_dict(self) -> dict:
        return {
            'validator_name': self.validator_name,
            'passed': self.passed,
            'tests_passed': self.tests_passed,
            'tests_failed': self.tests_failed,
            'total_tests': self.total_tests,
            'pass_rate': self.pass_rate,
            'execution_time': self.execution_time,
            'metrics': self.metrics,
            'test_results': [
                {
                    'test_name': t.test_name,
                    'feature': t.feature,
                    'passed': t.passed,
                    'statistic': t.statistic,
                    'threshold': t.threshold,
                    'p_value': t.p_value
                }
                for t in self.test_results
            ]
        }
    
    def summary(self) -> str:
        lines = [
            f"\n{'='*60}",
            f"Layer 2 Statistical Validation Result",
            f"{'='*60}",
            f"Status: {'PASSED' if self.passed else 'FAILED'}",
            f"Tests passed: {self.tests_passed} / {self.total_tests} ({self.pass_rate:.2f}%)",
            f"Time: {self.execution_time:.2f}s",
        ]
        
        if self.tests_failed > 0:
            lines.append(f"\nFailed tests:")
            failed_tests = [t for t in self.test_results if not t.passed]
            for test in failed_tests[:10]:
                lines.append(
                    f"  {test.test_name} ({test.feature}): "
                    f"{test.statistic:.4f} > {test.threshold:.4f}"
                )
        
        lines.append(f"{'='*60}\n")
        return '\n'.join(lines)


class DistributionComparator:
    """Compare distributions between real and synthetic data."""
    
    def __init__(self, ks_threshold: float = 0.05):
        self.ks_threshold = ks_threshold
    
    def ks_test(
        self,
        real_data: pd.Series,
        synthetic_data: pd.Series,
        feature_name: str
    ) -> StatisticalTest:
        """
        Kolmogorov-Smirnov test for distribution similarity.
        
        H0: Distributions are the same
        If p-value > 0.05 and KS < threshold, distributions match
        """
        ks_statistic, p_value = stats.ks_2samp(real_data, synthetic_data)
        
        passed = ks_statistic < self.ks_threshold
        
        return StatisticalTest(
            test_name="KS_test",
            feature=feature_name,
            passed=passed,
            statistic=ks_statistic,
            threshold=self.ks_threshold,
            p_value=p_value,
            details={
                'real_mean': real_data.mean(),
                'synthetic_mean': synthetic_data.mean(),
                'real_std': real_data.std(),
                'synthetic_std': synthetic_data.std()
            }
        )
    
    def js_divergence(
        self,
        real_data: pd.Series,
        synthetic_data: pd.Series,
        feature_name: str,
        n_bins: int = 50
    ) -> StatisticalTest:
        """
        Jensen-Shannon divergence for distribution comparison.
        
        Range: [0, 1], where 0 = identical distributions
        Threshold: 0.1 (good similarity)
        """
        real_hist, bins = np.histogram(real_data, bins=n_bins, density=True)
        synthetic_hist, _ = np.histogram(synthetic_data, bins=bins, density=True)
        
        real_hist = real_hist + 1e-10
        synthetic_hist = synthetic_hist + 1e-10
        
        real_hist = real_hist / real_hist.sum()
        synthetic_hist = synthetic_hist / synthetic_hist.sum()
        
        js_div = jensenshannon(real_hist, synthetic_hist)
        
        threshold = 0.1
        passed = js_div < threshold
        
        return StatisticalTest(
            test_name="JS_divergence",
            feature=feature_name,
            passed=passed,
            statistic=js_div,
            threshold=threshold
        )
    
    def moment_comparison(
        self,
        real_data: pd.Series,
        synthetic_data: pd.Series,
        feature_name: str
    ) -> List[StatisticalTest]:
        """
        Compare statistical moments (mean, std, skewness, kurtosis).
        
        Threshold: Relative difference < 10%
        """
        tests = []
        
        real_mean = real_data.mean()
        synthetic_mean = synthetic_data.mean()
        mean_diff = abs(real_mean - synthetic_mean) / (abs(real_mean) + 1e-10)
        
        tests.append(StatisticalTest(
            test_name="mean_comparison",
            feature=feature_name,
            passed=mean_diff < 0.1,
            statistic=mean_diff,
            threshold=0.1,
            details={'real_mean': real_mean, 'synthetic_mean': synthetic_mean}
        ))
        
        real_std = real_data.std()
        synthetic_std = synthetic_data.std()
        std_diff = abs(real_std - synthetic_std) / (real_std + 1e-10)
        
        tests.append(StatisticalTest(
            test_name="std_comparison",
            feature=feature_name,
            passed=std_diff < 0.1,
            statistic=std_diff,
            threshold=0.1,
            details={'real_std': real_std, 'synthetic_std': synthetic_std}
        ))
        
        return tests


class CorrelationValidator:
    """Validate correlation structure preservation."""
    
    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold
    
    def correlation_frobenius_norm(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        features: List[str]
    ) -> StatisticalTest:
        """
        Compare correlation matrices using Frobenius norm.
        
        Lower is better, threshold < 0.1 indicates good preservation
        """
        real_corr = real_data[features].corr()
        synthetic_corr = synthetic_data[features].corr()
        
        diff_matrix = real_corr - synthetic_corr
        frobenius_norm = np.linalg.norm(diff_matrix, 'fro')
        
        normalized_norm = frobenius_norm / np.sqrt(len(features))
        
        passed = normalized_norm < self.threshold
        
        return StatisticalTest(
            test_name="correlation_frobenius",
            feature="all_features",
            passed=passed,
            statistic=normalized_norm,
            threshold=self.threshold,
            details={
                'n_features': len(features),
                'raw_frobenius': frobenius_norm
            }
        )
    
    def pairwise_correlation_check(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        features: List[str],
        top_n: int = 10
    ) -> List[StatisticalTest]:
        """
        Check top N correlated pairs are preserved.
        
        Identifies pairs with highest correlation in real data,
        then checks if synthetic data maintains similar correlation.
        """
        real_corr = real_data[features].corr()
        synthetic_corr = synthetic_data[features].corr()
        
        mask = np.triu(np.ones_like(real_corr, dtype=bool), k=1)
        real_corr_values = real_corr.where(mask)
        
        top_pairs = []
        for i in range(len(features)):
            for j in range(i+1, len(features)):
                if pd.notna(real_corr_values.iloc[i, j]):
                    top_pairs.append((
                        features[i],
                        features[j],
                        abs(real_corr_values.iloc[i, j])
                    ))
        
        top_pairs.sort(key=lambda x: x[2], reverse=True)
        top_pairs = top_pairs[:top_n]
        
        tests = []
        for feat1, feat2, real_corr_val in top_pairs:
            synthetic_corr_val = abs(synthetic_corr.loc[feat1, feat2])
            diff = abs(real_corr_val - synthetic_corr_val)
            
            tests.append(StatisticalTest(
                test_name="pairwise_correlation",
                feature=f"{feat1}_vs_{feat2}",
                passed=diff < 0.1,
                statistic=diff,
                threshold=0.1,
                details={
                    'real_correlation': real_corr_val,
                    'synthetic_correlation': synthetic_corr_val
                }
            ))
        
        return tests


class OutlierDetector:
    """Detect outliers in synthetic data."""
    
    def __init__(self, zscore_threshold: float = 3.0):
        self.zscore_threshold = zscore_threshold
    
    def zscore_outliers(
        self,
        data: pd.DataFrame,
        feature: str
    ) -> StatisticalTest:
        """
        Detect outliers using z-score method.
        
        Threshold: < 1% of records should be extreme outliers (|z| > 3)
        """
        z_scores = np.abs(stats.zscore(data[feature]))
        outlier_count = np.sum(z_scores > self.zscore_threshold)
        outlier_rate = outlier_count / len(data)
        
        threshold = 0.01
        passed = outlier_rate < threshold
        
        return StatisticalTest(
            test_name="zscore_outliers",
            feature=feature,
            passed=passed,
            statistic=outlier_rate,
            threshold=threshold,
            details={
                'outlier_count': int(outlier_count),
                'total_records': len(data)
            }
        )
    
    def isolation_forest_outliers(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        features: List[str]
    ) -> StatisticalTest:
        """
        Use Isolation Forest for multivariate outlier detection.
        
        Train on real data, test on synthetic data.
        Threshold: < 5% of synthetic records flagged as outliers
        """
        clf = IsolationForest(contamination=0.01, random_state=42)
        clf.fit(real_data[features])
        
        predictions = clf.predict(synthetic_data[features])
        outlier_count = np.sum(predictions == -1)
        outlier_rate = outlier_count / len(synthetic_data)
        
        threshold = 0.05
        passed = outlier_rate < threshold
        
        return StatisticalTest(
            test_name="isolation_forest",
            feature="multivariate",
            passed=passed,
            statistic=outlier_rate,
            threshold=threshold,
            details={
                'outlier_count': int(outlier_count),
                'total_records': len(synthetic_data)
            }
        )


class Layer2StatisticalValidator:
    """
    Layer 2: Statistical validation using distribution tests.
    
    Performance target: 2 seconds per 10K records
    """
    
    def __init__(
        self,
        real_data_path: Optional[str] = None,
        ks_threshold: Optional[float] = None,
        correlation_threshold: Optional[float] = None
    ):
        self.real_data_path = real_data_path or config.REAL_DATA_PATH
        self.ks_threshold = ks_threshold or config.LAYER2_KS_THRESHOLD
        self.correlation_threshold = correlation_threshold or config.LAYER2_CORRELATION_THRESHOLD
        
        print(f"\nLoading real data for statistical comparison...")
        self.real_data = pd.read_csv(self.real_data_path)
        print(f"Loaded {len(self.real_data)} real transactions")
        
        self.distribution_comparator = DistributionComparator(self.ks_threshold)
        self.correlation_validator = CorrelationValidator(self.correlation_threshold)
        self.outlier_detector = OutlierDetector(config.LAYER2_OUTLIER_ZSCORE_THRESHOLD)
    
    def validate(self, data: pd.DataFrame) -> ValidationResult:
        print("\n" + "="*60)
        print("Layer 2: Statistical Validation")
        print("="*60)
        
        start_time = time.time()
        
        all_tests = []
        
        print("\n1. Distribution Tests (KS test for each feature)")
        all_features = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)] + ['Class']
        
        for feature in all_features:
            if feature in data.columns and feature in self.real_data.columns:
                ks_test = self.distribution_comparator.ks_test(
                    self.real_data[feature],
                    data[feature],
                    feature
                )
                all_tests.append(ks_test)
        
        print(f"  Completed {len(all_features)} KS tests")
        
        print("\n2. Moment Comparison (mean, std for key features)")
        key_features = ['Time', 'Amount', 'V3', 'V14', 'V17', 'V12', 'V10']
        for feature in key_features:
            if feature in data.columns:
                moment_tests = self.distribution_comparator.moment_comparison(
                    self.real_data[feature],
                    data[feature],
                    feature
                )
                all_tests.extend(moment_tests)
        
        print(f"  Completed moment comparison for {len(key_features)} features")
        
        print("\n3. Correlation Structure Validation")
        v_features = [f'V{i}' for i in range(1, 29)]
        
        frobenius_test = self.correlation_validator.correlation_frobenius_norm(
            self.real_data,
            data,
            v_features
        )
        all_tests.append(frobenius_test)
        
        pairwise_tests = self.correlation_validator.pairwise_correlation_check(
            self.real_data,
            data,
            v_features,
            top_n=10
        )
        all_tests.extend(pairwise_tests)
        
        print(f"  Completed correlation validation")
        
        print("\n4. Outlier Detection")
        for feature in ['Amount', 'V14', 'V10', 'V3']:
            if feature in data.columns:
                outlier_test = self.outlier_detector.zscore_outliers(data, feature)
                all_tests.append(outlier_test)
        
        isolation_test = self.outlier_detector.isolation_forest_outliers(
            self.real_data,
            data,
            v_features
        )
        all_tests.append(isolation_test)
        
        print(f"  Completed outlier detection")
        
        execution_time = time.time() - start_time
        
        tests_passed = sum(1 for t in all_tests if t.passed)
        tests_failed = len(all_tests) - tests_passed
        
        overall_passed = tests_failed == 0
        
        result = ValidationResult(
            validator_name="Layer2_StatisticalValidator",
            passed=overall_passed,
            tests_passed=tests_passed,
            tests_failed=tests_failed,
            test_results=all_tests,
            execution_time=execution_time,
            metrics={
                'ks_tests_passed': sum(1 for t in all_tests if t.test_name == 'KS_test' and t.passed),
                'ks_tests_total': sum(1 for t in all_tests if t.test_name == 'KS_test'),
                'correlation_preserved': frobenius_test.passed,
                'outlier_rate_acceptable': isolation_test.passed
            }
        )
        
        print(result.summary())
        
        return result
    
    def save_report(self, result: ValidationResult, output_path: Optional[str] = None):
        if output_path is None:
            output_path = f"{config.OUTPUT_DIR}/layer2_report.json"
        
        report = result.to_dict()
        
        failed_tests = [t for t in result.test_results if not t.passed]
        report['failed_tests_detail'] = [
            {
                'test': t.test_name,
                'feature': t.feature,
                'statistic': t.statistic,
                'threshold': t.threshold,
                'details': t.details
            }
            for t in failed_tests
        ]
        
        ks_tests = [t for t in result.test_results if t.test_name == 'KS_test']
        ks_passed = [t for t in ks_tests if t.passed]
        report['ks_summary'] = {
            'total': len(ks_tests),
            'passed': len(ks_passed),
            'pass_rate': len(ks_passed) / len(ks_tests) * 100 if ks_tests else 0,
            'best_features': [
                {'feature': t.feature, 'ks_statistic': t.statistic}
                for t in sorted(ks_tests, key=lambda x: x.statistic)[:5]
            ],
            'worst_features': [
                {'feature': t.feature, 'ks_statistic': t.statistic}
                for t in sorted(ks_tests, key=lambda x: x.statistic, reverse=True)[:5]
            ]
        }
        
        # Convert numpy types to Python types for JSON
        import numpy as np

        def convert_numpy(obj):
            if isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            return obj

        report = convert_numpy(report)

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nLayer 2 report saved to: {output_path}")


if __name__ == "__main__":
    """Example usage of Layer 2 validator."""
    
    print("="*80)
    print("LAYER 2 STATISTICAL VALIDATOR - Pure Python")
    print("="*80)
    
    from config import validate_config
    if not validate_config():
        print("\nPlease fix configuration errors before running.")
        exit(1)
    
    validator = Layer2StatisticalValidator(
        real_data_path=config.REAL_DATA_PATH,
        ks_threshold=config.LAYER2_KS_THRESHOLD
    )
    
    print(f"\nLoading synthetic data from {config.SYNTHETIC_DATA_PATH}...")
    synthetic_data = pd.read_csv(config.SYNTHETIC_DATA_PATH)
    print(f"Loaded {len(synthetic_data)} synthetic transactions")
    
    result = validator.validate(synthetic_data)
    
    validator.save_report(result)
    
    print("\n" + "="*80)
    print("LAYER 2 VALIDATION COMPLETE")
    print("="*80)
    print(f"Status: {'PASSED' if result.passed else 'FAILED'}")
    print(f"Pass rate: {result.pass_rate:.2f}%")
    print(f"KS tests passed: {result.metrics['ks_tests_passed']}/{result.metrics['ks_tests_total']}")
    print("="*80)
