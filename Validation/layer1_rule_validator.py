"""
Layer 1: Rule-Based Validator with Gemini Rule Discovery

Fast, deterministic validation using hardcoded rules + Gemini-discovered rules.

Performance target: 1000 records/second (100ms per record)
"""

import pandas as pd
import numpy as np
import json
import time
from typing import List, Dict, Any, Tuple, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import google.genai as genai
try:
    from Validation.config import config
except ImportError:
    from config import config


class Severity(Enum):
    """Severity levels for validation failures."""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationFailure:
    """Represents a single validation failure."""
    row_index: int
    rule_name: str
    rule_description: str
    actual_value: Any
    expected_constraint: str
    severity: Severity
    additional_info: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> dict:
        return {
            'row_index': self.row_index,
            'rule_name': self.rule_name,
            'rule_description': self.rule_description,
            'actual_value': str(self.actual_value),
            'expected_constraint': self.expected_constraint,
            'severity': self.severity.value,
            'additional_info': self.additional_info or {}
        }


@dataclass
class ValidationResult:
    """Results from running validation."""
    validator_name: str
    passed: bool
    valid_count: int
    invalid_count: int
    invalid_indices: List[int]
    failures: Dict[str, List[ValidationFailure]]
    execution_time: float
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def total_count(self) -> int:
        return self.valid_count + self.invalid_count
    
    @property
    def pass_rate(self) -> float:
        if self.total_count == 0:
            return 0.0
        return (self.valid_count / self.total_count) * 100
    
    def to_dict(self) -> dict:
        return {
            'validator_name': self.validator_name,
            'passed': self.passed,
            'valid_count': self.valid_count,
            'invalid_count': self.invalid_count,
            'total_count': self.total_count,
            'pass_rate': self.pass_rate,
            'invalid_indices': self.invalid_indices,
            'execution_time': self.execution_time,
            'metrics': self.metrics,
            'failures_by_rule': {
                rule: len(failures) for rule, failures in self.failures.items()
            }
        }
    
    def summary(self) -> str:
        lines = [
            f"\n{'='*60}",
            f"Layer 1 Validation Result",
            f"{'='*60}",
            f"Status: {'PASSED' if self.passed else 'FAILED'}",
            f"Valid: {self.valid_count} / {self.total_count} ({self.pass_rate:.2f}%)",
            f"Invalid: {self.invalid_count}",
            f"Time: {self.execution_time:.2f}s",
            f"Throughput: {self.metrics.get('records_per_second', 0):.0f} rec/s",
        ]
        
        if self.invalid_count > 0:
            lines.append(f"\nTop failed rules:")
            sorted_rules = sorted(
                self.failures.items(),
                key=lambda x: len(x[1]),
                reverse=True
            )[:5]
            for rule_name, failures in sorted_rules:
                lines.append(f"  {rule_name}: {len(failures)} failures")
        
        lines.append(f"{'='*60}\n")
        return '\n'.join(lines)


@dataclass
class ValidationRule:
    """Represents a validation rule."""
    name: str
    description: str
    check_function: Callable[[pd.Series], Tuple[bool, Optional[str]]]
    severity: Severity
    enabled: bool = True
    
    def validate_row(self, row: pd.Series, row_index: int) -> Optional[ValidationFailure]:
        if not self.enabled:
            return None
        
        try:
            is_valid, error_message = self.check_function(row)
            
            if not is_valid:
                return ValidationFailure(
                    row_index=row_index,
                    rule_name=self.name,
                    rule_description=self.description,
                    actual_value=error_message.split(':')[-1].strip() if error_message else "N/A",
                    expected_constraint=self.description,
                    severity=self.severity
                )
        except Exception as e:
            return ValidationFailure(
                row_index=row_index,
                rule_name=self.name,
                rule_description=self.description,
                actual_value=f"Error: {str(e)}",
                expected_constraint=self.description,
                severity=Severity.CRITICAL,
                additional_info={'error': str(e)}
            )
        
        return None


class StaticRuleEngine:
    """Hardcoded business rules for credit card transactions."""
    
    @staticmethod
    def get_static_rules() -> List[ValidationRule]:
        rules = []
        
        # Critical rules: Amount
        rules.append(ValidationRule(
            name="amount_non_negative",
            description="Transaction amount must be >= 0",
            check_function=lambda row: (
                row['Amount'] >= 0,
                f"Amount is negative: {row['Amount']}"
            ),
            severity=Severity.CRITICAL
        ))
        
        rules.append(ValidationRule(
            name="amount_reasonable",
            description="Transaction amount must be <= $50,000",
            check_function=lambda row: (
                row['Amount'] <= 50000,
                f"Amount exceeds maximum: {row['Amount']}"
            ),
            severity=Severity.WARNING
        ))
        
        # Critical rules: Time
        rules.append(ValidationRule(
            name="time_non_negative",
            description="Time must be >= 0",
            check_function=lambda row: (
                row['Time'] >= 0,
                f"Time is negative: {row['Time']}"
            ),
            severity=Severity.CRITICAL
        ))
        
        rules.append(ValidationRule(
            name="time_in_range",
            description="Time must be <= 172792 seconds",
            check_function=lambda row: (
                row['Time'] <= 172792,
                f"Time exceeds maximum: {row['Time']}"
            ),
            severity=Severity.CRITICAL
        ))
        
        # Critical rules: Class
        rules.append(ValidationRule(
            name="class_binary",
            description="Class must be 0 or 1",
            check_function=lambda row: (
                row['Class'] in [0, 1],
                f"Class is invalid: {row['Class']}"
            ),
            severity=Severity.CRITICAL
        ))
        
        # Critical rules: No nulls
        critical_columns = ['Time', 'Amount', 'Class']
        for col in critical_columns:
            rules.append(ValidationRule(
                name=f"no_null_{col.lower()}",
                description=f"{col} must not be null",
                check_function=lambda row, c=col: (
                    pd.notna(row[c]),
                    f"{c} is null"
                ),
                severity=Severity.CRITICAL
            ))
        
        # Warning rules: V-features
        for i in range(1, 29):
            v_col = f'V{i}'
            rules.append(ValidationRule(
                name=f"v{i}_zscore",
                description=f"{v_col} should have |z-score| <= 10",
                check_function=lambda row, col=v_col: (
                    abs(row[col]) <= 10,
                    f"{col} has extreme value: {row[col]}"
                ),
                severity=Severity.WARNING
            ))
        
        # Info rule: Fraud amount pattern
        rules.append(ValidationRule(
            name="fraud_amount_pattern",
            description="High amounts (>$1000) should rarely be fraud",
            check_function=lambda row: (
                not (row['Amount'] > 1000 and row['Class'] == 1),
                f"High-value fraud: Amount={row['Amount']}, Class={row['Class']}"
            ),
            severity=Severity.INFO
        ))
        
        return rules


class GeminiRuleDiscoverer:
    """Uses Gemini 1.5 Flash to discover validation rules from real data."""
    
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash-exp"):
        self.api_key = api_key
        self.model_name = model
        self.client = genai.Client(api_key=api_key)
    
    def discover_rules(
        self,
        real_data: pd.DataFrame,
        max_rules: int = 10
    ) -> List[ValidationRule]:
        print("\n" + "="*60)
        print("Gemini Rule Discovery")
        print("="*60)
        
        data_summary = self._generate_data_summary(real_data)
        prompt = self._create_rule_discovery_prompt(data_summary, max_rules)
        
        print("\nQuerying Gemini 2.0 Flash for rule discovery...")
        print(f"Model: {self.model_name}")
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            rules_text = response.text
            
            print(f"Received {len(rules_text)} characters of response")
            
            discovered_rules = self._parse_rules_from_gemini(rules_text)
            
            print(f"Successfully discovered {len(discovered_rules)} validation rules")
            print("="*60 + "\n")
            
            return discovered_rules
        
        except Exception as e:
            print(f"Error during rule discovery: {e}")
            print("Continuing without Gemini-discovered rules")
            return []
    
    def _generate_data_summary(self, data: pd.DataFrame) -> str:
        summary_parts = []
        
        summary_parts.append(f"Dataset: {len(data)} records, {len(data.columns)} features")
        summary_parts.append(f"Fraud rate: {(data['Class'].mean() * 100):.4f}%")
        
        amount_stats = data['Amount'].describe()
        summary_parts.append(f"\nAmount statistics:")
        summary_parts.append(f"  Mean: ${amount_stats['mean']:.2f}")
        summary_parts.append(f"  Median: ${amount_stats['50%']:.2f}")
        summary_parts.append(f"  Max: ${amount_stats['max']:.2f}")
        
        fraud_amounts = data[data['Class'] == 1]['Amount']
        legit_amounts = data[data['Class'] == 0]['Amount']
        summary_parts.append(f"\nFraud amounts: median=${fraud_amounts.median():.2f}")
        summary_parts.append(f"Legit amounts: median=${legit_amounts.median():.2f}")
        
        time_stats = data['Time'].describe()
        summary_parts.append(f"\nTime range: {time_stats['min']:.0f} to {time_stats['max']:.0f} seconds")
        
        v_features = [col for col in data.columns if col.startswith('V')]
        summary_parts.append(f"\nV-features: {len(v_features)} PCA-transformed features")
        
        fraud_data = data[data['Class'] == 1]
        legit_data = data[data['Class'] == 0]
        
        mean_diffs = {}
        for v_col in v_features:
            fraud_mean = fraud_data[v_col].mean()
            legit_mean = legit_data[v_col].mean()
            mean_diffs[v_col] = abs(fraud_mean - legit_mean)
        
        top_features = sorted(mean_diffs.items(), key=lambda x: x[1], reverse=True)[:5]
        summary_parts.append(f"\nTop fraud-discriminative features:")
        for v_col, diff in top_features:
            summary_parts.append(f"  {v_col}: |mean_diff|={diff:.2f}")
        
        return '\n'.join(summary_parts)
    
    def _create_rule_discovery_prompt(self, data_summary: str, max_rules: int) -> str:
        return f"""You are analyzing credit card transaction data to discover validation rules for detecting anomalies in synthetically generated data.

DATA SUMMARY:
{data_summary}

TASK:
Discover up to {max_rules} validation rules that can catch invalid or implausible transactions. Focus on:
1. Cross-feature constraints (relationships between Amount, Time, Class, V-features)
2. Fraud-specific patterns (unusual V-feature combinations)
3. Statistical bounds (reasonable ranges)
4. Business logic constraints

REQUIREMENTS:
- Rules must be programmatically checkable
- Rules should be generalizable (not overfitted)
- Focus on catching synthetic data hallucinations
- Avoid overly strict rules

OUTPUT FORMAT (JSON):
[
  {{
    "name": "rule_identifier",
    "description": "Human-readable description",
    "condition": "Python expression returning True if VALID, False if INVALID",
    "severity": "critical" | "warning" | "info",
    "rationale": "Why this rule matters"
  }}
]

CONDITION SYNTAX:
- Use 'row' to access columns: row['Amount'], row['V1'], row['Class']
- Return True if VALID, False if INVALID
- Example: "(row['Amount'] > 100) and (row['Class'] == 1) and (row['V14'] > 0)"

Generate {max_rules} validation rules in JSON format:"""
    
    def _parse_rules_from_gemini(self, rules_text: str) -> List[ValidationRule]:
        try:
            if '```json' in rules_text:
                json_start = rules_text.find('```json') + 7
                json_end = rules_text.find('```', json_start)
                rules_text = rules_text[json_start:json_end].strip()
            elif '```' in rules_text:
                json_start = rules_text.find('```') + 3
                json_end = rules_text.find('```', json_start)
                rules_text = rules_text[json_start:json_end].strip()
            
            rules_data = json.loads(rules_text)
            
            validation_rules = []
            for rule_dict in rules_data:
                condition_str = rule_dict['condition']
                
                if any(dangerous in condition_str for dangerous in ['import', 'exec', 'eval', '__']):
                    print(f"Skipping unsafe rule: {rule_dict['name']}")
                    continue
                
                def make_check_function(condition):
                    def check_function(row):
                        try:
                            is_valid = eval(condition, {"row": row, "pd": pd, "np": np})
                            error_msg = f"Failed: {condition}" if not is_valid else None
                            return is_valid, error_msg
                        except Exception as e:
                            return False, f"Condition error: {str(e)}"
                    return check_function
                
                validation_rules.append(ValidationRule(
                    name=f"gemini_{rule_dict['name']}",
                    description=rule_dict['description'],
                    check_function=make_check_function(condition_str),
                    severity=Severity[rule_dict['severity'].upper()],
                    enabled=True
                ))
            
            return validation_rules
        
        except Exception as e:
            print(f"Error parsing Gemini rules: {e}")
            print(f"Response was: {rules_text[:500]}...")
            return []


class Layer1RuleValidator:
    """
    Layer 1: Fast rule-based validation with Gemini rule discovery.
    
    Performance target: 1000 records/second
    """
    
    def __init__(
        self,
        enable_gemini_rules: bool = True,
        gemini_api_key: Optional[str] = None,
        real_data_path: Optional[str] = None
    ):
        self.enable_gemini_rules = enable_gemini_rules
        self.gemini_api_key = gemini_api_key or config.GEMINI_API_KEY
        self.real_data_path = real_data_path or config.REAL_DATA_PATH
        
        self.static_rules = StaticRuleEngine.get_static_rules()
        self.gemini_rules: List[ValidationRule] = []
        
        if enable_gemini_rules:
            self._discover_gemini_rules()
    
    def _discover_gemini_rules(self):
        print("\nLoading real data for Gemini rule discovery...")
        real_data = pd.read_csv(self.real_data_path)
        print(f"Loaded {len(real_data)} real transactions")
        
        discoverer = GeminiRuleDiscoverer(
            api_key=self.gemini_api_key,
            model=config.LAYER1_GEMINI_MODEL
        )
        
        self.gemini_rules = discoverer.discover_rules(
            real_data,
            max_rules=config.LAYER1_MAX_GEMINI_RULES
        )
    
    def get_all_rules(self) -> List[ValidationRule]:
        return self.static_rules + self.gemini_rules
    
    def validate(self, data: pd.DataFrame) -> ValidationResult:
        print("\n" + "="*60)
        print("Layer 1: Rule-Based Validation")
        print("="*60)
        
        start_time = time.time()
        
        all_rules = self.get_all_rules()
        print(f"\nTotal rules: {len(all_rules)}")
        print(f"  Static rules: {len(self.static_rules)}")
        print(f"  Gemini rules: {len(self.gemini_rules)}")
        
        failures_by_rule: Dict[str, List[ValidationFailure]] = {}
        invalid_indices = set()
        
        print(f"\nValidating {len(data)} records...")
        for idx in range(len(data)):
            row = data.iloc[idx]
            
            for rule in all_rules:
                failure = rule.validate_row(row, idx)
                if failure:
                    if rule.name not in failures_by_rule:
                        failures_by_rule[rule.name] = []
                    failures_by_rule[rule.name].append(failure)
                    invalid_indices.add(idx)
            
            if (idx + 1) % 10000 == 0:
                print(f"  Processed {idx + 1} / {len(data)} records...")
        
        execution_time = time.time() - start_time
        
        valid_count = len(data) - len(invalid_indices)
        invalid_count = len(invalid_indices)
        passed = invalid_count == 0
        
        result = ValidationResult(
            validator_name="Layer1_RuleValidator",
            passed=passed,
            valid_count=valid_count,
            invalid_count=invalid_count,
            invalid_indices=sorted(list(invalid_indices)),
            failures=failures_by_rule,
            execution_time=execution_time,
            metrics={
                'total_rules': len(all_rules),
                'static_rules': len(self.static_rules),
                'gemini_rules': len(self.gemini_rules),
                'records_per_second': len(data) / execution_time if execution_time > 0 else 0
            }
        )
        
        print(result.summary())
        
        return result
    
    def save_report(self, result: ValidationResult, output_path: Optional[str] = None):
        if output_path is None:
            output_path = f"{config.OUTPUT_DIR}/layer1_report.json"
        
        report = result.to_dict()
        report['failures_detail'] = {
            rule_name: {
                'count': len(failures),
                'samples': [f.to_dict() for f in failures[:5]]
            }
            for rule_name, failures in result.failures.items()
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nLayer 1 report saved to: {output_path}")


if __name__ == "__main__":
    """Example usage of Layer 1 validator."""
    
    print("="*80)
    print("LAYER 1 RULE VALIDATOR - Gemini Rule Discovery")
    print("="*80)
    
    from Validation.config import validate_config
    if not validate_config():
        print("\nPlease fix configuration errors before running.")
        exit(1)
    
    validator = Layer1RuleValidator(
        enable_gemini_rules=True,
        gemini_api_key=config.GEMINI_API_KEY,
        real_data_path=config.REAL_DATA_PATH
    )
    
    print(f"\nLoading synthetic data from {config.SYNTHETIC_DATA_PATH}...")
    synthetic_data = pd.read_csv(config.SYNTHETIC_DATA_PATH)
    print(f"Loaded {len(synthetic_data)} synthetic transactions")
    
    result = validator.validate(synthetic_data)
    
    validator.save_report(result)
    
    print("\n" + "="*80)
    print("LAYER 1 VALIDATION COMPLETE")
    print("="*80)
    print(f"Status: {'PASSED' if result.passed else 'FAILED'}")
    print(f"Pass rate: {result.pass_rate:.2f}%")
    print(f"Throughput: {result.metrics['records_per_second']:.0f} rec/s")
    print("="*80)
