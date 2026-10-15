"""Evaluation: splits, calibration, abstention, backtesting, bootstrapping."""

from eval.abstention import (
    RiskCoverageResult,
    apply_abstention_rule,
    compare_with_and_without_abstention,
    compute_expected_value,
    compute_risk_coverage_curve,
    tune_abstention_thresholds,
)
from eval.backtest import BacktestResult, EventDrivenBacktester, FeeScenario, TradeRecord
from eval.bootstrap import BootstrapResult, compute_block_bootstrap_ci, moving_block_bootstrap
from eval.calibration import CalibrationResult, TemperatureScaler, compute_calibration_metrics
from eval.report import (
    compile_full_research_report,
    export_report_figures,
    generate_ablation_benchmark_table,
    generate_abstention_comparison_report,
    generate_calibration_report,
    generate_wall_trust_decile_report,
)
from eval.splits import FoldSplit, create_train_test_split, walk_forward_splits

__all__ = [
    "create_train_test_split",
    "walk_forward_splits",
    "FoldSplit",
    "FeeScenario",
    "TradeRecord",
    "BacktestResult",
    "EventDrivenBacktester",
    "moving_block_bootstrap",
    "compute_block_bootstrap_ci",
    "BootstrapResult",
    "TemperatureScaler",
    "compute_calibration_metrics",
    "CalibrationResult",
    "apply_abstention_rule",
    "compute_expected_value",
    "tune_abstention_thresholds",
    "compute_risk_coverage_curve",
    "RiskCoverageResult",
    "compare_with_and_without_abstention",
    "generate_ablation_benchmark_table",
    "generate_wall_trust_decile_report",
    "generate_calibration_report",
    "generate_abstention_comparison_report",
    "export_report_figures",
    "compile_full_research_report",
]

__version__ = "0.1.0"
