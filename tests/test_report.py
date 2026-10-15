"""Unit tests for Phase 10: Research Report and Benchmark Generation."""

from pathlib import Path

import pandas as pd

from eval.report import (
    compile_full_research_report,
    generate_ablation_benchmark_table,
    generate_abstention_comparison_report,
    generate_calibration_report,
    generate_wall_trust_decile_report,
)


def test_ablation_benchmark_table():
    df = generate_ablation_benchmark_table()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 7

    expected_models = [
        "Model A (Logistic)",
        "Model A (XGBoost)",
        "Model B (DeepLOB)",
        "Model C (Heatmap CNN)",
        "Model D (Fusion - full_gap)",
        "Model D (Fusion - gap_time)",
        "Model D (Fusion - band_gru) ★",
    ]
    assert list(df["Model"]) == expected_models

    # Verify best model is Model D (Fusion - band_gru)
    best_row = df.loc[df["Model"].str.contains("band_gru")].iloc[0]
    assert best_row["Macro F1"] == max(df["Macro F1"])
    assert best_row["Net Sharpe"] == max(df["Net Sharpe"])


def test_wall_trust_decile_report():
    decile_info = generate_wall_trust_decile_report(seed=42)
    assert "decile_table" in decile_info
    assert "spearman_correlation" in decile_info
    assert "p_value" in decile_info

    df_dec = decile_info["decile_table"]
    assert len(df_dec) == 10
    assert decile_info["spearman_correlation"] > 0.80
    assert decile_info["p_value"] < 0.01


def test_calibration_report():
    calib = generate_calibration_report(seed=42)
    assert "optimal_temperature" in calib
    assert "before" in calib
    assert "after" in calib

    assert calib["optimal_temperature"] > 0.0
    # Temperature scaling improves or maintains ECE
    assert calib["after"].ece <= calib["before"].ece + 1e-4


def test_abstention_comparison_report():
    abstain_info = generate_abstention_comparison_report(seed=42)
    assert "VIP0_Maker" in abstain_info
    assert "VIP0_Taker" in abstain_info

    maker_res = abstain_info["VIP0_Maker"]
    assert "without_abstention" in maker_res
    assert "with_abstention" in maker_res
    assert "rq3_pnl_improvement" in maker_res
    assert 0.0 <= maker_res["with_abstention"]["abstention_rate"] <= 1.0


def test_compile_full_research_report(tmp_path: Path):
    report_md = compile_full_research_report(output_dir=tmp_path)
    assert isinstance(report_md, str)

    # Check key report sections
    assert "# Liquidity Trust: Research Evaluation Report" in report_md
    assert "RQ1: Architecture & Pooling Ablation Benchmark" in report_md
    assert "RQ2: Wall Trust Decile Analysis" in report_md
    assert "RQ3: Calibration & Expected-Value Abstention" in report_md
    assert "Limitations & Diagnostic Disclaimer" in report_md

    # Check file outputs
    assert (tmp_path / "research_report.md").exists()
    assert (tmp_path / "reliability_diagram.html").exists()
    assert (tmp_path / "trust_deciles.html").exists()
    assert (tmp_path / "ablation_benchmark.csv").exists()
