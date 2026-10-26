from pathlib import Path

def test_feasibility_report_exists_and_complete():
    report_path = Path("FeasibilityReport.md")
    assert report_path.exists(), "FeasibilityReport.md must exist at root"
    content = report_path.read_text(encoding="utf-8")

    # Verify required Stage 0 sections and keywords
    required_keywords = [
        "Матрица совместимости",
        "Python 3.11",
        "Demucs",
        "Basic Pitch",
        "All-In-One-Infer",
        "Аппаратное ускорение",
        "CPU",
        "CUDA",
        "ROCm",
        "Лицензии",
        "Бенчмарки",
        "Критерии приёмки Этапа 0",
    ]
    for kw in required_keywords:
        assert kw.lower() in content.lower(), f"Missing keyword: {kw}"
