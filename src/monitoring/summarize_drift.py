
"""Create a compact summary from an Evidently drift report."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = ROOT / "reports" / "data_drift_summary.json"
OUTPUT_PATH = ROOT / "reports" / "drift_status.json"


def find_drift_information(obj, results=None):
    """Recursively collect drift-related fields from report JSON."""
    if results is None:
        results = []

    if isinstance(obj, dict):
        lowered = {str(key).lower(): value for key, value in obj.items()}

        if "dataset_drift" in lowered or "drift_share" in lowered:
            results.append({
                key: value
                for key, value in lowered.items()
                if key in {"dataset_drift", "drift_share", "number_of_drifted_columns",
                           "number_of_columns"}
            })

        for value in obj.values():
            find_drift_information(value, results)

    elif isinstance(obj, list):
        for item in obj:
            find_drift_information(item, results)

    return results


def main():
    if not REPORT_PATH.exists():
        raise FileNotFoundError(
            f"Report not found: {REPORT_PATH}. Run data_drift.py first."
        )

    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    findings = find_drift_information(report)

    summary = {
        "monitor": "Evidently AI",
        "report_file": str(REPORT_PATH.relative_to(ROOT)),
        "status": "REVIEW_REQUIRED",
        "message": (
            "Inspect the Evidently HTML report to determine which features "
            "drifted and whether further investigation is needed."
        ),
        "extracted_findings": findings,
    }

    OUTPUT_PATH.write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )

    print("Drift summary generated.")
    print(f"Output: {OUTPUT_PATH}")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
