#!/usr/bin/env bash
# auto_report_pipeline.sh
# Runs the full post-training report pipeline for a completed experiment set.
# Usage: bash radio_gs/scripts/auto_report_pipeline.sh [--scene figurines|ramen|teatime|waldo_kitchen|all]

set -euo pipefail
SCENE="${1:-all}"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

ts() { date '+%F %T'; }

echo "[$(ts)] Starting auto_report_pipeline for scene=$SCENE"

# 1. Build per-experiment summaries for all completed runs
echo "[$(ts)] Building per-experiment summaries..."
for exp_dir in output/radio_gs/lerf_*/; do
  name=$(basename "$exp_dir")
  summary_json="$exp_dir/reports/experiment_summary.json"
  eval_json="$exp_dir/lerf_eval_best/summary.json"
  # Only build if eval is done but summary is missing or stale
  if [[ -f "$eval_json" ]]; then
    if [[ ! -f "$summary_json" ]] || [[ "$eval_json" -nt "$summary_json" ]]; then
      echo "[$(ts)]   Building summary for $name"
      bash radio_gs/scripts/run_repo_python.sh radio_gs/scripts/build_experiment_summary.py --exp_dir "$exp_dir" || true
    fi
  fi
done

# 2. Build seed robustness report (all 4 scenes)
echo "[$(ts)] Building seed robustness report..."
bash radio_gs/scripts/run_repo_python.sh radio_gs/scripts/build_lerf_seed_robustness_report.py

# 3. Build paper main table (Markdown + LaTeX)
echo "[$(ts)] Building paper main table..."
bash radio_gs/scripts/run_repo_python.sh radio_gs/scripts/build_paper_main_table.py

# 4. Build canonical submission tables
echo "[$(ts)] Building canonical submission tables..."
bash radio_gs/scripts/run_repo_python.sh radio_gs/scripts/build_submission_tables.py --lerf_eval_dir output/radio_gs

# 5. Build efficiency profile
echo "[$(ts)] Building efficiency profile..."
bash radio_gs/scripts/run_repo_python.sh radio_gs/scripts/profile_training_efficiency.py

# 6. Print summary
echo ""
echo "=========================================="
echo "  REPORT PIPELINE COMPLETE"
echo "=========================================="
echo "  Seed robustness: output/radio_gs/reports/seed_robustness_summary.md"
echo "  Paper main table: output/radio_gs/reports/paper_main_table.md"
echo "  Submission main table: output/radio_gs/reports/paper_submission_main_table.md"
echo "  Efficiency profile: output/radio_gs/reports/efficiency_profile.md"
echo "=========================================="
echo ""
# Print current paper table
cat output/radio_gs/reports/paper_main_table.md | grep -A20 "## Main Table"
