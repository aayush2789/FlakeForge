import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

def main():
    project_root = Path(__file__).parent
    root = project_root / "test_repos" / "synthetic"
    cases = []
    
    for manifest_path in root.glob("*/flake_manifest.json"):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
                cases.append({
                    "case_id": manifest_path.parent.name,
                    "difficulty": manifest.get("difficulty", "medium"),
                })
        except Exception as e:
            print(f"Error reading {manifest_path}: {e}")
            
    # Sort by difficulty: easy -> medium -> hard
    difficulty_order = {"easy": 0, "medium": 1, "hard": 2}
    cases.sort(key=lambda c: (difficulty_order.get(c["difficulty"], 99), c["case_id"]))
    
    print(f"Found {len(cases)} synthetic repos.")
    print("Running curriculum: Easy -> Medium -> Hard")
    
    results_dir = project_root / "outputs" / "curriculum_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    backup_root = project_root / "outputs" / "synthetic_original_backups"
    worktree_root = project_root / "outputs" / "curriculum_worktrees"
    backup_root.mkdir(parents=True, exist_ok=True)
    worktree_root.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = results_dir / f"curriculum_run_{timestamp}.jsonl"
    
    print(f"Results will be appended to: {summary_file}")
    print(f"Original repo backups: {backup_root}")
    print(f"Agent editable copies: {worktree_root}")
    
    for i, case in enumerate(cases, 1):
        print(f"\n{'='*80}")
        print(f"[{i}/{len(cases)}] Running {case['case_id']} (Difficulty: {case['difficulty'].upper()})")
        print(f"{'='*80}\n")
        
        cmd = [
            sys.executable, "inference.py",
            "--seed-root", str(root),
            "--case", case["case_id"],
            "--backup-root", str(backup_root),
        ]

        env = os.environ.copy()
        env["FF_INFERENCE_REPO_ROOT"] = str(worktree_root)

        # Safety: never pass --no-isolation or --no-backup here. inference.py
        # will patch only copied worktrees and keep original synthetic repos clean.
        completed = subprocess.run(cmd, check=False, cwd=project_root, env=env)
        with summary_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "case_id": case["case_id"],
                "difficulty": case["difficulty"],
                "returncode": completed.returncode,
            }) + "\n")

if __name__ == "__main__":
    main()
