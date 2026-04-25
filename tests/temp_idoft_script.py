import os
import subprocess
import urllib.request
import json
import re

repo_dir = r"c:\Users\Krish Raghuwanshi\Work\FlakeForge\seed_repos\idoft\pybrake"
os.makedirs(repo_dir, exist_ok=True)

print("Cloning repository...")
if not os.path.exists(os.path.join(repo_dir, ".git")):
    subprocess.run(["git", "clone", "https://github.com/airbrake/pybrake", repo_dir])

print("Checking out broken SHA...")
subprocess.run(["git", "checkout", "9bf82941d8bf521055b258cea91596a11e4eb81f"], cwd=repo_dir)

solution_dir = os.path.join(repo_dir, "solution")
os.makedirs(solution_dir, exist_ok=True)

print("Fetching PR Metadata...")
api_url = "https://api.github.com/repos/airbrake/pybrake/pulls/163"
pr_description = "N/A"
try:
    req = urllib.request.Request(api_url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        pr_data = json.loads(response.read().decode("utf-8"))
        pr_description = pr_data.get("body", "No description provided.")
        print(f"PR Title: {pr_data.get('title')}")
except Exception as e:
    print(f"Could not fetch API metadata: {e}")

print("Downloading PR diff...")
diff_url = "https://github.com/airbrake/pybrake/pull/163.diff"
diff_path = os.path.join(solution_dir, "fix.diff")

req = urllib.request.Request(diff_url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req) as response, open(diff_path, 'wb') as out_file:
    data = response.read()
    out_file.write(data)
    diff_text = data.decode("utf-8")

# Extract modified files from diff
files = re.findall(r"diff --git a/(.*?) b/", diff_text)
unique_files = list(set(files))

manifest = {
  "repo_name": "pybrake",
  "flake_category": "RESOURCE_LEAK", 
  "root_cause_file": "pybrake/test_celery_integration.py",
  "root_cause_function": "test_celery_integration",
  "root_cause_description": pr_description.split("\n")[0], # Using the first line of the human PR description
  "correct_actions": ["RESET_STATE", "isolate_state"],
  "correct_primitives": {
    "from": "pass",
    "to": "server.socket.close()"
  },
  "min_pass_rate_after_fix": 0.95,
  "expected_pass_rate_after_fix": 1.0,
  "solution_diff": "solution/fix.diff"
}

with open(os.path.join(repo_dir, "flake_manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

print("\n--- DIFF EXTRACT ---")
print(diff_text)
print("\n--- DONE ---")
