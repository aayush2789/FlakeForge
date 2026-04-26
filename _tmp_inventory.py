import json, pathlib, collections

root = pathlib.Path("seed_repos/idoft")
rows = []
for p in sorted(root.glob("*/flake_manifest.json")):
    try:
        m = json.loads(p.read_text(encoding="utf-8"))
        rows.append({
            "slug": p.parent.name,
            "difficulty": m.get("difficulty", "unknown"),
            "flake_category": m.get("flake_category", "UNKNOWN"),
            "has_test": bool(m.get("flaky_test_path") or m.get("test_identifier")),
        })
    except Exception as e:
        print("BAD", p.parent.name, e)

by_diff = collections.Counter(r["difficulty"] for r in rows)
by_cat = collections.Counter(r["flake_category"] for r in rows)
print(f"Total: {len(rows)}")
print("by_difficulty:", dict(by_diff))
print("by_category:", dict(by_cat))

for diff in ("easy", "medium", "hard", "unknown"):
    items = [r for r in rows if r["difficulty"] == diff]
    if not items:
        continue
    print(f"\n--- {diff} ({len(items)}) ---")
    for r in items:
        print(f"  {r['slug']}  |  {r['flake_category']}  |  has_test={r['has_test']}")
