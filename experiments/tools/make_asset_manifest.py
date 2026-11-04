"""Build configs/assets_manifest.json from the official huggingface.co API.

Run locally (for servers that cannot reach huggingface.co). The manifest pins each repo
to a commit and records, per file, the LFS sha256 or the git blob sha1, so the
copy fetched from a mirror can be verified against the official hashes.
"""
import argparse
import fnmatch
import json
import time
import urllib.request

# (kind, repo, local subdir, allow patterns)
ASSETS = [
    ("models", "Qwen/Qwen3.5-4B", "models/Qwen3.5-4B", ["*"]),
    ("models", "Qwen/Qwen3-VL-4B-Instruct", "models/Qwen3-VL-4B-Instruct", ["*"]),
    ("datasets", "lmms-lab/ChartQA", "data/lmms-lab/ChartQA", ["*.md", ".gitattributes", "data/test-*"]),
    ("datasets", "lmms-lab/textvqa", "data/lmms-lab/textvqa", ["*.md", ".gitattributes", "data/validation-*"]),
    ("datasets", "lmms-lab/DocVQA", "data/lmms-lab/DocVQA", ["*.md", ".gitattributes", "DocVQA/validation-*"]),
    ("datasets", "lmms-lab/MME", "data/lmms-lab/MME", ["*.md", ".gitattributes", "data/test-*"]),
    ("datasets", "lmms-lab/MMBench", "data/lmms-lab/MMBench", ["*.md", ".gitattributes", "en/dev-*"]),
]
# Route C (docs/W3_CONDITION_ANALYSIS.md §11-12), user-approved 2026-10-22.
# lmms-lab/DocVQA is the same commit as lmms-lab-encoder/DocVQA (what lmms-eval reads).
ASSETS_ROUTE_C = [
    ("datasets", "danjacobellis/stvqa_task1", "data/stvqa_task1", ["*.md", ".gitattributes", "data/train-*"]),
    ("datasets", "lmms-lab/DocVQA", "data/lmms-lab/DocVQA", ["InfographicVQA/validation-*"]),
]


def get_json(url):
    err = None
    for _ in range(5):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except Exception as e:  # network is flaky; retry
            err = e
            time.sleep(3)
    raise err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="configs/assets_manifest.json")
    ap.add_argument("--set", choices=["main", "route_c"], default="main")
    args = ap.parse_args()
    entries = []
    for kind, repo, subdir, patterns in (ASSETS if args.set == "main" else ASSETS_ROUTE_C):
        meta = get_json(f"https://huggingface.co/api/{kind}/{repo}?blobs=true")
        files = []
        for s in meta["siblings"]:
            name = s["rfilename"]
            if not any(fnmatch.fnmatch(name, p) for p in patterns):
                continue
            lfs = s.get("lfs")
            files.append({
                "path": name,
                "size": s.get("size"),
                "sha256": lfs["sha256"] if lfs else None,
                "git_blob_sha1": None if lfs else s.get("blobId"),
            })
        cd = meta.get("cardData") or {}
        entries.append({
            "kind": kind,
            "repo": repo,
            "revision": meta["sha"],
            "local_dir": subdir,
            "license": cd.get("license") or cd.get("license_name"),
            "total_bytes": sum(f["size"] or 0 for f in files),
            "files": files,
        })
        print(f"{repo}@{meta['sha'][:12]}: {len(files)} files, {entries[-1]['total_bytes'] / 1e9:.2f} GB")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"generated_from": "https://huggingface.co/api", "assets": entries}, f, indent=1)
        f.write("\n")


if __name__ == "__main__":
    main()
