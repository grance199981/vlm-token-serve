"""Download pinned assets from a Hugging Face mirror and verify them against
configs/assets_manifest.json (hashes taken from the official huggingface.co API).

Standard library only, so it runs with the shared Python before the project
venv exists. Resumable (HTTP Range), retries, writes to *.part and renames only
after the hash matches. Already-verified files are skipped.

Usage (on the GPU server):
  python -I experiments/remote/fetch_assets.py --manifest configs/assets_manifest.json \
      --root $VTS_ROOT [--only Qwen/Qwen3.5-4B ...] [--verify-only]
"""
import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request

CHUNK = 8 << 20


def file_hashes(path, size_hint=None):
    sha256 = hashlib.sha256()
    sha1 = hashlib.sha1()
    size = os.path.getsize(path)
    sha1.update(b"blob %d\0" % size)
    with open(path, "rb") as f:
        while True:
            b = f.read(CHUNK)
            if not b:
                break
            sha256.update(b)
            sha1.update(b)
    return sha256.hexdigest(), sha1.hexdigest(), size


def verified(path, entry):
    if not os.path.exists(path) or os.path.getsize(path) != entry["size"]:
        return False
    s256, s1, _ = file_hashes(path)
    if entry["sha256"]:
        return s256 == entry["sha256"]
    return s1 == entry["git_blob_sha1"]


def url_for(endpoint, asset, path):
    prefix = "datasets/" if asset["kind"] == "datasets" else ""
    return f"{endpoint}/{prefix}{asset['repo']}/resolve/{asset['revision']}/{path}"


def download(url, dest, size, retries=8):
    part = dest + ".part"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    for attempt in range(1, retries + 1):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if have > size:
            os.remove(part)
            have = 0
        if have == size:
            break
        req = urllib.request.Request(url, headers={"User-Agent": "vlm-tokenserve-fetch"})
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                if have and r.status != 206:  # server ignored Range: restart
                    have = 0
                with open(part, "ab" if have else "wb") as f:
                    while True:
                        b = r.read(CHUNK)
                        if not b:
                            break
                        f.write(b)
        except Exception as e:
            print(f"    attempt {attempt} failed: {e}", flush=True)
            time.sleep(min(60, 5 * attempt))
            continue
        if os.path.getsize(part) == size:
            break
    if not os.path.exists(part) or os.path.getsize(part) != size:
        raise RuntimeError(f"incomplete download: {url}")
    return part


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--endpoint", default="https://hf-mirror.com")
    ap.add_argument("--only", nargs="*", default=None, help="repo ids to process")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    manifest = json.load(open(args.manifest, encoding="utf-8"))
    report = []
    failures = 0
    for asset in manifest["assets"]:
        if args.only and asset["repo"] not in args.only:
            continue
        print(f"== {asset['repo']}@{asset['revision'][:12]} -> {asset['local_dir']}", flush=True)
        for entry in asset["files"]:
            dest = os.path.join(args.root, asset["local_dir"], entry["path"])
            status = "ok"
            if verified(dest, entry):
                status = "ok(cached)"
            elif args.verify_only:
                status = "MISSING_OR_BAD"
            else:
                t0 = time.time()
                part = download(url_for(args.endpoint, asset, entry["path"]), dest, entry["size"])
                os.replace(part, dest)
                if not verified(dest, entry):
                    os.replace(dest, dest + ".bad")
                    status = "HASH_MISMATCH"
                else:
                    dt = time.time() - t0
                    status = f"ok({entry['size'] / max(dt, 1e-6) / 1e6:.1f} MB/s)"
            if not status.startswith("ok"):
                failures += 1
            print(f"   {status:22s} {entry['path']} ({entry['size']} B)", flush=True)
            report.append({"repo": asset["repo"], "revision": asset["revision"],
                           "path": entry["path"], "status": status})
    out = os.path.join(args.root, "logs", "fetch_assets_report.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"failures": failures, "files": report}, f, indent=1)
    print(f"FAILURES: {failures}  report: {out}", flush=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
