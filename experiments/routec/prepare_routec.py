"""Route C data + tasks (docs/W3_CONDITION_ANALYSIS.md §12). Local files only, idempotent.

  ST-VQA: M4C imdb_subval (2628 questions, Rosetta-en OCR) joined by image_path with the
          danjacobellis/stvqa_task1 train parquet (image bytes) -> data/routec/stvqa_subval.parquet
  InfoVQA: lmms-lab/DocVQA InfographicVQA validation parquet as downloaded.
  Tasks:  tasks_routec/vts_stvqa_subval, tasks_routec/vts_infovqa_val (same prompt and ANLS
          metric as lmms-eval's infovqa_val).
  Features for the analysis (doc_id -> n_words, area, h_px): data/routec/*_features.json

Usage: envs/vts/bin/python experiments/routec/prepare_routec.py --root $VTS_ROOT
"""
import argparse
import ast
import glob
import json
import os
import shutil

import lmms_eval
import numpy as np
import pyarrow.parquet as pq
from datasets import Dataset, Features, Image, Sequence, Value

YAML = """task: "{task}"
dataset_path: parquet
dataset_kwargs:
  data_files:
    {split}: {files}
test_split: {split}
output_type: generate_until
doc_to_target: "answers"
doc_to_visual: !function utils.infovqa_doc_to_visual
doc_to_text: !function utils.infovqa_doc_to_text
generation_kwargs:
  max_new_tokens: 32
  temperature: 0
  do_sample: False
metric_list:
  - metric: anls
    aggregation: mean
    higher_is_better: true
lmms_eval_specific_kwargs:
  default:
    pre_prompt: ""
    post_prompt: "\\nAnswer the question using a single word or phrase."
"""


def boxes_xyxy(b):
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    return b


def stvqa(root, out_dir):
    sub = np.load(os.path.join(root, "data/m4c_stvqa/extracted/m4c_stvqa/imdb_subval.npy"), allow_pickle=True)[1:]
    need = {r["image_path"] for r in sub}
    img = {}
    for f in sorted(glob.glob(os.path.join(root, "data/stvqa_task1/data/train-*.parquet"))):
        for batch in pq.ParquetFile(f).iter_batches(batch_size=256, columns=["path", "image"]):
            for p, im in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist()):
                key = p.split("ST-VQA/", 1)[-1]
                if key in need and key not in img:
                    img[key] = im["bytes"]
    missing = need - set(img)
    assert not missing, f"{len(missing)} images missing"
    rows, feats = [], {}
    for i, r in enumerate(sub):
        b = boxes_xyxy(r["ocr_normalized_boxes"])
        w, h = (b[:, 2] - b[:, 0]).clip(0), (b[:, 3] - b[:, 1]).clip(0)
        feats[i] = {"n_words": int(len(r["ocr_tokens"])), "area": float(min(1.0, (w * h).sum())),
                    "h_px": float(np.median(h) * float(r["image_height"])) if len(b) else float("nan"),
                    "question": r["question"]}
        answers = list(dict.fromkeys(a for a in r["answers"]))  # unique, order kept
        rows.append({"question_id": str(r["question_id"]), "question": r["question"], "answers": answers,
                     "image_path": r["image_path"], "image": {"bytes": img[r["image_path"]], "path": None}})
    features = Features({"question_id": Value("string"), "question": Value("string"),
                         "answers": Sequence(Value("string")), "image_path": Value("string"), "image": Image()})
    out = os.path.join(out_dir, "stvqa_subval.parquet")
    Dataset.from_list(rows, features=features).to_parquet(out)
    json.dump(feats, open(os.path.join(out_dir, "stvqa_features.json"), "w"))
    return out, len(rows)


def infovqa(root, out_dir):
    files = sorted(glob.glob(os.path.join(root, "data/lmms-lab/DocVQA/InfographicVQA/validation-*.parquet")))
    feats, i = {}, 0
    for f in files:
        t = pq.read_table(f, columns=["question", "ocr"]).to_pylist()
        for r in t:
            words = json.loads(ast.literal_eval(r["ocr"])[0]).get("WORD", [])
            bb = [w["Geometry"]["BoundingBox"] for w in words]
            feats[i] = {"n_words": len(words), "area": float(min(1.0, sum(b["Width"] * b["Height"] for b in bb))),
                        "h_rel": float(np.median([b["Height"] for b in bb])) if bb else float("nan"),
                        "question": r["question"]}
            i += 1
    json.dump(feats, open(os.path.join(out_dir, "infovqa_features.json"), "w"))
    return os.path.join(root, "data/lmms-lab/DocVQA/InfographicVQA/validation-*.parquet"), i


def make_task(root, task, split, files):
    dst = os.path.join(root, "tasks_routec", task)
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    os.makedirs(dst)
    src = os.path.join(os.path.dirname(lmms_eval.__file__), "tasks", "infovqa", "utils.py")
    shutil.copy(src, os.path.join(dst, "utils.py"))
    open(os.path.join(dst, f"{task}.yaml"), "w").write(YAML.format(task=task, split=split, files=files))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    out_dir = os.path.join(a.root, "data/routec")
    os.makedirs(out_dir, exist_ok=True)
    sp, n_st = stvqa(a.root, out_dir)
    ip, n_info = infovqa(a.root, out_dir)
    make_task(a.root, "vts_stvqa_subval", "validation", sp)
    make_task(a.root, "vts_infovqa_val", "validation", ip)
    print(f"stvqa: {n_st} questions -> {sp}; infovqa: {n_info} questions; tasks in {a.root}/tasks_routec")


if __name__ == "__main__":
    main()
