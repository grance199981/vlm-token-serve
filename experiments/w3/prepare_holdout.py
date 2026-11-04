"""Hold-out set for the H-C4 confirmation (docs/W3_CONDITION_ANALYSIS.md §7): TextVQA val
rows 1500-4999 (not used in W3) as their own parquet + an lmms-eval task
`vts_textvqa_holdout` (a copy of tasks_local/vts_textvqa_val with new data_files).
Idempotent; derived from the local copy only (no download).

Usage: envs/vts/bin/python experiments/w3/prepare_holdout.py --root $VTS_ROOT
"""
import argparse
import glob
import os
import re
import shutil

import pyarrow as pa
import pyarrow.parquet as pq

START, END = 1500, 5000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    src = sorted(glob.glob(os.path.join(a.root, "data/lmms-lab/textvqa/data/validation-*.parquet")))
    out_dir = os.path.join(a.root, "data/holdout")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"textvqa_val_{START}_{END - 1}.parquet")
    if not os.path.exists(out):
        table = pa.concat_tables([pq.read_table(f) for f in src])
        assert table.num_rows == END, table.num_rows
        pq.write_table(table.slice(START, END - START), out)
    t = pq.read_table(out, columns=["question_id", "question"])
    full = pa.concat_tables([pq.read_table(f, columns=["question_id"]) for f in src])
    assert t.num_rows == END - START
    assert t.column("question_id")[0].as_py() == full.column("question_id")[START].as_py()
    task_src = os.path.join(a.root, "tasks_local/vts_textvqa_val")
    task_dst = os.path.join(a.root, "tasks_holdout/vts_textvqa_holdout")
    if os.path.isdir(task_dst):
        shutil.rmtree(task_dst)
    os.makedirs(task_dst)
    for f in os.listdir(task_src):
        if f == "__pycache__":
            continue
        if f == "vts_textvqa_val.yaml":
            text = open(os.path.join(task_src, f), encoding="utf-8").read()
            text, n1 = re.subn(r'^task: "vts_textvqa_val"$', 'task: "vts_textvqa_holdout"', text, flags=re.M)
            text, n2 = re.subn(r"^(\s+validation: ).*$", r"\g<1>" + out, text, flags=re.M)
            assert n1 == 1 and n2 == 1, (n1, n2)
            open(os.path.join(task_dst, "vts_textvqa_holdout.yaml"), "w", encoding="utf-8").write(text)
        else:
            shutil.copy(os.path.join(task_src, f), task_dst)
    print(f"holdout: {out} ({t.num_rows} rows, first question_id {t.column('question_id')[0].as_py()}); task in {task_dst}")


if __name__ == "__main__":
    main()
