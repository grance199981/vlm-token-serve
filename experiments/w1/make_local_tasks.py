"""Generate lmms-eval task variants (prefix ``vts_``) that read our verified local
parquet copies (the server cannot reach huggingface.co).

Only data loading and task names change: ``dataset_path/dataset_name/dataset_kwargs``
are removed from the copied yaml files and replaced by the ``parquet`` builder with
explicit ``data_files`` (only the needed split was downloaded, so the repo README
configs cannot be used). Prompts, generation kwargs and metrics are untouched.

Usage (on the GPU server): envs/vts/bin/python experiments/w1/make_local_tasks.py --root $VTS_ROOT
"""
import argparse
import os
import re
import shutil

import lmms_eval

# task -> (task dir, main yaml, extra files to copy, split, data glob relative to data/lmms-lab)
TASKS = {
    "chartqa": ("chartqa", "chartqa.yaml", ["utils.py"], "test", "ChartQA/data/test-*.parquet"),
    "textvqa_val": ("textvqa", "textvqa_val.yaml", ["_default_template_textvqa_yaml", "utils.py"],
                    "validation", "textvqa/data/validation-*.parquet"),
    "docvqa_val": ("docvqa", "docvqa_val.yaml", ["_default_template_docvqa_yaml", "utils.py"],
                   "validation", "DocVQA/DocVQA/validation-*.parquet"),
    "mme": ("mme", "mme.yaml", ["utils.py"], "test", "MME/data/test-*.parquet"),
    "mmbench_en_dev_static": ("mmbench", "mmbench_en_dev_static.yaml",
                              ["_default_template_mmbench_en_yaml", "en_utils.py"],
                              "dev", "MMBench/en/dev-*.parquet"),
}
DATA_KEYS = ("dataset_path:", "dataset_name:", "dataset_kwargs:")


def strip_dataset_keys(text):
    out, skipping = [], False
    for line in text.splitlines():
        if skipping and (line.startswith(" ") or line.startswith("\t")):
            continue  # indented continuation of a removed key
        skipping = False
        if line.startswith(DATA_KEYS):
            skipping = line.startswith("dataset_kwargs:")
            continue
        out.append(line)
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    src_root = os.path.join(os.path.dirname(lmms_eval.__file__), "tasks")
    out_root = os.path.join(args.root, "tasks_local")
    data_root = os.path.join(args.root, "data", "lmms-lab")
    if os.path.isdir(out_root):
        shutil.rmtree(out_root)
    for task, (tdir, main_yaml, extras, split, glob) in TASKS.items():
        src, dst = os.path.join(src_root, tdir), os.path.join(out_root, "vts_" + task)
        os.makedirs(dst)
        for f in extras:
            text = open(os.path.join(src, f), encoding="utf-8").read()
            if f.endswith("yaml") or f.startswith("_default_template"):
                text = strip_dataset_keys(text)
            if f == "en_utils.py":  # reads mmbench.yaml next to itself; point it at the installed copy
                text = text.replace('Path(__file__).parent / "mmbench.yaml"', repr(os.path.join(src, "mmbench.yaml")))
            open(os.path.join(dst, f), "w", encoding="utf-8").write(text)
        text = strip_dataset_keys(open(os.path.join(src, main_yaml), encoding="utf-8").read())
        text, n = re.subn(r'^task:\s*"?' + re.escape(task) + r'"?\s*$', f'task: "vts_{task}"', text, flags=re.M)
        assert n == 1, (task, n)
        files = os.path.join(data_root, glob)
        text += f"dataset_path: parquet\ndataset_kwargs:\n  data_files:\n    {split}: {files}\n"
        open(os.path.join(dst, "vts_" + main_yaml), "w", encoding="utf-8").write(text)
        assert "lmms-lab-encoder" not in "".join(open(os.path.join(dst, f), encoding="utf-8").read()
                                                 for f in os.listdir(dst) if f.endswith(("yaml", "_yaml")))
        print(f"vts_{task}: {dst} <- {files}")


if __name__ == "__main__":
    main()
