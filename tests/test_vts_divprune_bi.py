"""DivPrune under vLLM batch invariance (VLLM_BATCH_INVARIANT=1). An earlier version called
x @ v with a 1-D v, which the batch-invariant aten::matmul override rejects (the server failed at
startup). Runs in a subprocess because the override is process-global; needs CUDA and vllm, and
CUDA_VISIBLE_DEVICES set so it lands on the GPU the launcher reserved."""
import os
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

CODE = r"""
import torch
from vllm.model_executor.determinism.batch_invariant import enable_batch_invariant_mode
from vts_plugin.select import divprune_indices
torch.manual_seed(0)
x = torch.randn(4096, 2560, device="cuda", dtype=torch.bfloat16)  # profile run: 16384 x 2560
enable_batch_invariant_mode()
v = torch.nn.functional.normalize(x.float(), dim=-1)
try:  # negative control: the 0.1.8 form must be rejected, else the override is not active
    v @ v[0]
    print("OLD_FORM_OK")
except ValueError:
    print("OLD_FORM_REJECTED")
a, b = divprune_indices(x, 512), divprune_indices(x, 512)
print("K", a.numel(), torch.unique(a).numel(), "REPEAT_EQUAL", bool(torch.equal(a, b)))
"""


@pytest.mark.skipif(not torch.cuda.is_available() or not os.environ.get("CUDA_VISIBLE_DEVICES"),
                    reason="needs a reserved CUDA device")
def test_divprune_runs_under_batch_invariance():
    src = os.path.join(os.path.dirname(__file__), "..", "src")
    env = dict(os.environ, PYTHONPATH=src + os.pathsep + os.environ.get("PYTHONPATH", ""))
    out = subprocess.run([sys.executable, "-c", CODE], env=env, capture_output=True, text=True,
                         timeout=600)
    assert out.returncode == 0, out.stderr[-3000:]
    assert "OLD_FORM_REJECTED" in out.stdout, out.stdout
    assert "K 512 512 REPEAT_EQUAL True" in out.stdout, out.stdout
