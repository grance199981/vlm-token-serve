"""Triton DivPrune must make the same greedy choices as the PyTorch versions.

1. Bitwise: on the same similarity matrix S, the kernel's sequence equals the pure-PyTorch
   greedy loop over S (same per-element arithmetic, same first-max tie rule).
2. Against select.divprune_indices (k GEMVs instead of one GEMM, so last-bit differences):
   a greedy certificate in float64 -- at every step the chosen token's min distance to the
   already selected set is the maximum, up to 1e-4 -- plus the overlap of the two sets.
   Negative control: a random selection and an argmin "greedy" must fail the certificate.
3. Routing: select("divprune", impl="triton") really runs the kernel (counter), and falls
   back to PyTorch on CPU and above MAX_TOKENS.
GPU tests skip without CUDA; config and fallback tests run on CPU.
"""
import pytest

torch = pytest.importorskip("torch")

from vts_plugin import divprune_triton as dt  # noqa: E402
from vts_plugin.config import parse_config  # noqa: E402
from vts_plugin.select import DIVPRUNE_TRITON_STATS, divprune_indices, select  # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available() or dt.triton is None,
                          reason="needs CUDA + Triton")


def feats(kind, n, d, seed, dtype=torch.float32, device="cuda"):
    g = torch.Generator().manual_seed(seed)
    if kind == "gauss":
        x = torch.randn(n, d, generator=g)
    else:  # clustered: unequal clusters plus noise, closer to image features
        c = torch.randn(12, d, generator=g)
        w = torch.rand(12, generator=g) ** 2 + 0.05
        x = c[torch.multinomial(w / w.sum(), n, replacement=True, generator=g)]
        x = x + 0.6 * torch.randn(n, d, generator=g)
    return x.to(dtype).to(device)


def certificate_violations(x, sel, tol=1e-4):
    """Steps where the chosen token is not (within tol) the farthest from the selected set."""
    x = torch.nn.functional.normalize(x.double(), dim=-1)
    sel = sel.tolist()
    bad = 0
    sim = x @ x.T
    nn = torch.topk(1.0 - sim, 2, dim=1, largest=False).values[:, 1]
    bad += nn[sel[0]] < nn.max() - tol
    md = 1.0 - sim[sel[0]]
    for i in sel[1:]:
        bad += md[i] < md.max() - tol
        md = torch.minimum(md, 1.0 - sim[i])
    return int(bad)


SHAPES = [(17, 64), (100, 256), (784, 2560), (1600, 2560), (3136, 1024), (4096, 512)]


@cuda
@pytest.mark.parametrize("kind", ["gauss", "clustered"])
@pytest.mark.parametrize("n,d", SHAPES)
@pytest.mark.parametrize("keep", [0.5, 0.25, 0.125])
def test_kernel_equals_torch_loop_on_same_matrix(kind, n, d, keep):
    x = feats(kind, n, d, n + d)
    k = max(1, int(n * keep))
    sim = dt.similarity(x).contiguous()
    first = dt.first_pick(sim)
    assert torch.equal(dt.greedy_from_sim_triton(sim, k, first), dt.greedy_from_sim_torch(sim, k, first))


@cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("kind", ["gauss", "clustered"])
@pytest.mark.parametrize("n,d", SHAPES[1:])
def test_greedy_certificate_and_overlap_vs_divprune_indices(kind, n, d, dtype):
    x = feats(kind, n, d, 7 * n + d, dtype)
    k = n // 2
    tri = dt.divprune_indices_triton(x, k)
    ref = divprune_indices(x, k)
    assert certificate_violations(x, tri) == 0
    assert certificate_violations(x, ref) == 0
    assert len(set(tri.tolist())) == k  # no token picked twice
    jacc = len(set(tri.tolist()) & set(ref.tolist())) / len(set(tri.tolist()) | set(ref.tolist()))
    assert jacc >= 0.9, jacc


@cuda
def test_certificate_rejects_non_greedy_choices():
    x = feats("clustered", 784, 256, 1)
    k = 392
    rnd = torch.randperm(784, generator=torch.Generator().manual_seed(0))[:k]
    assert certificate_violations(x, rnd) > k // 2
    sim = dt.similarity(x)
    md, sel = torch.full((784,), float("inf"), device="cuda"), [int(dt.first_pick(sim))]
    for _ in range(k - 1):  # "greedy" that takes the CLOSEST unselected token
        md = torch.minimum(md, 1.0 - sim[sel[-1]])
        cand = md.clone()
        cand[sel] = float("inf")
        sel.append(int(torch.argmin(cand)))
    assert certificate_violations(x, torch.tensor(sel)) > k // 2


@cuda
def test_select_routes_to_triton_and_falls_back_above_max_tokens():
    gh, gw = 28, 28
    x = feats("clustered", gh * gw, 4 * 64, 2)
    r = torch.arange(gh, device="cuda").repeat_interleave(gw)
    c = torch.arange(gw, device="cuda").repeat(gh)
    pos = torch.stack([torch.zeros_like(r), r, c, torch.full_like(r, gw), torch.full_like(r, gh)], 1)
    before = dict(DIVPRUNE_TRITON_STATS)
    f_t, p_t = select("divprune", x, pos, k=392, grid_h=gh, grid_w=gw, seed=0, score_dim=64, divprune_impl="triton")
    assert DIVPRUNE_TRITON_STATS["triton"] == before["triton"] + 1
    f_r, _ = select("divprune", x, pos, k=392, grid_h=gh, grid_w=gw, seed=0, score_dim=64)
    assert DIVPRUNE_TRITON_STATS["triton"] == before["triton"] + 1  # default impl is torch
    assert f_t.shape == f_r.shape == (392, 4 * 64)
    big = feats("gauss", dt.MAX_TOKENS + 1, 32, 3)
    pos_big = torch.zeros(big.shape[0], 5, dtype=torch.long, device="cuda")
    select("divprune", big, pos_big, k=100, grid_h=1, grid_w=big.shape[0], seed=0, divprune_impl="triton")
    assert DIVPRUNE_TRITON_STATS["fallback"] == before["fallback"] + 1


def test_cpu_input_falls_back_to_torch():
    x = feats("gauss", 64, 32, 0, device="cpu")
    pos = torch.zeros(64, 5, dtype=torch.long)
    before = DIVPRUNE_TRITON_STATS["fallback"]
    f, _ = select("divprune", x, pos, k=16, grid_h=8, grid_w=8, seed=0, divprune_impl="triton")
    assert DIVPRUNE_TRITON_STATS["fallback"] == before + 1
    assert torch.equal(f, x[torch.sort(divprune_indices(x, 16)).values])


def test_config_parsing():
    assert parse_config({"VTS_METHOD": "divprune", "VTS_KEEP": "0.5"}).divprune_impl == "torch"
    cfg = parse_config({"VTS_METHOD": "divprune", "VTS_KEEP": "0.5", "VTS_DIVPRUNE_IMPL": "triton"})
    assert cfg.divprune_impl == "triton"
    with pytest.raises(ValueError):
        parse_config({"VTS_METHOD": "rate", "VTS_DIVPRUNE_IMPL": "triton"})
    with pytest.raises(ValueError):
        parse_config({"VTS_METHOD": "divprune", "VTS_DIVPRUNE_IMPL": "cuda"})
