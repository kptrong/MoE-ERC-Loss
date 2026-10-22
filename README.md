# MoE + ERC research

Build a decoder-only language model from random initialization to compare Vanilla
MoE with Expert-Router Coupling (ERC), following *Coupling Experts and Routers in
Mixture-of-Experts via an Auxiliary Loss* (ICLR 2026). The reference is
`references/erc_paper.pdf`. Target hardware: one NVIDIA A100 on Google Colab Pro.

STEP 1 provides validated configuration and analytical parameter counts. STEP 2
adds standalone RMSNorm, RoPE and causal self-attention in PyTorch. STEP 3 adds
SwiGLU experts, a full-softmax Top-K router and sparse MoE dispatch. STEP 4 adds
Pre-Norm decoder blocks, the complete tied-embedding LM, next-token cross entropy
and synthetic forward/backward smoke tests. ERC, load-balancing loss, data loading
and training remain future steps. There are no pretrained weights or dataset downloads.

## Architecture and budget

| Setting | Value |
| --- | --- |
| Vocabulary / hidden width | 16,000 / 512 |
| Transformer blocks / attention heads | 6 / 8 |
| Attention | Full multi-head causal self-attention; no GQA |
| Positional encoding / normalization | RoPE / RMSNorm |
| Experts per block / selected per token | 16 / 2 |
| Expert | SwiGLU: gate, up and down projections; intermediate width 256 |
| Maximum sequence length / dropout | 512 / 0.0 |
| Linear biases / shared experts | None / none |
| Input embedding and LM head | Tied |

Each block has one attention sublayer, one MoE sublayer, its own router, and two
scale-only RMSNorms. A final scale-only RMSNorm follows the stack. This explicitly
fixes the normalization layout assumed by the requested budget. Both sublayers
use Pre-Norm and residual connections. RoPE theta 10,000 and RMSNorm epsilon 1e-6
are explicit, nonlearned project defaults, not claims about the paper's recipe.

With vocabulary V, hidden width d, depth L, expert count n and expert width D:

| Component | Formula | Default parameters |
| --- | --- | ---: |
| Token embeddings, including tied output | V d | 8,192,000 |
| Attention Q, K, V, output | L (4 d²) | 6,291,456 |
| SwiGLU experts | L n (3 d D) | 37,748,736 |
| Router weights | L n d | 49,152 |
| RMSNorm scales | (2 L + 1) d | 6,656 |
| **Total unique trainable parameters** | Sum above | **52,288,000** |

The LM head adds zero parameters because of tying. RoPE adds no learned
parameters. Top-K changes computation, not the number of stored parameters.
The STEP 4 test checks every component and the total against an instantiated
PyTorch model, counting the shared embedding/LM-head Parameter only once.

This 52.288M configuration is a scaled research design. The paper's 3B baseline
uses 12 layers, d=1536, D=768, 64 experts and K=8; its 15B experiment uses 24
layers and 256 experts with K=8. Results and optimal alpha do not automatically
transfer to this smaller model. No ERC choices are implemented yet.

## Project files

```text
moe_erc/
  __init__.py           Public configuration and counting API
  __main__.py           Parameter-budget command-line entry point
  config.py             Frozen dataclass, validation, serializable settings
  parameter_count.py    Component formulas and total count
  rmsnorm.py            Scale-only RMSNorm with FP32 low-precision arithmetic
  rope.py               Parameter-free adjacent-pair rotary encoding for Q/K
  attention.py          Bias-free causal multi-head self-attention using SDPA
  experts.py            Bias-free SwiGLU expert with accessible gate projection
  router.py             Full-softmax Top-K routing; no Top-K renormalization
  moe.py                Sparse expert dispatch and optional detached diagnostics
  decoder.py            Pre-Norm attention/MoE block with both residuals
  model.py              Tied-embedding LM, input validation and shifted CE loss
  smoke_test.py         Full-size synthetic forward/backward CLI
tests/
  test_config.py        Defaults, validation and immutability tests
  test_parameter_count.py  Component totals and architecture sensitivity
  test_transformer_components.py  Numerical, causal, backward and CUDA tests
  test_experts.py        Expert formula, parameter count and gradient tests
  test_router.py         Routing probabilities, Top-K and router gradient tests
  test_moe.py            Sparse/dense comparison, dispatch, diagnostics and BF16
  test_decoder.py        Pre-Norm/residual structure and block gradients
  test_model.py          Full budget, tying, causal loss, validation and diagnostics
references/
  erc_paper.pdf          Existing primary methodology reference
pyproject.toml          Package metadata, Python requirement and torch dependency
.gitignore              Generated Python/build artifacts to ignore
README.md               Scope, architecture and usage
```

## Environment setup

Use Python 3.10 or newer and PyTorch 2.0 or newer (SDPA is required).
In Colab select a GPU runtime and check the assigned
device; the requested GPU is an A100. Colab commonly supplies PyTorch already.
From the project root install the local package:

```bash
python -m pip install -e .
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

For this Windows workspace, use the existing Conda environment with
`conda activate ml_env`, then confirm `python -c "import sys; print(sys.executable)"`
points into `miniconda3/envs/ml_env`. To select it explicitly without activation:

```powershell
conda run -n ml_env python -m unittest discover -s tests -v
```

An isolated `.venv` is an alternative, not required alongside Conda.
Install a CUDA-compatible PyTorch build if your environment lacks one.
The minimal dependency is declared in pyproject.toml; this is not a pinned
reproduction environment. Record the actual Python, PyTorch, CUDA and GPU versions
before future training runs. No additional test dependency is needed.

## Verification commands

Run from the repository root:

```bash
python -m moe_erc
python -m unittest discover -s tests -v
python -m moe_erc.smoke_test
python -m moe_erc.smoke_test --device cuda --bf16
```

The counter must report **52,288,000** total parameters. STEP 1's counter and
tests still require only the standard library; STEP 2 through STEP 4 tests require PyTorch.
CUDA/BF16 tests are skipped explicitly when suitable hardware is absent. CPU
tests include a numerical reference for RMSNorm, nonzero RoPE rotations, a small
independent attention calculation, actual future-token perturbations, backward
passes and instantiated module counts. Full-size CPU forward/backward, true
embedding tying and the 52,288,000-parameter model are also tested. Most behavioral
tests use a smaller configuration; the full budget test uses the default model.

## STEP 2 building blocks

```python
import torch
from moe_erc import ModelConfig
from moe_erc.rmsnorm import RMSNorm
from moe_erc.rope import RotaryPositionalEmbedding
from moe_erc.attention import CausalSelfAttention

config = ModelConfig()
norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
attention = CausalSelfAttention(config)
x = torch.randn(2, 128, config.d_model)
y = attention(norm(x))  # [2, 128, 512]; no decoder residual/block yet
```

RMSNorm computes `x * rsqrt(mean(x**2) + eps) * weight` over the final dimension,
with no mean subtraction or bias. FP16/BF16 normalization and scaling use FP32;
the output retains the input dtype. Its 512 learned scales match the budget.

RoPE rotates adjacent coordinate pairs with angles
`position * base**(-2 * pair_index / head_dim)`. This fixes the coordinate
convention explicitly. Positions start at zero on each call and lengths 1 through
512 are supported. Angles are computed on the input device in FP32 (FP64 for
double input), without learned parameters or dtype-sensitive cached buffers.
The attention module rotates Q/K only, leaves V unchanged, calls
`scaled_dot_product_attention(..., is_causal=True)`, then merges heads and applies
the output projection. SDPA dropout uses the configuration during training and
zero in evaluation, as required by the [PyTorch SDPA documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html).
There are exactly **1,048,576** learned parameters per attention module, with four
512-by-512 projection matrices and no biases. No residual connection, padding
mask, KV cache, router or expert is included in this building block.

For A100 BF16 execution, move modules and inputs to CUDA and use
`torch.autocast(device_type="cuda", dtype=torch.bfloat16)`. The tests also cover
explicit BF16 module conversion when CUDA supports it. GPU execution must be
verified on the actual target runtime; CPU tests cannot establish A100 behavior.

Earlier verification on 2026-10-22 was blocked by missing PyTorch. The complete
suite has now run successfully in `ml_env`; see the STEP 4 verification below.

## STEP 3 sparse MoE

```python
from moe_erc.moe import SparseMoE

moe = SparseMoE(config)
output = moe(x)  # Tensor [B,T,d_model]
output, diagnostics = moe(x, return_diagnostics=True)
router_weight = moe.router.weight       # [16,512], original Parameter R
gate_weight = moe.experts[0].gate_proj.weight  # [256,512]
all_gate_weights = moe.get_gate_weights()     # [16,256,512], differentiable stack
```

Each expert computes `down_proj(silu(gate_proj(x)) * up_proj(x))`. PyTorch stores
each gate as `[D,d]`, transposed relative to the paper's `[d,D]` convention.
Router output exposes `logits`, `probabilities`, `topk_indices` and `topk_weights`.
It accepts `[B,T,d]` or `[N,d]`. Probabilities are softmax over **all** experts;
selected weights are **not renormalized**. Low-precision logits use FP32 softmax.

SparseMoE flattens tokens, loops over experts, gathers only assigned tokens and
uses `index_add_` to sum weighted outputs at original token positions. There is
no loop over tokens and no capacity limit or token dropping. Empty experts are
not executed and may have `None` gradients. Selected experts, routing weights and
input retain autograd connectivity. No normalization parameters are added.
FP16/BF16 expert outputs and routing weights are multiplied and accumulated in
FP32, preserving the full-softmax weighted-sum formula; the final output is cast
to the input dtype. Under CUDA autocast with FP32 input this means FP32 output.

Optional diagnostics expose detached `token_counts`, `assignment_fractions`,
`mean_probabilities` and `topk_indices` without caching them in the module.
Counts sum to `B*T*K`; fractions divide by that number and sum to one. Mean
probabilities average the full distribution across tokens. Indices have shape
`[B,T,K]`. These diagnostics are for logging; do not use their detached values
to construct a differentiable load-balancing loss. The gate accessor and original
router Parameter remain attached for future ERC work; no ERC is implemented.

| Component | Parameters |
| --- | ---: |
| One SwiGLU expert | 393,216 |
| Sixteen experts | 6,291,456 |
| Router | 8,192 |
| One SparseMoE | 6,299,648 |
| Six SparseMoE layers | 37,797,888 |

The dense implementation exists only in tests. Tests compare sparse and dense
outputs and gradients in FP64, count actual expert dispatch through hooks, check
empty experts and single tokens, and validate diagnostics and ERC parameter
access. CUDA tests exercise both BF16 autocast and explicit BF16 modules when
hardware supports them. Seeds are fixed; exact bitwise CUDA reproducibility is
not asserted for parallel indexed accumulation or tied Top-K ordering.

STEP 3's sparse/dense output and gradient checks now pass, along with the rest of
the STEP 1-4 suite described below.

## STEP 4 complete language model

Each decoder block computes exactly:

```text
h = x + attention(norm_1(x))
output = h + moe(norm_2(h))
```

The model applies token embedding, six independent blocks, final RMSNorm and a
bias-free LM head. `model.lm_head.weight is model.token_embedding.weight` is true:
these are the same Parameter, including after device/dtype conversion.

```python
from moe_erc.model import MoELanguageModel

model = MoELanguageModel(ModelConfig())
input_ids = torch.randint(16000, (2, 16))
logits = model(input_ids)  # Tensor [2,16,16000]
result = model(input_ids, labels=input_ids, return_routing_info=True)
result.loss.backward()
info = result.routing_info[0]
assert info.layer_index == 0
first_layer_mean_probabilities = info.diagnostics.mean_probabilities

# Existing parameter access stays connected to autograd for future ERC:
R = model.layers[0].moe.router.weight
W_gate = model.layers[0].moe.experts[0].gate_proj.weight
W_gates = model.layers[0].moe.get_gate_weights()
```

Input IDs must be int32/int64 `[B,T]`, with B positive, T in [1,512] and token IDs
in the vocabulary. Optional labels have the same shape/device and integer dtype;
only vocabulary IDs and -100 are accepted. The model compares `logits[:,:-1]`
with `labels[:,1:]` using FP32 cross entropy, with `ignore_index=-100`. There is
no auxiliary loss. For T=1 or no nonignored shifted targets, loss is differentiable
zero, including a valid backward pass.

`model(input_ids)` returns logits directly. Supplying labels or setting
`return_routing_info=True` returns `LanguageModelOutput` with `logits`, optional
`loss`, and optional `routing_info`. Each routing entry has an explicit zero-based
`layer_index` and the existing detached `RoutingDiagnostics`. Mean routing
probabilities are provided rather than full per-token probabilities/logits.
Diagnostics are allocated only on request and never stored on the model.

### Verified locally on 2026-10-23

Environment: Conda `ml_env`, Python 3.11.16, PyTorch 2.11.0+cu128, NVIDIA GeForce
RTX 4050 Laptop GPU (6 GB), driver 582.05. No package installation was needed.

- Complete STEP 1-4 suite: **59 passed, 0 failed, 0 skipped**, including CUDA/BF16.
- The suite used two CPU threads to avoid overhead from many small expert calls:
  `python -B -c "import torch,unittest; torch.set_num_threads(2); suite=unittest.defaultTestLoader.discover('tests'); result=unittest.TextTestRunner(verbosity=2).run(suite); raise SystemExit(not result.wasSuccessful())"`.
- Actual unique trainable parameters: **52,288,000**, matching all analytical components.
- Full-size CPU smoke: logits `[1,16,16000]` FP32, finite loss 468.234009,
  finite backward gradients and tied embedding confirmed.
- Full-size CUDA/BF16 smoke: logits `[1,16,16000]` BF16, FP32 loss 482.044281,
  finite backward gradients and tied embedding confirmed.

The loss numbers are sanity checks on random IDs with default random PyTorch
initialization, not quality measurements. CPU and CUDA generate random input on
their own devices, so these two smoke losses are not a precision comparison.
Unselected experts can legitimately have `None` gradients. Smoke scripts perform
one backward pass without an optimizer or weight update. Execution on the target
Colab A100 has not yet been verified. No STEP 5 work is included.
