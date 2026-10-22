# MoE + ERC research

Build a decoder-only language model from random initialization to compare Vanilla
MoE with Expert-Router Coupling (ERC), following *Coupling Experts and Routers in
Mixture-of-Experts via an Auxiliary Loss* (ICLR 2026). The reference is
`references/erc_paper.pdf`. Target hardware: one NVIDIA A100 on Google Colab Pro.

STEP 1 provides validated configuration and analytical parameter counts. STEP 2
adds standalone RMSNorm, RoPE and causal self-attention in PyTorch. STEP 3 adds
SwiGLU experts, a full-softmax Top-K router and sparse MoE dispatch. STEP 4 adds
Pre-Norm decoder blocks, the complete tied-embedding LM, next-token cross entropy
and synthetic forward/backward smoke tests. STEP 5 adds train-only byte-level BPE,
Dolma/local streaming readers, resumable packing, memmapped datasets and data
inspection CLIs. ERC, load-balancing loss and model training remain future steps.
STEP 5A adds a separate Autonomy-of-Experts baseline with factorized gate
projections and norm-based routing, using the same decoder/LM forward and data pipeline.
There are no pretrained weights or real dataset shards downloaded in verification.

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
  dolma_reader.py       Manifest selection, bounded downloads and JSONL/gzip reader
  tokenizer_utils.py    Stable document split and frozen train-only byte-level BPE
  data_processing.py    Packing, shard checkpoints, fingerprints and integrity audit
  dataset.py            Lazy uint16 memmap Dataset and seeded DataLoader
  data_smoke.py         Offline data-to-model synthetic smoke test
  aoe_expert.py         Factorized SwiGLU expert with reusable gate-down activation
  aoe.py                Sparse AoE, norm routing, diagnostics and AoE decoder subclass
  model_factory.py      Select vanilla_moe or aoe without a new ERC architecture
scripts/
  inspect_dolma.py      Dry-run shard selection and optional explicit downloads
  train_tokenizer.py    Train and save production/test tokenizer
  prepare_dataset.py    Build or resume a packed corpus version
  inspect_batch.py      Inspect/decode batches and optionally check the LM
  smoke_test_aoe.py      Same-batch Vanilla/AoE full-size forward/backward checks
  _common.py            Shared source/CLI configuration and logging
  __init__.py           Enables python -m scripts.<command> from repository root
tests/
  __init__.py            Offline test package
  data_fixtures.py       Synthetic fixture helpers
  test_config.py        Defaults, validation and immutability tests
  test_parameter_count.py  Component totals and architecture sensitivity
  test_transformer_components.py  Numerical, causal, backward and CUDA tests
  test_experts.py        Expert formula, parameter count and gradient tests
  test_router.py         Routing probabilities, Top-K and router gradient tests
  test_moe.py            Sparse/dense comparison, dispatch, diagnostics and BF16
  test_decoder.py        Pre-Norm/residual structure and block gradients
  test_model.py          Full budget, tying, causal loss, validation and diagnostics
  test_dolma_reader.py   Offline reader, mocked download and manifest tests
  test_tokenizer.py      Train-only tokenizer, persistence and fingerprint tests
  test_data_processing.py Packing reference, budgets, dedup and crash/resume tests
  test_dataset.py        Memmap, workers, labels and data/model integration tests
  test_data_cli.py       CLI help and no-download dry-run tests
  test_aoe_expert.py     Factorized gate, rank validation and cached-z tests
  test_aoe.py            Norm routing, sparse/dense gradients and BF16 tests
  test_aoe_model.py      Full AoE budget, causal loss and independent layers
  test_model_factory.py Factory, both state_dict round trips and data-to-AoE batch
references/
  erc_paper.pdf          Existing primary methodology reference
pyproject.toml          Package metadata, Python requirement and torch dependency
.gitignore              Generated Python/build artifacts to ignore
README.md               Scope, architecture and usage
```

## Environment setup

Use Python 3.10 or newer and PyTorch 2.0 or newer (SDPA is required).
STEP 5 also requires NumPy >=1.24 and Hugging Face tokenizers >=0.20,<1; all
dependencies remain declared in pyproject.toml rather than a duplicate requirements file.
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
Colab A100 has not yet been verified. These are the STEP 4 verification results.

## STEP 5 data pipeline

The public source is [allenai/dolma](https://huggingface.co/datasets/allenai/dolma),
using the official [v1_5-sample URL manifest](https://huggingface.co/datasets/allenai/dolma/blob/main/urls/v1_5-sample.txt).
This is the nearest public source chosen for the small experiment, not a claim
that these are the exact shards used for the paper's `dolmap-v1.5-sample` runs.
The word "sample" does not imply a small download. No real shards were downloaded
or processed for STEP 5 verification.

The pipeline is:

```text
frozen manifest/subset -> local compressed shards -> document SHA-256 split
  -> train-only BPE sample (train once)
  -> frozen BPE + EOS per document -> independent train/validation token streams
  -> 512-token sequences -> little-endian uint16 binary segments
  -> memory mapping -> torch.long input_ids and cloned labels -> LM causal shift
```

Raw shard acquisition is opt-in. Gzip shards are cached compressed, then read
line by line using `gzip.open(..., 'rt')`; they are never fully decompressed into
RAM. Local `.jsonl` and `.json.gz` files use the same document reader. Malformed
JSON, missing/non-string text and empty documents are counted and skipped; corrupt
gzip or unreadable files fail explicitly. Raw source checksums are computed in
bounded chunks. One document and its token IDs may be resident at a time.

### Frozen source and split policy

Selection sorts unique manifest URLs, shuffles them with the supplied seed and
takes at most `--max-shards`. `--source-contains` filters URL paths explicitly;
this is not stratified sampling or a promise of the full Dolma source mixture.
Do not assume the first few shards represent the corpus distribution. Selected
URLs, selection seed, manifest source and SHA-256 are saved in a new selection
JSON; existing selections are not overwritten.

Each text has CRLF/CR normalized to LF and outer whitespace stripped. Its UTF-8
SHA-256 defines content identity. The split hashes `seed:content_sha256` with
SHA-256 and compares against the validation fraction (default 0.01). Split occurs
before tokenization. Identical canonical content always belongs to one split,
even with different document IDs; an on-disk SQLite ledger deduplicates it across
shards. This is exact canonical-content deduplication, not fuzzy near-deduplication.
`audit_document_splits()` checks ledger assignments without loading all hashes
into RAM. Empty validation is reported and never filled with training documents.

Use the same source order, split seed/fraction, tokenizer and validation budget
for Vanilla and ERC experiments. Train and validation caps are independent;
processing continues past the train cap to fill validation. If sources are
exhausted, metadata reports actual counts and `budget_reached=False`. Freeze an
adequately filled validation corpus before comparisons; changing the source plan
while validation is still short can change that incomplete validation corpus.

### Tokenizer policy

The tokenizer is a [Hugging Face tokenizers byte-level BPE trained from an iterator](https://huggingface.co/docs/tokenizers/training_from_memory).
It uses all 256 byte symbols, no prefix space, minimum merge frequency 2, and
special tokens `<pad>`, `<bos>`, `<eos>`, `<unk>` (IDs recorded explicitly, 0-3).
The first `--max-documents` unique **training** documents in the frozen source
order form the fixed training sample. Validation text is never passed to BPE.
SQLite keeps deduplication out of RAM; the BPE trainer itself still needs memory
for its vocabulary/merge statistics. Training-sample content fingerprint, consumed
source checksums, trainer settings and tokenizers version are saved alongside JSON.

Production requires an actual vocabulary of exactly **16000 including specials**;
if the corpus cannot produce that vocabulary, training fails without pretending
it reached 16000. A small fixture can use `--test-only --vocab-size 320`, explicitly
marked in metadata; preprocessing must then opt in with `--allow-test-tokenizer`.
Model compatibility requires exact vocabulary agreement. The saved wrapper offers
encode/decode only, with no add-token or retraining API. Existing tokenizer paths
are not overwritten. Train production once, preserve JSON plus `.meta.json`,
and reuse them for every data budget and model variant.

### Packing, storage and resume

Budgets count actual encoded IDs **including appended EOS**, not characters or
words. At a cap, the last encoded document can be truncated (so its appended EOS
may fall outside the retained prefix). Separate split buffers carry tails across
shards; only the final incomplete sequence is discarded. Metadata separates
encoded, accepted, packed, budget-discarded and tail-discarded tokens. Documents
skipped after a split reaches its cap are counted separately and are not tokenized;
no token count is guessed for their unencoded text. A token cap not divisible by
512 produces fewer packed tokens than the cap, reported exactly.

Packed samples contain no padding or BOS. EOS separates documents, and **causal
attention can cross earlier document boundaries inside a sequence**; there is no
document-boundary mask. Dataset samples return `labels = input_ids.clone()` with
no shift. The model alone shifts once; a full 512-token sequence supplies 511
target positions. Neither Dataset nor DataLoader transfers the corpus to GPU.

Each committed shard produces separate train/validation binary segments and a
checkpoint containing carry buffers, counts, content fingerprints and source
checksums. SQLite rows have shard indices. After interruption, `--resume` removes
uncheckpointed ledger rows and rebuilds that shard's orphaned segment files, so
sequences are not appended twice. Only committed shard work is reused; there is
no mid-shard or partial-HTTP-transfer resume. Keep the raw sources/cache, tokenizer,
`documents.sqlite`, segments and checkpoint. Use one preprocessing writer per
output directory. Resume rejects changes to the source plan, tokenizer or config.

`dataset.json` is published only after completion. It includes the source
provenance, split algorithm/config, tokenizer and corpus fingerprints, actual
processed shards, counts, segment SHA-256/byte lengths, packing config and runtime
versions. Files use `<u2` and become copied `torch.long` tensors on sampling.
Readers validate sizes/metadata; `--verify-checksums` additionally scans binary
checksums and token ranges. Existing dataset directories are refused unless
`--resume` is supplied. To increase the train budget, create a new corpus version
with the same frozen tokenizer and validation configuration; no tokenizer retraining
is needed. Old versions remain intact. Large artifacts under `data/`, checkpoints
and runs are ignored by Git.

### CLI examples (run manually from the repository root)

All four commands support `--help` and Windows/Linux. Inspection defaults to
dry-run: manifest requests and optional HEAD requests only, no shard bodies.
Unknown/missing Content-Length is printed as `unknown`; it is never estimated.

```bash
python -m scripts.inspect_dolma --max-shards 1 --seed 42 --check-size --save-selection data/debug-selection.json
```

Review the URLs and sizes before explicitly permitting shard downloads. The
following commands **have not been executed on real Dolma**. The byte limit bounds
compressed bytes received per invocation, including failed attempts; cached files
are verified and reused. Unknown-length responses are stopped if the limit is
exceeded. A single compressed shard may exceed this example limit and will then
be refused. Retry count and timeout are configurable on the source CLIs.

```bash
python -m scripts.train_tokenizer --selection data/debug-selection.json --allow-download --max-download-bytes 2000000000 --output data/tokenizer/production.json --max-documents 20000
python -m scripts.prepare_dataset --selection data/debug-selection.json --allow-download --max-download-bytes 2000000000 --tokenizer data/tokenizer/production.json --output-dir data/processed/debug --train-max-tokens 1000000 --validation-max-tokens 100000
python -m scripts.inspect_batch --dataset data/processed/debug --tokenizer data/tokenizer/production.json --batch-size 1 --verify-checksums --decode --model-check
```

If tokenizer training reports insufficient vocabulary, supply a larger fixed
training sample before freezing production. A new run needs a new tokenizer path;
never replace the tokenizer used by an existing corpus. For already-local shards,
replace `--selection ... --allow-download ...` with `--inputs path/a.json.gz path/b.jsonl`.
Source order and split options must match across tokenizer and corpus creation.
For a clean pause use `--max-shards-per-run 1`; repeat the identical prepare command
with `--resume` to continue. Increase the download limit only after reviewing sizes.

Suggested experimental caps below are project choices, not paper-validated 52M
training scales. Freeze one adequately large source plan and one 16K tokenizer
for the comparison runs. Use a new output directory for each corpus version.

| Experiment | --train-max-tokens | --validation-max-tokens | Example output |
| --- | ---: | ---: | --- |
| Debug | 1000000 to 5000000 | 100000 | data/processed/debug |
| Pilot | 50000000 | 1000000 | data/processed/pilot-50m |
| Main | 200000000 | 1000000 | data/processed/main-200m |
| Extended | 500000000 | 1000000 | data/processed/extended-500m |

For example (not executed):

```bash
python -m scripts.prepare_dataset --selection data/frozen-selection.json --allow-download --tokenizer data/tokenizer/production.json --output-dir data/processed/pilot-50m --train-max-tokens 50000000 --validation-max-tokens 1000000 --seed 42
```

For Main/Extended change the train cap and output directory only. Supply an
appropriate reviewed download limit if more compressed shards are needed.
Validation DataLoaders never shuffle; training uses a seeded torch.Generator.
Use `num_workers=0` on Windows for debugging, or multiple workers on Linux/Colab.
Two-worker loading and spawned/pickled lazy memmaps are also tested on Windows.
Gradient accumulation can consume these ordinary CPU batches in STEP 6; no
optimizer or model training loop exists in STEP 5.

### Offline verification

```bash
python -m unittest discover -s tests -v
python -m moe_erc.data_smoke
```

To retain synthetic artifacts for inspection, use a fresh path:

```bash
python -m moe_erc.data_smoke --output-dir data/offline-smoke
python -m scripts.inspect_batch --dataset data/offline-smoke/packed --tokenizer data/offline-smoke/tokenizer.json --batch-size 2 --verify-checksums --tiny-model
```

The synthetic smoke fixture deliberately searches for documents already assigned
to each split; it does not reassign training data or represent Dolma proportions.
It trains a **test-only 320-token vocabulary**, packs real 512-token sequences and
uses a small instance of the unchanged LM implementation. The production 52M
architecture remains unchanged.

STEP 5 verification on 2026-10-23 in `ml_env`: **83 tests passed, 0 failed,
0 skipped**, including prior CUDA/BF16 tests, Windows two-worker DataLoader,
sparse/dense tests, source budgets and simulated shard-commit crash recovery.
The full suite used `torch.set_num_threads(2)` as in STEP 4. Added dependencies
were installed with approval; tokenizers version is 0.23.3. All data tests are
offline and remote transfers are mocked.

Standalone offline smoke: input IDs and labels `[2,512]`, both `torch.int64`,
unshifted identical values, token range `[15,319]`; logits `[2,512,320]`, finite
loss **16.337498**, backward **PASS**, no split overlap. Packed counts were
4096 train and 2048 validation tokens. No 16K production tokenizer, real Dolma
transfer, 50M/200M/500M corpus, Colab throughput or A100 data run is claimed as
verified here. No STEP 6 work was performed.

## STEP 5A Autonomy-of-Experts baseline

The implementation follows Section 2/Figure 2, Section 4.1 and Appendix C.1 of
`references/erc_paper.pdf`. AoE replaces the independent linear router with
expert-derived activation norms; ERC remains a future auxiliary loss on Vanilla
MoE, not an architecture alias for AoE.

For each expert, gate projection W_g is factorized into `gate_down_proj: d -> r`
and `gate_up_proj: r -> D`. The separate SwiGLU `up_proj: d -> D` and final
`down_proj: D -> d` retain their roles:

```text
z_i = gate_down_proj_i(x)
scores_i = L2_norm(z_i)                 # before SiLU, for ALL experts
p = softmax(scores over ALL experts)
selected weights, indices = TopK(p)    # no Top-K renormalization
E_i(x, z_i) = down_proj_i(silu(gate_up_proj_i(z_i)) * up_proj_i(x))
y = sum(p_i * E_i(x, cached_z_i) for selected i)
```

All experts compute gate-down exactly once per forward. Only the selected experts
execute gate-up, SwiGLU up and output projections. The cached gate-down activation
is reused for selected tokens without detaching. Unselected experts can still
receive gate-down gradients through full-softmax probabilities; their remaining
projections are not executed and can have None gradients. There is no independent
learned router, random routing, capacity limit or token dropping. No ERC or
load-balancing loss is applied.

### Configuration and shared interfaces

`ModelConfig` remains unchanged. `AoEConfig(ModelConfig)` adds validated positive
integer `aoe_rank=171`. The paper's 3B setup uses d=1536, D=768, r=512; our scaled
choice comes from rounding d*D/(d+D)=170.666... to 171 at d=512,D=256. No other
width, depth or vocabulary is adjusted to force equal counts. The small tests
set their own explicit ranks; the full-size architecture always uses r=171.

```python
from moe_erc import ModelConfig, AoEConfig, count_parameters
from moe_erc.model_factory import create_model

vanilla = create_model("vanilla_moe", ModelConfig())
aoe = create_model("aoe", AoEConfig())
aoe_budget = count_parameters(AoEConfig(), model_type="aoe")
# create_model("aoe", ModelConfig()) also preserves all base settings and adds r=171.
logits = aoe(input_ids)
result = aoe(input_ids, labels=labels, return_routing_info=True)
result.loss.backward()
scores = result.routing_info[0].diagnostics.scores  # [B,T,16], detached logging data
```

`AoEDecoderBlock` inherits the existing block forward; `AoELanguageModel` inherits
the existing model forward, input validation, tied embedding and shifted FP32 CE.
Two class-level construction hooks select the FFN layer and decoder class without
duplicating Attention, RMSNorm, RoPE, residual arithmetic or LM loss. The existing
`layer.moe` attribute is the common FFN slot: it holds SparseMoE in Vanilla and
SparseAoE in AoE. The Vanilla routing implementation and Data Pipeline files are
unchanged. Both architectures support state_dict save/load with tied weights.

`SparseAoE.compute_routing(flat)` exposes attached activations `[N,16,171]`, scores,
probabilities, Top-K indices and weights. Norms and softmax use FP32 for low-precision
activations, preserving FP64 for reference tests. Selected-output accumulation
follows the existing FP32 low-precision policy, with the output cast to input dtype.

Diagnostics are opt-in and per layer. AoE returns the existing counts, assignment
fractions, mean probabilities and Top-K indices plus full `scores`, full
`probabilities` and `mean_activation_norms`. Every logging tensor is detached;
the activations/weights used for forward are not. Full scores/probabilities can
be large, so they are never retained as module attributes or returned by default.
Assignment fractions divide counts by N*K. Mean probabilities/norms are per expert.

### Analytical and actual parameter budgets

| Component | Vanilla MoE | AoE (r=171) |
| --- | ---: | ---: |
| One expert | 393,216 | 393,472 |
| Expert parameters per layer | 6,291,456 | 6,295,552 |
| Independent router per layer | 8,192 | 0 |
| Complete expert layer | 6,299,648 | 6,295,552 |
| Six expert layers | 37,797,888 | 37,773,312 |
| Complete tied-embedding LM | **52,288,000** | **52,263,424** |

The four AoE matrices contribute 87,552 + 43,776 + 131,072 + 131,072 per expert.
Rounding r adds 256 parameters per expert compared with Vanilla; removing six
linear routers saves 49,152, so AoE is **24,576 parameters smaller** overall.
`ParameterCounts.swiglu_experts` includes the factorized gate in AoE; its existing
field name is preserved for compatibility. Tied embeddings are counted once.

```bash
python -m moe_erc                       # Vanilla, unchanged default
python -m moe_erc --model-type aoe
python -m scripts.smoke_test_aoe
python -m scripts.smoke_test_aoe --device cuda --bf16
```

AoE's additional gate-down work scales with token count in training and inference.
At equal token budgets, compute costs differ. Appendix C.1's general extra FFN
FLOPs expression is T*(2*n*d*r + 2*K*D*r - 2*K*D*d). Its simplified expression
2*T*(n-K)*d*r assumes exact parameter-preserving r; rounding to 171 introduces
a small discrepancy, so the exact rounded-r expression should be used for future
compute accounting. Neither expression is a measured throughput result.

Future comparisons must freeze tokenizer/corpus/validation, context, base model
dimensions and evaluation tooling, then control optimizer, schedule, seeds and
token budget. Quality at equal tokens and efficiency at equal compute are separate
evaluations. No speed or quality advantage is inferred from untrained smoke losses.

### Verification (2026-10-23)

Final full STEP 1-5A suite: **101 passed, 0 failed, 0 errors, 0 skipped**
(18 added test methods). Subtest counts are not separately reported by the runner.
Both full-size parameter counts match PyTorch actual counts. CPU and CUDA/BF16
comparison smoke tests pass on the local RTX 4050 Laptop GPU with PyTorch
2.11.0+cu128; both variants use the same `[1,16]` IDs within each invocation and
return logits `[1,16,16000]`, finite CE and finite backward gradients. They print
per-layer selection counts and basic activation/probability statistics. CPU losses
were 473.166870 (Vanilla) and 472.301270 (AoE); CUDA/BF16 losses were 477.190094
and 480.352875. These are technical sanity results on random initialization only.

The suite covers factorization, no post-Top-K renormalization, pre-SiLU norms,
all-expert routing gradients, selected-only continuation, single tokens and empty
experts, gate-down reuse, sparse/dense output and gradient equality, zero-norm
stability, causal LM behavior, independent blocks, BF16, factory selection and
both state_dict round trips. An additional test reads a STEP 5 memmapped batch
into AoE and completes backward. Target A100 execution, throughput, memory scaling
and trained-model quality remain unverified. No Dolma shards were downloaded,
no model weights were trained, and no STEP 6 implementation was added.
