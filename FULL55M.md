# Full55M Kaggle experiment

Deliverable: `notebooks/MoE_ERC_Full55M_Kaggle_T4x2.ipynb`.
The source runtime is pinned to a real local Git commit. The notebook includes a
Git bundle of that commit, so a public GitHub base clone can fetch it even before
the new local commit is pushed. No untracked source overlay is used in training.
No automatic push is performed.

## Input

Attach the complete **unpacked** verified tokenized dataset, not just `.bin`
files. Include `dataset_manifest.json`, `COMPLETE.json`, `dataset.json`,
`packing_design.json`, every file listed by the dataset manifest (including the
document ledger/checkpoint metadata), and `tokenizer/tokenizer.json` plus its
`.meta.json`. The old 50M pilot manifest/indices are not required for this run.
The existing dataset/tokenizer/source provenance and checksums remain strict.
The document split ledger is audited with the existing read-only API.

The new manifest is PCG64(42).permutation(107478), raw int64 index hash plus a
canonical JSON fingerprint. All three workers reconstruct and check the same
identity. There are 55,028,736 input positions, 54,921,258 targets, 3,359 updates
and a final partial update of 22 sequences. No validation sequence is selected.
The old pilot and its scheduler/checkpoints are not reused as training state.

## Run (five steps)

1. Upload the notebook to Kaggle; Add Input for the complete dataset. For resume,
   also attach a genuinely saved previous notebook output.
2. Select T4 x2 (one GPU runs sequentially), enable Internet for source/dependencies
   and benchmark datasets, and review CONFIG paths.
3. Run All. GPU0 runs Vanilla then ERC; GPU1 runs AoE. TRAIN is real full-target
   training, not a benchmark/smoke simulation.
4. If the session ends cleanly at its configured deadline, save actual Kaggle
   version/output and verify the checkpoint files are present. Attach that output
   next session and Run All with the same runtime/source identity.
5. Download checkpoints with all manifest/sidecar/COMPLETE files and logs/CSVs/PNGs
   through Output/FileLink; verify against the export SHA manifest on Windows.

## Scientific and operational limits

`/kaggle/working` alone is not a durable backup; the notebook cannot automatically
publish a Kaggle version or download to Windows. Hard session kill can lose work
after the latest persisted optimizer boundary. Retention keeps two verified
heads plus final within each newly created full55M namespace; small metadata is
retained. Other runs and read-only Kaggle Input are never deleted.

FP16 retry restores pre-attempt RNG and replays the same uncommitted batch and
ERC proxies with reduced loss scale. Optimizer/scheduler/counters advance only
on a valid update. Forward NaN/Inf or repeated overflow is fatal. Scaler and
committed event history are included in the existing exact-state checkpoint
mechanism. Source/config/runtime mismatches reject resume.

Harness version is 0.4.9. Default task YAMLs and zero-shot settings are used for
arc_challenge, hellaswag, boolq, winogrande and sciq. The custom TemplateLM uses
the project's own BPE tokenizer and conditional continuation likelihood, with
FP32 log-softmax; long contexts are left truncated, long continuations slide.
Task document hashes and fixed sample IDs (up to 200) are recorded, compared on
later milestones, and carried by checkpoint event history. No other pretrained
tokenizer/model is loaded. MC accuracy and normalized accuracy remain separate.

Subset benchmarks run at actual boundaries reaching 0/11/22/33/44M and endpoint;
the final checkpoint also evaluates full task evaluation splits. Missing datasets
or failed evaluation never become fake accuracy. Before every benchmark, a full
checkpoint protects training state. Live benchmark availability and the pinned
harness integration must still be verified on Kaggle; local tests do not certify
remote datasets or actual T4 training.

Outputs: `benchmark_curve_subset.csv`, `benchmark_final_full.csv`, `validation.csv`,
`training.csv`, per-task JSON/sample plans, JSONL logs, state checkpoints, result
JSON, PNG figures and `export-*.json/.sha256`. Empty/partial plots are labelled
explicitly and never synthesize missing scores. The fixed-subset curve averages
raw `acc` only when all five tasks exist. AoE and Vanilla LB both use the existing
raw n*sum(f_i*mean_probability_i), but their routing scores/parameterizations
differ; the plots label that distinction.

This is a 52M/55M research experiment, not a claim to replicate the paper's large
model results. Multiple Kaggle sessions may be needed, particularly for GPU0's
sequential models and full benchmarks. No Kaggle training or harness dataset run
has been performed locally.
