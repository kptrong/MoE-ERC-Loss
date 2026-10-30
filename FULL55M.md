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
after the latest persisted optimizer boundary. Retention keeps **at most two
verified heads per model**, including the final checkpoint; final is not an
additional third checkpoint. Saves run at the optimizer boundary first reaching
each 5,000,000-token validation threshold and at the 55,028,736 endpoint, as well
as before/after benchmarks and at a clean session end. The old every-100-update
recovery writes are removed. Validation/benchmark/training protocols are unchanged.

The checkpoint manager finishes staging, fsync, serialization equality, SHA-256
read-back and COMPLETE publication before retention starts. Retention verifies
the two protected heads again before deleting any older owned checkpoint. A sole
valid checkpoint is never pruned. Small metadata is archived in `logs/METHOD/retired`;
logs, metrics, benchmark results, partial/corrupt directories, other runs and
read-only Kaggle Input are never deleted. Failed saves cannot trigger pruning.
Transiently there can be two old heads plus one replacement during publication;
after a successful save/retention there are at most two valid full checkpoints.

Kaggle documents a **20 GB** saved-output limit. CONFIG sets an **18,000,000,000-byte**
budget across ALL `/kaggle/working`, not just checkpoints. Cell 4 checks room for
six retained full states plus one staging state and logs; Cell 5 rechecks before
launching. Both workers use one OS lock for staging/retention, reserve at least
1 GiB (or a larger model-derived estimate) before each save, then measure again.
Six full model+AdamW states are expected to be roughly 4 GB; actual sizes/storage
audits are authoritative, not that estimate. Dependency/Hugging Face/plot caches
are placed in `/tmp/moe-erc-cache`, outside saved output. On insufficient space,
the runner stops with the existing resume points intact rather than deleting
the last checkpoint or logs to make room. Unexpected files, abandoned partial
saves and unrelated user writes count towards the budget; no code can guarantee
a global limit against arbitrary concurrent external writes. Cell 8 rechecks
actual total bytes and verifies the retained checkpoints before export.

Official limits: https://www.kaggle.com/docs/notebooks

## Download to Windows and resume from a Private Dataset

1. Wait for Cell 5 to end cleanly (COMPLETED or INCOMPLETE), then run Cells 6–8.
   Save a Notebook Version **with its output** and confirm the files are present
   in its Output tab. Download the complete output from that saved version.
   Downloading only the `.ipynb` does not download checkpoints. Do not run another
   training session while making this export/checksum snapshot.
2. Alternatively, with the Kaggle CLI installed and authenticated on Windows,
   download the exact saved version to a new directory (replace all placeholders):

   ```powershell
   kaggle kernels output OWNER/NOTEBOOK-SLUG/VERSION -p "D:\MoE_Backups\session-01"
   ```

   No `--force` is used. Use Kaggle's normal local credential setup; never put a
   token in the notebook. CLI reference:
   https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md
3. Locate `moe_erc_training/export-*.json` and its matching `.sha256`. Verify the
   manifest checksum and **all** listed files in Windows PowerShell (set the
   precise downloaded manifest path; this only reads files):

   ```powershell
   $ManifestPath = "D:\MoE_Backups\session-01\moe_erc_training\export-XXXXXXXX.json"
   $ManifestPath = (Resolve-Path -LiteralPath $ManifestPath).Path
   $SnapshotRoot = Split-Path -Parent $ManifestPath
   $Expected = (Get-Content -LiteralPath ([IO.Path]::ChangeExtension($ManifestPath, ".sha256")) -Raw).Trim()
   if ((Get-FileHash -LiteralPath $ManifestPath -Algorithm SHA256).Hash -ne $Expected) { throw "Manifest checksum FAIL" }
   $Entries = (Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json).files
   foreach ($Entry in $Entries) {
       $FilePath = [IO.Path]::GetFullPath((Join-Path $SnapshotRoot $Entry.path))
       if (-not $FilePath.StartsWith($SnapshotRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) { throw "Unsafe path" }
       $File = Get-Item -LiteralPath $FilePath -ErrorAction Stop
       if ($File.Length -ne $Entry.bytes) { throw "Size FAIL: $FilePath" }
       if ((Get-FileHash -LiteralPath $FilePath -Algorithm SHA256).Hash -ne $Entry.sha256) { throw "SHA256 FAIL: $FilePath" }
   }
   Write-Host "PASS: verified $($Entries.Count) files"
   ```

4. Kaggle → Datasets → New Dataset → upload the verified `moe_erc_training`
   directory, **preserving its relative directory structure**, and set visibility
   to **Private**. Confirm the Dataset's Files view exposes the checkpoint
   directories (not just an unextracted archive). Include both retained heads
   per model and all four files in each: `checkpoint.pt`, `manifest.json`,
   `manifest.sha256`, `COMPLETE.json`. Also keep `run-identity.json`, receipts,
   logs, metrics, benchmark JSON/sample plans, CSVs, training manifest/indices and
   export checksums. Do not upload just `checkpoint.pt` or flatten directories.
   Private Datasets: https://www.kaggle.com/docs/datasets
5. In this same notebook, Add Input for the new private checkpoint Dataset and
   the original verified tokenized dataset/tokenizer. Keep the **same updated
   notebook's SOURCE_COMMIT and runtime dependencies**. Restart the kernel and
   rerun Cells 1–4, then Cell 5. `RESUME_ROOTS=["/kaggle/input"]` discovers the
   latest valid head independently for each model, checks identity and SHA, loads
   directly from read-only Input, and resumes without copying `.pt` into working.
   Source/runtime/config mismatches are fatal; never bypass identity checks.
6. Verify the worker console log contains `restore` with all immediate equality
   checks PASS and the expected saved step/cursor. Subsequent updates continue
   that cursor and token count. A finished model has no further training updates.
   Logs/events and benchmark metadata carry forward from the checkpoint/Input.
   Save/download a new output snapshot after every session. A Private Dataset
   exists durably only after its upload/version has completed successfully.

When replacing the older notebook before the first Cell 5 run, restart the kernel
and rerun Cells 1–4 with the updated notebook. The old kernel's imported runner
does not automatically acquire these fixes. Checkpoints from a different source
commit deliberately fail identity validation; this patch does not migrate them.

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
