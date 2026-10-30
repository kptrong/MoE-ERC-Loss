"""Package the committed runtime, including a Git bundle for not-yet-pushed commits."""
import base64, hashlib, json, subprocess, tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE="f76587e51965cbd4b6d46e1ec11be25461a72fee"

def build():
    head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT).decode().strip()
    bundle=ROOT/".test-tmp"/"runtime.bundle"
    bundle.parent.mkdir(exist_ok=True)
    subprocess.run(["git","bundle","create",str(bundle),BASE+".."+head,"HEAD"],cwd=ROOT,check=True)
    blob=bundle.read_bytes(); payload=base64.b64encode(blob).decode()
    cells=[]
    def md(s):cells.append({"cell_type":"markdown","metadata":{},"source":s.splitlines(True)})
    def code(s):
        compile(s,"<cell>","exec")
        cells.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":s.splitlines(True)})
    md('''# MoE + ERC — Full55M / Kaggle T4 ×2
Một experiment 52M parameters / 55M input positions, không phải tái lập kết quả 3B/500B.
Ba mô hình cùng full manifest, optimizer/LR và validation. MC scoring dùng lm_eval 0.4.9,
zero-shot, task YAML chuẩn; acc và acc_norm lưu riêng. Fixed subset tối đa 200/task,
seed 42, cùng document hashes cho mọi mốc/model. Full split chỉ ở endpoint.
Không có smoke/preflight notebook phụ. TRAIN chạy budget thật 3.359 updates/model.
Source clone GitHub và checkout commit runtime chính xác. Git bundle nhúng bên dưới
cho phép lấy commit local chưa push, không áp dụng source overlay ẩn.
''')
    md("## 1 CONFIG")
    code(f'''from pathlib import Path
import os,sys,json,time,hashlib,subprocess,base64,concurrent.futures,uuid
REPO_URL="https://github.com/kptrong/MoE-ERC-Loss.git"
SOURCE_COMMIT="{head}"
SOURCE=Path("/kaggle/working/moe_erc_source")
INPUT=Path("/kaggle/input")
OUTPUT=Path("/kaggle/working/moe_erc_training")
DATASET_DIR=None # Optional explicit directory containing dataset_manifest.json.
TOKENIZER_PATH=None # Default: dataset/tokenizer/tokenizer.json
RESUME_ROOTS=[str(INPUT)] # Attach saved output from previous session.
SESSION_HOURS=10.5 # Clean boundary stop; target remains 55.028.736/model.
assert 0<SESSION_HOURS<12
OUTPUT.mkdir(parents=True,exist_ok=True)
''')
    md("## 2 SETUP — exact source commit và dependencies")
    code(f'''from importlib.metadata import version
if any(k=="moe_erc" or k.startswith("moe_erc.") for k in sys.modules):
    raise RuntimeError("Restart kernel before SETUP; resume from saved output.")
constraint=OUTPUT/"torch-constraint.txt"
constraint.write_text("torch=="+version("torch")+"\\n")
subprocess.run([sys.executable,"-m","pip","install","-c",str(constraint),
               "lm_eval==0.4.9","datasets==3.6.0","transformers==4.56.2","matplotlib","tokenizers>=0.20,<1"],check=True)
if not SOURCE.exists():subprocess.run(["git","clone",REPO_URL,str(SOURCE)],check=True)
def git(*args):return subprocess.check_output(["git","-C",str(SOURCE),*args]).decode().strip()
assert git("remote","get-url","origin")==REPO_URL
assert not git("status","--porcelain"),"Dirty source; refusing reset."
encoded="{payload}"
raw=base64.b64decode(encoded)
assert hashlib.sha256(raw).hexdigest()=="{hashlib.sha256(blob).hexdigest()}"
bundle=Path("/kaggle/working/full55m-runtime.bundle");bundle.write_bytes(raw)
if subprocess.run(["git","-C",str(SOURCE),"cat-file","-e",SOURCE_COMMIT],capture_output=True).returncode:
    subprocess.run(["git","-C",str(SOURCE),"fetch",str(bundle),"HEAD"],check=True)
subprocess.run(["git","-C",str(SOURCE),"checkout","--detach",SOURCE_COMMIT],check=True)
assert git("rev-parse","HEAD")==SOURCE_COMMIT
sys.path.insert(0,str(SOURCE))
print("Runtime commit:",SOURCE_COMMIT)
''')
    md("## 3 DATA — toàn bộ train split; không dùng pilot 50M")
    code('''import numpy as np
from moe_erc.pilot_data import load_pilot_artifacts,file_sha256
from moe_erc.pilot_config import PilotConfig
from moe_erc.pilot_checkpoint_core import write_json
if DATASET_DIR is None:
    candidates=[p.parent for p in INPUT.rglob("dataset_manifest.json")
                if json.loads(p.read_text()).get("combined_source_fingerprint")==
                "92acf38d5017658cd71df8449d8edc553d663cd32fcad4350bb55b7c8cc6e144"]
    assert len(candidates)==1,f"Specify DATASET_DIR; matching inputs: {candidates}"
    DATASET_DIR=candidates[0]
DATASET_DIR=Path(DATASET_DIR)
TOKENIZER_PATH=Path(TOKENIZER_PATH) if TOKENIZER_PATH else DATASET_DIR/"tokenizer/tokenizer.json"
assets=load_pilot_artifacts(DATASET_DIR,TOKENIZER_PATH,PilotConfig(bf16=False),full_training=True)
assert (len(assets.plan.indices),assets.plan.total_updates,assets.plan.total_input_tokens,
        assets.plan.total_targets)==(107478,3359,55028736,54921258)
assert len(assets.plan.update_indices(3358))==22
manifest=OUTPUT/"full55m-training-manifest.json"
text=json.dumps(assets.identity["training_manifest"],sort_keys=True)
if manifest.exists():assert manifest.read_text()==text,"Training manifest changed"
else:manifest.write_text(text)
assert file_sha256(manifest)==assets.identity["pilot_manifest_sha256"]
manifest.with_suffix(".sha256").write_text(file_sha256(manifest)+"\\n")
indices=OUTPUT/"full55m-indices.npy"
if indices.exists():assert np.array_equal(np.load(indices,allow_pickle=False),assets.plan.indices)
else:np.save(indices,np.asarray(assets.plan.indices,dtype="<i8"),allow_pickle=False)
indices.with_suffix(".sha256").write_text(file_sha256(indices)+"\\n")
SETTINGS=OUTPUT/"settings.json"
SETTINGS.write_text(json.dumps({"source":str(SOURCE),"output":str(OUTPUT),
    "dataset":str(DATASET_DIR),"tokenizer":str(TOKENIZER_PATH),"resume_roots":RESUME_ROOTS}))
print(assets.identity["training_manifest"])
''')
    md("## 4 GPU — independent FP16 workers")
    code('''import torch
assert torch.cuda.is_available(),"Enable GPU accelerator"
N_GPU=torch.cuda.device_count()
for i in range(N_GPU):print(i,torch.cuda.get_device_name(i),torch.cuda.get_device_properties(i).total_memory/1024**3,"GiB")
LANES=[(0,["vanilla_moe","vanilla_moe_erc"]),(1,["aoe"])] if N_GPU>=2 else [(0,["vanilla_moe","aoe","vanilla_moe_erc"])]
print(LANES)
''')
    md("## 5 TRAIN + EVALUATE — full target, không simulation")
    code('''deadline=time.time()+SESSION_HOURS*3600
def lane(gpu,methods):
    results=[]
    for method in methods:
        if time.time()>=deadline:break
        env=dict(os.environ)
        mapping=env.get("CUDA_VISIBLE_DEVICES","").split(",")
        env["CUDA_VISIBLE_DEVICES"]=mapping[gpu] if len(mapping)>gpu and mapping[gpu] else str(gpu)
        env["PYTHONPATH"]=str(SOURCE)+os.pathsep+env.get("PYTHONPATH","")
        console=OUTPUT/"logs"/(method+"-"+uuid.uuid4().hex[:8]+".log")
        console.parent.mkdir(exist_ok=True)
        print("START",method,"GPU",gpu,console,flush=True)
        with console.open("x") as log:
            p=subprocess.Popen([sys.executable,"-u","-m","moe_erc.full55m",str(SETTINGS),method,str(deadline)],
                               cwd=SOURCE,env=env,stdout=log,stderr=subprocess.STDOUT)
            while p.poll() is None:time.sleep(10)
        result_path=OUTPUT/"results"/(method+".json")
        r=json.loads(result_path.read_text()) if result_path.exists() else {"status":"FAILED","error":"No worker report"}
        if p.returncode:r={**r,"status":"FAILED","returncode":p.returncode}
        results.append({"method":method,**r})
        print(method,r["status"],flush=True)
        if r["status"]!="COMPLETED":break
    return results
with concurrent.futures.ThreadPoolExecutor(max_workers=len(LANES)) as pool:
    jobs=[pool.submit(lane,*l) for l in LANES]
    JOB_RESULTS=[r for f in jobs for r in f.result()]
(OUTPUT/"launcher-results.json").write_text(json.dumps(JOB_RESULTS,indent=2))
print([(r["method"],r["status"]) for r in JOB_RESULTS])
print("INCOMPLETE/FAILED is not completion. Save output before session ends.")
''')
    md("## 6 RESULTS — subset và full results tách riêng")
    code('''from IPython.display import display,FileLink
for m in ("vanilla_moe","aoe","vanilla_moe_erc"):
    p=OUTPUT/"results"/(m+".json")
    display(json.loads(p.read_text()) if p.exists() else {"method":m,"status":"NOT STARTED"})
rs=[json.loads(p.read_text()) for p in (OUTPUT/"results").glob("*.json")] if (OUTPUT/"results").exists() else []
hashes={r["method"]:r.get("initial_hash") for r in rs}
if hashes.get("vanilla_moe") and hashes.get("vanilla_moe_erc"):
    assert hashes["vanilla_moe"]==hashes["vanilla_moe_erc"]
print("Kaggle completion:",len(rs)==3 and all(r["status"]=="COMPLETED" for r in rs))
''')
    md("## 7 PLOTS — chỉ dữ liệu thực có sẵn, không điền điểm giả")
    code('''from moe_erc.full_plots import render
from IPython.display import Image
for p in render(OUTPUT):display(Image(filename=str(p)))
''')
    md("""## 8 EXPORT / RESUME
Save actual Notebook Version **with output**, then confirm checkpoint/log files appear
in Output; or download all checkpoint files below and verify SHA. Attach the saved
output as Input next session, keep source commit/runtime, and Run All: each model
finds its own latest valid checkpoint. No automatic durable upload is claimed.
`/kaggle/working` alone is not a backup. Multiple sessions may be needed.
Harness task docs: https://github.com/EleutherAI/lm-evaluation-harness/tree/v0.4.9/lm_eval/tasks
""")
    code('''files=[]
for p in sorted(OUTPUT.rglob("*")):
    if p.is_file() and not p.name.startswith("export-") and ".partial" not in p.name:
        files.append({"path":p.relative_to(OUTPUT).as_posix(),"bytes":p.stat().st_size,"sha256":file_sha256(p)})
export=OUTPUT/("export-"+uuid.uuid4().hex[:8]+".json")
export.write_text(json.dumps({"files":files,"durable_backup":"USER MUST VERIFY SAVED VERSION/DOWNLOAD"},indent=2))
export.with_suffix(".sha256").write_text(file_sha256(export)+"\\n")
os.chdir("/kaggle/working")
for p in [export,export.with_suffix(".sha256")]:display(FileLink(str(p.relative_to(Path.cwd()))))
for e in files:
    if Path(e["path"]).name in ("checkpoint.pt","manifest.json","manifest.sha256","COMPLETE.json") or e["path"].endswith(".csv"):
        display(FileLink(str((OUTPUT/e["path"]).relative_to(Path.cwd()))))
print("All files listed in export manifest. Download logs/figures via Kaggle Output too.")
''')
    for i,c in enumerate(cells):c["id"]=f"full55m-{i}"
    nb={"nbformat":4,"nbformat_minor":5,"metadata":{"kernelspec":{"name":"python3","display_name":"Python 3","language":"python"}},"cells":cells}
    out=ROOT/"notebooks/MoE_ERC_Full55M_Kaggle_T4x2.ipynb"
    out.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+"\n",encoding="utf-8")
    print(out,head)

if __name__=="__main__":build()
