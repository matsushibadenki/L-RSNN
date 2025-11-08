#!/usr/bin/env python3
# tools/health_check.py
"""
Detailed health check for the RSNN (DI + LangChain) project.
Writes JSON report to outputs/health_report.json

This script is adapted from the rsnn_restructured_C project.
"""
import sys, subprocess, json, time, pathlib, importlib, os

# 実行スクリプト (tools/health_check.py) の親ディレクトリ (プロジェクトルート)
ROOT = pathlib.Path(__file__).resolve().parents[1]
# src をパスに追加
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# 出力先ディレクトリ (config/experiment_params.json に合わせる)
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = OUTPUT_DIR / "health_report.json"

# configから期待される出力ファイル名を取得
CONFIG_PATH = ROOT / "config" / "experiment_params.json"
summary_json_name = "rsnn_summary_di_lc.json"
readme_md_name = "README_rsnn_di_lc.md"
try:
    with open(CONFIG_PATH, 'r') as f:
        config_data = json.load(f)
        summary_json_name = config_data.get("output_paths", {}).get("summary_json", summary_json_name)
        readme_md_name = config_data.get("output_paths", {}).get("readme_md", readme_md_name)
except Exception:
    print(f"Warning: Could not read {CONFIG_PATH}, using default output filenames.")

SUMMARY_JSON_FILE = OUTPUT_DIR / summary_json_name
README_MD_FILE = OUTPUT_DIR / readme_md_name


report = {}
report['timestamp'] = time.time()
report['project_root'] = str(ROOT)
report['python'] = sys.executable
report['python_version'] = sys.version
report['venv'] = bool(sys.prefix != sys.base_prefix)

# 1. パッケージチェック (requirements.txt に基づく)
print("--- 1. Checking Packages ---")
packages = {
    "numpy": "numpy",
    "sklearn": "sklearn",        # scikit-learn
    "matplotlib": "matplotlib",
    "pandas": "pandas",          # matsushibadenki/rsnn の要件
    # DI/LangChain 関連は requirements.txt に記載があれば追加
    "dependency_injector": "dependency_injector",
    "langchain_core": "langchain_core",
}
pkg_info = {}
req_file = ROOT / "requirements.txt"
found_packages = set()

if req_file.exists():
    print(f"Reading {req_file}")
    with open(req_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                pkg_name = line.split('<')[0].split('>')[0].split('=')[0]
                if pkg_name in packages:
                    found_packages.add(pkg_name)

# requirements.txt に記載がなくても、辞書にあるものはチェック
for name, mod in packages.items():
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, "__version__", "unknown")
        pkg_info[name] = {"ok": True, "version": ver, "required": (name in found_packages)}
        print(f"  [OK] {mod} (Version: {ver})")
    except Exception as e:
        pkg_info[name] = {"ok": False, "error": str(e), "required": (name in found_packages)}
        print(f"  [FAIL] {mod}: {e}")
report["packages"] = pkg_info


# 2. スモークテスト (DIコンテナの初期化)
print("\n--- 2. Running Smoke Tests (DI Container Init) ---")
smoke = {}
try:
    from rsnn.di.containers import ApplicationContainer
    
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")

    container = ApplicationContainer()
    container.config.from_json(CONFIG_PATH) # type: ignore[attr-defined]
    
    # サービスがインスタンス化できるか
    service = container.services.experiment_service()
    smoke["di_container_init_ok"] = True
    print("  [OK] DI container initialized and config loaded.")
    smoke["experiment_service_ok"] = True
    print("  [OK] ExperimentService instantiated.")

except Exception as e:
    smoke["di_container_init_ok"] = False
    smoke["di_error"] = str(e)
    print(f"  [FAIL] DI Container init failed: {e}")
report["smoke_tests"] = smoke


# 3. 実験実行チェック (src/main.py)
print(f"\n--- 3. Checking Experiment Run (python src/main.py) ---")
print("  (This may take a moment...)")
experiments = {}
main_script = ROOT / "src" / "main.py"
ran_ok = False
try:
    # 既存の出力ファイルを削除
    if SUMMARY_JSON_FILE.exists():
        SUMMARY_JSON_FILE.unlink()
    if README_MD_FILE.exists():
        README_MD_FILE.unlink()

    env = os.environ.copy()
    # PYTHONPATH に src を追加してサブプロセスを実行
    env['PYTHONPATH'] = str(SRC_PATH) + os.pathsep + env.get('PYTHONPATH', '')
    
    proc = subprocess.run(
        [sys.executable, str(main_script)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        env=env
    )
    
    # 修正: stdout/stderr を常に出力
    if proc.stdout:
        print("  --- stdout from src/main.py ---")
        print(proc.stdout, end='') # print() は改行を追加するため end=''
        print("  -------------------------------")
    if proc.stderr:
        print("  --- stderr from src/main.py ---")
        print(proc.stderr, end='') # print() は改行を追加するため end=''
        print("  -------------------------------")

    if proc.returncode == 0:
        # 修正: 終了コード0でも、ファイルが生成されたか確認
        if SUMMARY_JSON_FILE.exists() and README_MD_FILE.exists():
            experiments["src_main_py"] = {"ran": True, "code": proc.returncode, "files_created": True}
            print("  [OK] src/main.py executed successfully and created output files.")
            ran_ok = True
        else:
            experiments["src_main_py"] = {"ran": False, "code": proc.returncode, "files_created": False, "error": "Process exited 0 but output files are missing."}
            print(f"  [FAIL] src/main.py exited 0, but output files are missing.")
            print(f"  Expected: {SUMMARY_JSON_FILE}")
            print(f"  Expected: {README_MD_FILE}")

    else:
        experiments["src_main_py"] = {"ran": False, "code": proc.returncode, "stderr": proc.stderr[-500:]}
        print(f"  [FAIL] src/main.py failed (Code: {proc.returncode}). See stderr above.")

except subprocess.TimeoutExpired as e:
    experiments["src_main_py"] = {"ran": False, "code": -1, "errmsg": "TimeoutExpired (120s)"}
    print("  [FAIL] src/main.py timed out (120s).")
except Exception as e:
    experiments["src_main_py"] = {"ran": False, "code": -1, "errmsg": str(e)}
    print(f"  [FAIL] Failed to run src/main.py: {e}")
report["experiments"] = experiments


# 4. 出力ファイルチェック (最終確認)
print("\n--- 4. Checking Outputs ---")
report["outputs"] = []
if SUMMARY_JSON_FILE.exists():
    report["outputs"].append({"name": SUMMARY_JSON_FILE.name, "size": SUMMARY_JSON_FILE.stat().st_size, "mtime": SUMMARY_JSON_FILE.stat().st_mtime})
    print(f"  [OK] Found output: {SUMMARY_JSON_FILE.name}")
else:
    report["outputs"].append({"name": SUMMARY_JSON_FILE.name, "found": False})
    print(f"  [WARN] Output file not found: {SUMMARY_JSON_FILE.name}")

if README_MD_FILE.exists():
    report["outputs"].append({"name": README_MD_FILE.name, "size": README_MD_FILE.stat().st_size, "mtime": README_MD_FILE.stat().st_mtime})
    print(f"  [OK] Found output: {README_MD_FILE.name}")
else:
    report["outputs"].append({"name": README_MD_FILE.name, "found": False})
    print(f"  [WARN] Output file not found: {README_MD_FILE.name}")


# レポート書き込み
with OUT_JSON.open("w") as f:
    json.dump(report, f, indent=2)
print(f"\nHealth check complete. Report saved to: {OUT_JSON}")

# 最終的な終了コード
if not all(p.get("ok", False) for p in pkg_info.values()) or \
   not smoke.get("di_container_init_ok", False) or \
   not ran_ok:
    print("\nHealth check FAILED one or more critical steps.")
    sys.exit(1)