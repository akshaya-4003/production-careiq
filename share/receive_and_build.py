"""Pull the CareIQ program files from the share server, install them, and build everything.

Paste this one file onto the receiving server and run:

    python3 receive_and_build.py --url https://<codespace>-5000.app.github.dev --token <TOKEN>

What it does, in order:
  1. Downloads the program files (a zip) from the share server and checks every
     file against the sender's sha256 manifest.
  2. Refuses any unsafe path in the zip (absolute paths, '..').
  3. Backs up every local file it is about to overwrite into
     <target>/.share_backups/<timestamp>.zip.
  4. Overwrites the program files in --target. It never touches data/ when the
     target already has one (data/ is downloaded only when it is missing, or with
     --with-data), and never touches .env.
  5. Installs requirements.txt (skip with --skip-install).
  6. Runs every entry point to generate the outputs, then the checks:
         run_data_generation.py      check the 9 datasets + 6 PDFs
         run_database_setup.py       database/careiq.db + demo accounts
         run_data_preparation.py     processed/*.csv + output/data_quality_report.json
         run_model_training.py       output/models/* + output/model_metrics.json
         run_knowledge_indexing.py   database/chroma_store (offline embedder)
         run_evaluation.py           output/evaluation_report.json
         pytest tests.py / tests_regression.py, Python 3.9 syntax check
  7. Writes <target>/build_report.json with every step's status and duration.

Standard library only, so it runs before anything is installed. Python 3.9+.
"""

import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

MANIFEST_NAME = "_share_manifest.txt"
PROTECTED_TOP_LEVEL = {"data"}          # never overwritten when it already exists
PROTECTED_FILES = {".env"}              # local secrets stay local

BUILD_STEPS = [
    ("check datasets", [sys.executable, "run_data_generation.py"], True),
    ("database setup", [sys.executable, "run_database_setup.py"], True),
    ("data preparation", [sys.executable, "run_data_preparation.py"], True),
    ("model training", [sys.executable, "run_model_training.py"], True),
    ("knowledge indexing", [sys.executable, "run_knowledge_indexing.py", "--backend", "offline"],
     True),
    ("evaluation", [sys.executable, "run_evaluation.py"], False),
]
CHECK_STEPS = [
    ("specification tests", [sys.executable, "-m", "pytest", "tests.py", "-q",
                             "-p", "no:cacheprovider"], False),
    ("regression tests", [sys.executable, "-m", "pytest", "tests_regression.py", "-q",
                          "-p", "no:cacheprovider"], False),
    ("python 3.9 check", [sys.executable, "scripts/check_python39_syntax.py"], False),
]


def log(message):
    print("[receive] " + message, flush=True)


def download_zip(base_url, token, with_data):
    query = urllib.parse.urlencode({"token": token, "include_data": "1" if with_data else "0"})
    url = base_url.rstrip("/") + "/download?" + query
    request = urllib.request.Request(url, headers={"User-Agent": "careiq-receiver/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            payload = response.read()
            kind = response.headers.get("Content-Type", "")
    except urllib.error.HTTPError as error:
        if error.code == 401:
            raise SystemExit("The share server refused the token (401). Copy it again from the "
                             "sender's console.")
        raise SystemExit("The share server answered HTTP {0}: {1}".format(error.code, error.reason))
    except urllib.error.URLError as error:
        raise SystemExit("Could not reach {0}: {1}. If the sender is a codespace, make port "
                         "public: gh codespace ports visibility 5000:public -c <name>"
                         .format(base_url, error.reason))
    if "zip" not in kind:
        raise SystemExit("Expected a zip but got '{0}'. A codespace port that is still private "
                         "returns a GitHub sign-in page instead.".format(kind))
    return zipfile.ZipFile(io.BytesIO(payload))


def safe_members(archive):
    """Every file member, with unsafe paths refused and the manifest verified."""
    names = [name for name in archive.namelist() if not name.endswith("/")]
    for name in names:
        normalised = os.path.normpath(name)
        if os.path.isabs(name) or normalised.startswith("..") or ":" in name.split("/")[0]:
            raise SystemExit("Refusing unsafe path in the zip: " + name)
    expected = {}
    if MANIFEST_NAME in names:
        for line in archive.read(MANIFEST_NAME).decode("utf-8").splitlines():
            digest, _, path = line.partition("  ")
            expected[path] = digest
    files = [name for name in names if name != MANIFEST_NAME]
    for name in files:
        if expected and hashlib.sha256(archive.read(name)).hexdigest() != expected.get(name):
            raise SystemExit("Checksum mismatch for " + name + "; the download is corrupt.")
    log("{0} files received, every checksum verified".format(len(files)))
    return files


def install_files(archive, files, target, backup):
    os.makedirs(target, exist_ok=True)
    data_present = {top for top in PROTECTED_TOP_LEVEL if os.path.isdir(os.path.join(target, top))}
    to_write, skipped = [], []
    for name in files:
        top = name.split("/")[0]
        if top in data_present or os.path.basename(name) in PROTECTED_FILES:
            skipped.append(name)
        else:
            to_write.append(name)

    overwritten = [name for name in to_write if os.path.isfile(os.path.join(target, name))]
    if backup and overwritten:
        folder = os.path.join(target, ".share_backups")
        os.makedirs(folder, exist_ok=True)
        backup_path = os.path.join(folder, time.strftime("%Y%m%d_%H%M%S") + ".zip")
        with zipfile.ZipFile(backup_path, "w", zipfile.ZIP_DEFLATED) as saved:
            for name in overwritten:
                saved.write(os.path.join(target, name), name)
        log("backed up {0} files that will be overwritten -> {1}".format(len(overwritten),
                                                                       backup_path))

    for name in to_write:
        destination = os.path.join(target, name)
        os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
        with open(destination, "wb") as handle:
            handle.write(archive.read(name))
    log("wrote {0} files ({1} overwritten, {2} new); left untouched: {3}".format(
        len(to_write), len(overwritten), len(to_write) - len(overwritten),
        ", ".join(sorted(data_present | PROTECTED_FILES)) or "nothing"))
    return {"written": len(to_write), "overwritten": len(overwritten), "skipped": len(skipped)}


def run_steps(steps, target, environment, keep_going, results):
    for label, command, required in steps:
        log("running {0}: {1}".format(label, " ".join(command[1:])))
        started = time.time()
        completed = subprocess.run(command, cwd=target, env=environment)
        seconds = round(time.time() - started, 1)
        ok = completed.returncode == 0
        results.append({"step": label, "command": " ".join(command[1:]), "ok": ok,
                        "exit_code": completed.returncode, "seconds": seconds})
        log("{0} {1} in {2}s".format(label, "OK" if ok else "FAILED (exit {0})".format(
            completed.returncode), seconds))
        if not ok and (required or not keep_going):
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Receive and build CareIQ.")
    parser.add_argument("--url", required=True, help="share server base URL")
    parser.add_argument("--token", default=os.environ.get("SHARE_TOKEN", ""),
                        help="token printed by the share server (or set SHARE_TOKEN)")
    parser.add_argument("--target", default="./Project", help="folder to install into")
    parser.add_argument("--with-data", action="store_true",
                        help="download data/ too (automatic when the target has none)")
    parser.add_argument("--skip-install", action="store_true", help="do not pip install")
    parser.add_argument("--skip-tests", action="store_true", help="build only, no checks")
    parser.add_argument("--no-backup", action="store_true", help="do not back up overwritten files")
    parser.add_argument("--keep-going", action="store_true",
                        help="continue past a failing evaluation or check")
    arguments = parser.parse_args()
    if not arguments.token:
        raise SystemExit("A token is required: --token <TOKEN> (printed by the share server).")

    target = os.path.abspath(arguments.target)
    with_data = arguments.with_data or not os.path.isdir(os.path.join(target, "data"))
    log("target folder: " + target)
    log("downloading program files" + (" and data/" if with_data else " (keeping local data/)"))
    archive = download_zip(arguments.url, arguments.token, with_data)
    files = safe_members(archive)
    installed = install_files(archive, files, target, backup=not arguments.no_backup)

    results = []
    environment = dict(os.environ)
    environment.setdefault("CAREIQ_EMBEDDING_BACKEND", "offline")
    environment.setdefault("PYTHONUNBUFFERED", "1")
    ok = True
    if not arguments.skip_install:
        ok = run_steps([("install requirements", [sys.executable, "-m", "pip", "install", "-q",
                                                   "-r", "requirements.txt"], True)],
                       target, environment, False, results)
    ok = ok and run_steps(BUILD_STEPS, target, environment, arguments.keep_going, results)
    if ok and not arguments.skip_tests:
        ok = run_steps(CHECK_STEPS, target, environment, arguments.keep_going, results) and ok

    report = {"source": arguments.url, "target": target, "installed": installed,
              "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"), "succeeded": ok,
              "steps": results}
    with open(os.path.join(target, "build_report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print()
    for step in results:
        print("  {0:<4} {1:<22} {2:>7}s".format("OK" if step["ok"] else "FAIL", step["step"],
                                              step["seconds"]))
    print()
    if ok:
        print("Build complete. Start the platform from " + target + ":")
        print("    python3 run_api_service.py")
        print("    python3 -m streamlit run run_dashboard.py")
    else:
        print("Build stopped at a failing step; see the output above and build_report.json.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
