"""Share the CareIQ program files over HTTP so another server can pull and build them.

Run in this codespace:

    python3 share/code_share_server.py                  # serves ../Project on port 5000
    python3 share/code_share_server.py --port 5000 --root Project

Endpoints (all but /health need the token):
    GET /health                              is it up (no token)
    GET /manifest?token=...                  every file that would be sent, with size and sha256
    GET /download?token=...                  a zip of the program files (no data/)
    GET /download?token=...&include_data=1   the same plus data/ (for a receiver that has none)

Safety:
    * A random token is generated at start (or set SHARE_TOKEN) and printed once.
      Requests without it are refused with 401.
    * Never sent: .env and any .env.* except .env.example, .git, caches, virtualenvs,
      and everything generated (database/, output/, processed/), so no key, password
      hash or database ever leaves this machine.
    * data/ is sent only when include_data=1 is asked for.
"""

import argparse
import hashlib
import hmac
import io
import os
import secrets
import time
import zipfile

from flask import Flask, Response, abort, jsonify, request

HERE = os.path.dirname(os.path.abspath(__file__))
REPOSITORY_ROOT = os.path.dirname(HERE)

# Folder names never walked into, wherever they appear.
EXCLUDED_DIRECTORIES = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
                        "venv", ".venv", "env", "node_modules", ".streamlit",
                        "database", "output", "processed", ".share_backups"}
# Top-level folders that are only sent on request.
DATA_DIRECTORIES = {"data"}
EXCLUDED_SUFFIXES = (".pyc", ".pyo", ".db", ".sqlite", ".sqlite3", ".pkl", ".log")
ALLOWED_ENV_FILES = {".env.example"}


def is_secret_file(name):
    return (name == ".env" or name.startswith(".env.")) and name not in ALLOWED_ENV_FILES


def collect_files(root, include_data):
    """Relative paths (forward slashes) of every file that may be shared, sorted."""
    files = []
    for folder, directories, names in os.walk(root):
        relative_folder = os.path.relpath(folder, root)
        top = relative_folder.split(os.sep)[0]
        directories[:] = sorted(
            d for d in directories
            if d not in EXCLUDED_DIRECTORIES
            and not (relative_folder == "." and d in DATA_DIRECTORIES and not include_data))
        if top in DATA_DIRECTORIES and not include_data:
            continue
        for name in sorted(names):
            if is_secret_file(name) or name.endswith(EXCLUDED_SUFFIXES):
                continue
            path = os.path.join(folder, name)
            if os.path.islink(path):
                continue
            files.append(os.path.relpath(path, root).replace(os.sep, "/"))
    return files


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 16), b""):
            digest.update(block)
    return digest.hexdigest()


def create_app(root, token):
    app = Flask(__name__)

    def require_token():
        supplied = request.args.get("token") or request.headers.get("X-Share-Token", "")
        if not hmac.compare_digest(supplied.encode(), token.encode()):
            abort(401, description="A valid token is required.")

    def wants_data():
        return request.args.get("include_data", "0").lower() in ("1", "true", "yes")

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "project": os.path.basename(root)})

    @app.get("/manifest")
    def manifest():
        require_token()
        files = collect_files(root, wants_data())
        return jsonify({"root": os.path.basename(root), "file_count": len(files),
                        "files": [{"path": rel, "bytes": os.path.getsize(os.path.join(root, rel)),
                                   "sha256": sha256_of(os.path.join(root, rel))} for rel in files]})

    @app.get("/download")
    def download():
        require_token()
        files = collect_files(root, wants_data())
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for rel in files:
                archive.write(os.path.join(root, rel), rel)
            archive.writestr("_share_manifest.txt", "\n".join(
                "{0}  {1}".format(sha256_of(os.path.join(root, rel)), rel) for rel in files))
        print("[share] sent {0} files ({1:.1f} KB) to {2}{3}".format(
            len(files), buffer.tell() / 1024.0, request.remote_addr,
            " with data/" if wants_data() else ""))
        return Response(buffer.getvalue(), mimetype="application/zip", headers={
            "Content-Disposition": "attachment; filename=careiq_code_{0}.zip".format(
                time.strftime("%Y%m%d_%H%M%S"))})

    return app


def main():
    parser = argparse.ArgumentParser(description="Share the CareIQ program files.")
    parser.add_argument("--root", default=os.path.join(REPOSITORY_ROOT, "Project"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    arguments = parser.parse_args()

    root = os.path.abspath(arguments.root)
    if not os.path.isdir(root):
        raise SystemExit("No such folder: " + root)
    token = os.environ.get("SHARE_TOKEN") or secrets.token_urlsafe(24)

    codespace = os.environ.get("CODESPACE_NAME")
    domain = os.environ.get("GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN", "app.github.dev")
    public = "https://{0}-{1}.{2}".format(codespace, arguments.port, domain) if codespace \
        else "http://<this-host>:{0}".format(arguments.port)

    print("Sharing   : " + root)
    print("Files     : {0} program files ({1} with data/)".format(
        len(collect_files(root, False)), len(collect_files(root, True))))
    print("Token     : " + token)
    print("Base URL  : " + public)
    if codespace:
        print("Make the port reachable from the other server (codespace ports are private):")
        print("    gh codespace ports visibility {0}:public -c {1}".format(arguments.port, codespace))
    print("On the receiver:")
    print("    python3 receive_and_build.py --url {0} --token {1} --target ./Project".format(
        public, token))
    create_app(root, token).run(host=arguments.host, port=arguments.port)


if __name__ == "__main__":
    main()
