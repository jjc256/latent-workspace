"""Interactive labeling UI for SemanticGate data.

Serves a local page at http://127.0.0.1:8000 that walks through
data/gate_data/unlabeled.jsonl. For each record, shows the image + text +
current label + hint and asks you to classify it as:
    0 = VALID            (grammatical English + identifiable image subject)
    1 = LOW_CONFIDENCE   (ambiguous/truncated text, or blurred/dark image)
    2 = INVALID          (non-English, gibberish, or noise/solid-color image)

Keyboard shortcuts:
    1 / 2 / 3     set the label and advance
    right / space advance without changing the label
    left          go back

Each label change writes through to the JSONL immediately. Safe to quit
at any time (Ctrl-C).

Run:
    python -m train.label_gate_ui
"""
from __future__ import annotations

import argparse
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse


DATA_DIR = Path("data/gate_data")
JSONL_PATH = DATA_DIR / "unlabeled.jsonl"
IMG_DIR = (DATA_DIR / "images").resolve()

_records: list[dict] = []
_lock = threading.Lock()


def load_records() -> None:
    global _records
    _records = [json.loads(line) for line in JSONL_PATH.read_text().splitlines() if line.strip()]


def save_records() -> None:
    with _lock:
        tmp = JSONL_PATH.with_suffix(".jsonl.tmp")
        with tmp.open("w") as f:
            for r in _records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        tmp.replace(JSONL_PATH)


INDEX_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>SemanticGate Labeler</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 0; padding: 20px; background: #1a1a1a; color: #eee; }
  .card { max-width: 900px; margin: auto; }
  .progress { height: 4px; background: #333; margin-bottom: 14px; border-radius: 2px; }
  .progress-bar { height: 100%; background: #4a9aff; border-radius: 2px; transition: width 0.2s; }
  .meta { display: flex; gap: 20px; align-items: center; margin: 0 0 12px; font-size: 14px; color: #aaa; }
  .meta .id { font-family: ui-monospace, Menlo, monospace; color: #ccc; }
  .image-wrap { background: #000; border-radius: 6px; padding: 6px; text-align: center; }
  .image { max-width: 100%; max-height: 55vh; display: block; margin: auto; }
  .text { font-size: 18px; line-height: 1.4; margin: 16px 0 6px; padding: 14px 16px; background: #2a2a2a; border-radius: 6px; min-height: 24px; white-space: pre-wrap; }
  .text.empty { color: #777; font-style: italic; }
  .hint { color: #888; font-size: 13px; margin-bottom: 14px; }
  .question { font-size: 15px; color: #bbb; margin: 18px 0 8px; text-align: center; }
  .buttons { display: flex; gap: 10px; justify-content: center; flex-wrap: wrap; }
  button { padding: 14px 18px; font-size: 15px; border: none; border-radius: 6px; cursor: pointer; color: white; min-width: 140px; }
  .b-valid { background: #2d7a2d; }
  .b-lowconf { background: #a0681a; }
  .b-invalid { background: #8a2626; }
  .b-skip { background: #444; }
  button:hover { filter: brightness(1.15); }
  .kb { opacity: 0.7; font-size: 12px; margin-left: 6px; }
  .current { margin-left: auto; font-weight: 600; }
  .current-0 { color: #6ace6a; }
  .current-1 { color: #e6a94a; }
  .current-2 { color: #e66a6a; }
  .current-null { color: #888; }
</style>
</head>
<body>
<div class="card">
  <div class="progress"><div class="progress-bar" id="progress"></div></div>
  <div class="meta">
    <span id="index"></span>
    <span class="id" id="id"></span>
    <span class="current">current: <span id="current"></span></span>
  </div>
  <div class="image-wrap"><img id="img" class="image"></div>
  <div class="text" id="text"></div>
  <div class="hint" id="hint"></div>
  <div class="question">Does this input belong to VALID, LOW_CONFIDENCE, or INVALID?</div>
  <div class="buttons">
    <button class="b-valid" onclick="setLabel(0)">VALID<span class="kb">[1]</span></button>
    <button class="b-lowconf" onclick="setLabel(1)">LOW_CONFIDENCE<span class="kb">[2]</span></button>
    <button class="b-invalid" onclick="setLabel(2)">INVALID<span class="kb">[3]</span></button>
    <button class="b-skip" onclick="advance(-1)">BACK<span class="kb">[&larr;]</span></button>
    <button class="b-skip" onclick="advance(1)">KEEP &amp; NEXT<span class="kb">[&rarr;]</span></button>
  </div>
</div>

<script>
let records = [];
let idx = 0;
const LABEL = { 0: "VALID", 1: "LOW_CONFIDENCE", 2: "INVALID" };

async function init() {
  records = await (await fetch("/api/records")).json();
  const firstUnlabeled = records.findIndex(r => r.label === null);
  idx = firstUnlabeled >= 0 ? firstUnlabeled : 0;
  render();
}

function render() {
  const r = records[idx];
  document.getElementById("img").src = "/img/" + encodeURIComponent(r.id) + ".jpg";
  const t = document.getElementById("text");
  t.textContent = r.text && r.text.length ? r.text : "(empty)";
  t.className = "text" + (r.text && r.text.length ? "" : " empty");
  document.getElementById("hint").textContent = r.hint || "";
  document.getElementById("id").textContent = r.id;
  document.getElementById("index").textContent = (idx + 1) + " / " + records.length;
  const cur = document.getElementById("current");
  cur.textContent = r.label === null ? "unlabeled" : LABEL[r.label];
  cur.className = "current-" + (r.label === null ? "null" : r.label);
  document.getElementById("progress").style.width = ((idx + 1) / records.length * 100) + "%";
}

async function setLabel(v) {
  const r = records[idx];
  if (r.label !== v) {
    r.label = v;
    await fetch("/api/label", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({id: r.id, label: v}),
    });
  }
  advance(1);
}

function advance(delta) {
  idx = Math.max(0, Math.min(records.length - 1, idx + delta));
  render();
}

document.addEventListener("keydown", e => {
  if (e.key === "1") setLabel(0);
  else if (e.key === "2") setLabel(1);
  else if (e.key === "3") setLabel(2);
  else if (e.key === "ArrowRight" || e.key === " ") { e.preventDefault(); advance(1); }
  else if (e.key === "ArrowLeft") advance(-1);
});

init();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            self._send(200, INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/api/records":
            body = json.dumps(_records, ensure_ascii=False).encode("utf-8")
            self._send(200, body, "application/json; charset=utf-8")
        elif path.startswith("/img/"):
            filename = Path(path).name  # prevent traversal
            fp = (IMG_DIR / filename).resolve()
            try:
                fp.relative_to(IMG_DIR)
            except ValueError:
                self.send_error(403)
                return
            if not fp.is_file():
                self.send_error(404)
                return
            self._send(200, fp.read_bytes(), "image/jpeg")
        else:
            self.send_error(404)

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/label":
            length = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(length))
            except json.JSONDecodeError:
                self.send_error(400)
                return
            rec_id = body.get("id")
            label = body.get("label")
            if label not in (0, 1, 2):
                self.send_error(400)
                return
            with _lock:
                for r in _records:
                    if r["id"] == rec_id:
                        r["label"] = label
                        break
                else:
                    self.send_error(404)
                    return
            save_records()
            self._send(200, b'{"ok": true}', "application/json")
        else:
            self.send_error(404)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    load_records()
    server = HTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}"
    print(f"Serving {len(_records)} records at {url}")
    print("Shortcuts: 1=VALID, 2=LOW_CONFIDENCE, 3=INVALID, right/space=next, left=back.")
    print("Ctrl-C to stop (every label change was already saved).")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
