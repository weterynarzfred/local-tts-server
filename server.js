const http = require("http");
const fs = require("fs");
const path = require("path");
const { execFile } = require("child_process");

const HOST = process.env.HOST || "0.0.0.0";
const PORT = Number(process.env.PORT) || 3776;
const ROOT_DIR = __dirname;
const OUTPUT_DIR = path.join(ROOT_DIR, "output");
const PAGE_PATH = path.join(ROOT_DIR, "index.html");
const VENV_PYTHON = path.join(ROOT_DIR, ".venv", "Scripts", "python.exe");
const PYTHON_CMD = fs.existsSync(VENV_PYTHON) ? VENV_PYTHON : "python";

function sendText(res, status, body) {
  res.writeHead(status, { "Content-Type": "text/plain; charset=utf-8" });
  res.end(body);
}

function sendJson(res, status, body) {
  res.writeHead(status, { "Content-Type": "application/json; charset=utf-8" });
  res.end(JSON.stringify(body));
}

function parseRequestBody(req, onDone) {
  let body = "";
  req.on("data", (chunk) => {
    body += chunk;
    if (body.length > 1_000_000) {
      req.destroy(new Error("Request body too large"));
    }
  });
  req.on("end", () => onDone(null, new URLSearchParams(body)));
  req.on("error", (err) => onDone(err));
}

function findLatestMp3() {
  if (!fs.existsSync(OUTPUT_DIR)) {
    return null;
  }
  const entries = fs
    .readdirSync(OUTPUT_DIR, { withFileTypes: true })
    .filter((entry) => entry.isFile() && entry.name.toLowerCase().endsWith(".mp3"))
    .map((entry) => {
      const fullPath = path.join(OUTPUT_DIR, entry.name);
      return { fullPath, mtimeMs: fs.statSync(fullPath).mtimeMs };
    })
    .sort((a, b) => b.mtimeMs - a.mtimeMs);
  return entries.length ? entries[0].fullPath : null;
}

function runTts(text, exaggeration, audioPrompt, callback) {
  const args = ["tts.py", "--text", "-", "--exaggeration", String(exaggeration)];
  if (audioPrompt) {
    args.push("--audio-prompt", audioPrompt);
  }
  try {
    const child = execFile(
      PYTHON_CMD,
      args,
      { cwd: ROOT_DIR, maxBuffer: 10 * 1024 * 1024 },
      (error, stdout, stderr) => {
        if (error) {
          callback(
            new Error(
              `Failed to run tts.py.\n${(stdout || "").trim()}\n${(stderr || "").trim()}`.trim(),
            ),
          );
          return;
        }

        const merged = `${stdout || ""}\n${stderr || ""}`;
        const match = merged.match(/Saved:\s*(.+)/);
        let filePath = null;
        if (match && match[1]) {
          filePath = path.resolve(ROOT_DIR, match[1].trim());
        }
        if (!filePath || !fs.existsSync(filePath)) {
          filePath = findLatestMp3();
        }
        if (!filePath || !fs.existsSync(filePath)) {
          callback(new Error("tts.py finished, but no output MP3 was found."));
          return;
        }

        callback(null, filePath);
      },
    );

    if (child.stdin) {
      child.stdin.on("error", () => {
        // Callback handles process errors; ignore pipe close races.
      });
      child.stdin.end(text, "utf8");
    }
  } catch (error) {
    callback(new Error(`Failed to start tts.py: ${error.message}`));
  }
}

const server = http.createServer((req, res) => {

  const requestUrl = new URL(req.url || "/", `http://${req.headers.host || "localhost"}`);

  if (req.method === "GET" && requestUrl.pathname === "/") {
    fs.readFile(PAGE_PATH, "utf8", (readErr, html) => {
      if (readErr) {
        sendText(res, 500, `Failed to load page: ${readErr.message}`);
        return;
      }
      res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
      res.end(html);
    });
    return;
  }

  if (req.method === "GET" && requestUrl.pathname.startsWith("/audio/")) {
    const fileName = path.basename(requestUrl.pathname.slice(7));
    if (!fileName || !/^[\w.-]+\.mp3$/.test(fileName)) {
      sendText(res, 400, "Invalid filename.");
      return;
    }
    const filePath = path.join(OUTPUT_DIR, fileName);
    if (!filePath.startsWith(OUTPUT_DIR + path.sep)) {
      sendText(res, 403, "Forbidden.");
      return;
    }
    fs.stat(filePath, (statErr, stat) => {
      if (statErr || !stat.isFile()) {
        sendText(res, 404, "File not found.");
        return;
      }
      res.writeHead(200, {
        "Content-Type": "audio/mpeg",
        "Content-Length": stat.size,
      });
      const stream = fs.createReadStream(filePath);
      stream.on("error", (err) => sendText(res, 500, `Failed to read file: ${err.message}`));
      stream.pipe(res);
    });
    return;
  }

  if (req.method === "POST" && requestUrl.pathname === "/synthesize") {
    parseRequestBody(req, (parseErr, params) => {
      if (parseErr) {
        sendJson(res, 400, { error: `Invalid request: ${parseErr.message}` });
        return;
      }

      const text = (params.get("text") || "")
        .trim()
        .replace(/[^\S\n]+/g, " ")
        .replace(/\n{3,}/g, "\n\n")
        .replaceAll(/['']/g, "'")
        .replaceAll(/[""]/g, "\"")
        .replaceAll(/[–—]/g, "-");
      const exaggerationRaw = params.get("exaggeration") || "";
      const exaggeration = Number(exaggerationRaw);
      const audioPrompt = (params.get("audio_prompt") || "").trim();

      if (!text) {
        sendJson(res, 400, { error: "Text is required." });
        return;
      }
      if (!Number.isFinite(exaggeration) || exaggeration < 0 || exaggeration > 1) {
        sendJson(res, 400, { error: "Exaggeration must be a number between 0 and 1." });
        return;
      }

      runTts(text, exaggeration, audioPrompt, (ttsErr, filePath) => {
        if (ttsErr) {
          sendJson(res, 500, { error: ttsErr.message });
          return;
        }

        const fileName = path.basename(filePath);
        sendJson(res, 200, { url: `/audio/${fileName}`, filename: fileName });
      });
    });
    return;
  }

  sendText(res, 404, "Not found");
});

server.listen(PORT, HOST, () => {
  console.log(`Server running at http://${HOST}:${PORT}`);
});
