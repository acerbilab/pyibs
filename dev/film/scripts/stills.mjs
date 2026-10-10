// Draw one still per shot of film.html in headless Chrome and write the shot list.
//
//   node scripts/stills.mjs [--size WxH] [--quality Q] [--shots 1,2,...]
//
// Writes storyboard/frames/s<ID>.jpg, one per shot, named by the shot's line id (s4.2.jpg), each drawn at the end of
// its line (t = dur), and storyboard/shots.json, the page's own shot list (id, line, caption, note, dur), which
// gen_storyboard.py reads so that the page and the renderer cannot disagree. The script serves the film's folder (the
// parent of scripts/) itself on a free port. Chrome is CHROME or its default install location on Windows. Chrome, the
// server and Chrome's temporary profile are closed and removed whether the run succeeds or fails. Needs Node 22 or later.
import { spawn } from "node:child_process";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { dirname, extname, join, normalize, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const opt = {};
for (const argv = process.argv.slice(2); argv.length;) { const a = argv.shift(); if (a.startsWith("--")) opt[a.slice(2)] = argv.shift(); }
const [W, H] = (opt.size || "1280x720").split("x").map(Number), QUALITY = Number(opt.quality || 90);
const CHROME = process.env.CHROME || "C:/Program Files/Google/Chrome/Application/chrome.exe";
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const TYPES = { ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".json": "application/json" };
const server = createServer((req, res) => {
  const path = normalize(join(ROOT, decodeURIComponent(new URL(req.url, "http://localhost").pathname)));
  try { if (!path.startsWith(ROOT + sep) || !statSync(path).isFile()) throw new Error(); res.writeHead(200, { "content-type": TYPES[extname(path)] || "application/octet-stream" }).end(readFileSync(path)); }
  catch { res.writeHead(404).end(); }
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const url = `http://127.0.0.1:${server.address().port}/film.html?shot=1`;

const profile = mkdtempSync(join(tmpdir(), "ibs-film-stills-"));
const port = 9300 + Math.floor(Math.random() * 600);
const chrome = spawn(CHROME, ["--headless=new", "--hide-scrollbars", `--remote-debugging-port=${port}`, `--user-data-dir=${profile}`, "about:blank"], { stdio: "ignore" });
let sock, chromeError;
chrome.on("error", (e) => { chromeError = e; });
try {
  let target;
  for (let k = 0; k < 100 && !target; k++) {
    if (chromeError) throw new Error(`Chrome (${CHROME}) did not start: ${chromeError.message}`);
    try { target = (await (await fetch(`http://127.0.0.1:${port}/json/list`)).json()).find((t) => t.type === "page"); } catch { /* not listening yet */ }
    if (!target) await sleep(200);
  }
  if (!target) throw new Error(`Chrome (${CHROME}) did not open its DevTools port`);
  sock = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((r, j) => { sock.addEventListener("open", r); sock.addEventListener("error", j); });
  let nextId = 0; const pending = new Map();
  sock.addEventListener("message", (e) => { const m = JSON.parse(e.data); if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); } });
  const send = (method, params = {}) => new Promise((r) => { const id = ++nextId; pending.set(id, r); sock.send(JSON.stringify({ id, method, params })); });
  const evaluate = async (expression) => {
    const m = await send("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true });
    if (m.result?.exceptionDetails) throw new Error(`page: ${m.result.exceptionDetails.exception?.description || expression}`);
    return m.result?.result?.value;
  };
  await send("Emulation.setDeviceMetricsOverride", { width: W, height: H, deviceScaleFactor: 1, mobile: false });
  await send("Page.navigate", { url });
  let ready = false;
  for (let k = 0; k < 600 && !ready; k++) { ready = await evaluate("document.title.startsWith('ready') || document.title.startsWith('ERROR')").catch(() => false); if (!ready) await sleep(100); }
  const title = await evaluate("document.title").catch(() => "?");
  if (!ready || title.startsWith("ERROR")) throw new Error(`${url} did not start: ${title}`);
  const shots = await evaluate("JSON.stringify(ibsFilm.shots)").then(JSON.parse);
  const dir = join(ROOT, "storyboard", "frames"); mkdirSync(dir, { recursive: true });
  writeFileSync(join(ROOT, "storyboard", "shots.json"), JSON.stringify(shots, null, 1));
  const wanted = opt.shots ? new Set(opt.shots.split(",").map(Number)) : null;
  let n = 0;
  for (let k = 0; k < shots.length; k++) {
    if (wanted && !wanted.has(k + 1)) continue;
    await evaluate(`ibsFilm.render(${k}, ${shots[k].dur}), true`);
    await sleep(60);
    const png = (await send("Page.captureScreenshot", { format: "jpeg", quality: QUALITY })).result.data;
    writeFileSync(join(dir, `s${shots[k].id}.jpg`), Buffer.from(png, "base64")); n++;
  }
  const err = await evaluate("document.getElementById('err').textContent");
  if (err) throw new Error(`the page reported: ${err}`);
  console.log(`wrote ${n} stills to ${dir} and storyboard/shots.json (${shots.length} shots)`);
} finally {
  try { sock?.close(); } catch { /* already closed */ }
  server.close();
  if (!chromeError && chrome.exitCode === null) {
    const exited = new Promise((r) => chrome.once("exit", r));
    chrome.kill();
    await Promise.race([exited, sleep(5000)]);
  }
  try { rmSync(profile, { recursive: true, force: true, maxRetries: 10, retryDelay: 200 }); } catch { /* still held: a temp folder */ }
}
