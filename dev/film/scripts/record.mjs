// Record film.html through its hooks: a clip of one shot (ibsFilm.render(k, t)) or the whole film on the voice's timeline
// (ibsFilm.frame(T)), as an MP4, or stills at chosen times.
//
//   node scripts/record.mjs OUT.mp4 --shot ID [--fps N] [--size WxH] [--scale S] [--clean] [--from S] [--to S] [--hold S] [--crf N]
//   node scripts/record.mjs DIR --shot ID --times T1,T2,...                                          stills, DIR/s<ID>_T.png
//   node scripts/record.mjs OUT.mp4 --film [--audio WAV] [--from S] [--to S] ...                     the whole film
//   node scripts/record.mjs DIR --film --times T1,T2,...                                             stills, DIR/film_T.png
//   node scripts/record.mjs OUT.json --film --holds [--fps N] [--min S]                              how long each picture holds
//   node scripts/record.mjs OUT.json --film --events                                                 the film's events, for the score
//   node scripts/record.mjs OUT.srt --film --captions                                                the captions as subtitles, OUT.srt and OUT.vtt
//
// ID is a line id (3.4) or a shot's number counted from 1, as film.html?shot=K. A clip runs from --from (default 0)
// to --to (default the shot's dur) at --fps (default 25) and holds its last frame for --hold seconds (default 0.6).
// With --film the recording steps the film (film_timeline.js, written by scripts/voice.py) from --from (default 0) to
// --to (default its end), with no hold, and --audio muxes the narration in from the same start.
// --holds steps the film at --fps (default 10) through ibsFilm.holds, which finds the last change of each picture,
// writes its rows to OUT.json and prints them, marking each picture that holds still for less than --min seconds
// (default 1.5) before the next replaces it. --events writes the film's events (ibsFilm.events()), which
// scripts/score.py follows, and prints how many fall outside their picture's span. --captions writes the captions as
// the film shows them (ibsFilm.captions(), a sentence at a time), as SubRip and as WebVTT beside it.
// The captions change by sentence (film.html?timed=1); --clean leaves them out (captions=0). --scale S draws the page
// at S device pixels per CSS pixel: --scale 1.5 records the 1280 x 720 layout at 1920 x 1080.
// The script serves the film's folder (the parent of scripts/) itself on a free port. Chrome is CHROME or its default
// install location on Windows; ffmpeg is FFMPEG, or ffmpeg on the PATH. Chrome, the server and Chrome's temporary
// profile are closed and removed whether the recording succeeds or fails. Needs Node 22 or later (global WebSocket).
// Adapted from the PyBADS film's recorder (acerbilab/pybads, branch feat-film, dev/film/scripts/record.mjs), whose
// page has the same hooks.
import { spawn } from "node:child_process";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { dirname, extname, join, normalize, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const pos = [], opt = {};
for (const argv = process.argv.slice(2); argv.length;) { const a = argv.shift(); if (["--clean", "--film", "--holds", "--events", "--captions"].includes(a)) opt[a.slice(2)] = true; else if (a.startsWith("--")) opt[a.slice(2)] = argv.shift(); else pos.push(a); }
if (pos.length !== 1 || !(opt.shot || opt.film)) { console.error("usage: node scripts/record.mjs OUT.mp4|DIR (--shot ID | --film) [--fps N] [--size WxH] [--scale S] [--clean] [--from S] [--to S] [--times T1,T2] [--hold S] [--crf N] [--audio WAV]"); process.exit(2); }
if ((opt.holds || opt.events) && !(opt.film && pos[0].endsWith(".json"))) { console.error("--holds and --events go with --film and write a .json"); process.exit(2); }
if (opt.captions && !(opt.film && pos[0].endsWith(".srt"))) { console.error("--captions goes with --film and writes a .srt"); process.exit(2); }
if (!opt.times && !opt.holds && !opt.events && !opt.captions && !pos[0].endsWith(".mp4")) { console.error(`a recording is written as an MP4: ${pos[0]} does not end in .mp4`); process.exit(2); }
const out = resolve(pos[0]), FPS = Number(opt.fps || 25), [W, H] = (opt.size || "1280x720").split("x").map(Number), SCALE = Number(opt.scale || 1);
const CHROME = process.env.CHROME || "C:/Program Files/Google/Chrome/Application/chrome.exe";
const FFMPEG = process.env.FFMPEG || "ffmpeg";
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const TYPES = { ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".json": "application/json" };
const server = createServer((req, res) => {
  const path = normalize(join(ROOT, decodeURIComponent(new URL(req.url, "http://localhost").pathname)));
  try { if (!path.startsWith(ROOT + sep) || !statSync(path).isFile()) throw new Error(); res.writeHead(200, { "content-type": TYPES[extname(path)] || "application/octet-stream" }).end(readFileSync(path)); }
  catch { res.writeHead(404).end(); }
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const url = `http://127.0.0.1:${server.address().port}/film.html?timed=1&` + (opt.film ? "film=1" : "shot=1") + (opt.clean ? "&captions=0" : "");

const profile = mkdtempSync(join(tmpdir(), "ibs-film-record-"));
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
  await send("Emulation.setDeviceMetricsOverride", { width: W, height: H, deviceScaleFactor: SCALE, mobile: false });
  await send("Page.navigate", { url });
  let ready = false;
  for (let k = 0; k < 600 && !ready; k++) { ready = await evaluate("document.title.startsWith('ready') || document.title.startsWith('ERROR')").catch(() => false); if (!ready) await sleep(100); }
  const title = await evaluate("document.title").catch(() => "?");
  if (!ready || title.startsWith("ERROR")) throw new Error(`${url} did not start: ${title}`);
  // The page waits at most 2.5 s for its fonts, and a fresh profile can take longer to fetch them: load every face the
  // film draws with before the first frame, or the frames fall back to Arial.
  const faces = ['16px "IBM Plex Sans"', '500 16px "IBM Plex Sans"', '16px "IBM Plex Mono"', '600 16px "IBM Plex Sans Condensed"', 'italic 16px "STIX Two Text"'];
  const loaded = await evaluate(`Promise.race([Promise.all(${JSON.stringify(faces)}.map((f) => document.fonts.load(f))).then(() => ${JSON.stringify(faces)}.every((f) => document.fonts.check(f))), new Promise((r) => setTimeout(() => r(false), 20000))])`);
  if (!loaded) throw new Error("the film's fonts did not load within 20 s");
  const shots = await evaluate("JSON.stringify(ibsFilm.shots)").then(JSON.parse);
  let k = -1, id = "film", dur = await evaluate("ibsFilm.seconds");
  if (opt.film && !dur) throw new Error("film.html has no timeline: run scripts/voice.py first");
  if (!opt.film) {
    k = String(opt.shot).includes(".") ? shots.findIndex((s) => s.id === String(opt.shot)) : Number(opt.shot) - 1;
    if (!(k >= 0 && k < shots.length)) throw new Error(`no shot ${opt.shot}: the shots are ${shots.map((s) => s.id).join(", ")}`);
    id = shots[k].id; dur = shots[k].dur;
  }
  const frame = async (t) => {
    await evaluate(opt.film ? `ibsFilm.frame(${t}), true` : `ibsFilm.render(${k}, ${t}), true`);
    const err = await evaluate("document.getElementById('err').textContent");
    if (err) throw new Error(`the page reported, at ${opt.film ? "T" : "shot " + id + ", t"} = ${t}: ${err}`);
    return Buffer.from((await send("Page.captureScreenshot", { format: "png" })).result.data, "base64");
  };

  const started = Date.now();
  if (opt.captions) {
    const cues = await evaluate("JSON.stringify(ibsFilm.captions())").then(JSON.parse);
    const stamp = (t, sep) => { const ms = Math.round(t * 1000), h = Math.floor(ms / 3600000), m = Math.floor(ms / 60000) % 60, s = Math.floor(ms / 1000) % 60; return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}${sep}${String(ms % 1000).padStart(3, "0")}`; };
    mkdirSync(dirname(out), { recursive: true });
    writeFileSync(out, cues.map(([a, b, text], i) => `${i + 1}\n${stamp(a, ",")} --> ${stamp(b, ",")}\n${text}\n`).join("\n"));
    writeFileSync(out.replace(/\.srt$/, ".vtt"), "WEBVTT\n\n" + cues.map(([a, b, text]) => `${stamp(a, ".")} --> ${stamp(b, ".")}\n${text}\n`).join("\n"));
    console.log(`wrote ${cues.length} captions to ${out} and its .vtt`);
  } else if (opt.events) {
    const ev = await evaluate("JSON.stringify(ibsFilm.events())").then(JSON.parse);
    const err = await evaluate("document.getElementById('err').textContent");
    if (err) throw new Error(`the page reported: ${err}`);
    mkdirSync(dirname(out), { recursive: true });
    writeFileSync(out, JSON.stringify(ev) + "\n");
    const kinds = {};
    for (const e of ev.events) kinds[e[1]] = (kinds[e[1]] || 0) + 1;
    console.log(`${ev.events.length} events: ${Object.entries(kinds).map(([k, n]) => `${k} ${n}`).join(", ")}`);
    for (const e of ev.dropped) console.log(`outside its picture: ${e[1]} of line ${e[3]} at ${e[0]} s`);
    console.log(`wrote ${out}`);
  } else if (opt.holds) {
    const rows = await evaluate(`JSON.stringify(ibsFilm.holds(${Number(opt.fps || 10)}))`).then(JSON.parse);
    const err = await evaluate("document.getElementById('err').textContent");
    if (err) throw new Error(`the page reported: ${err}`);
    mkdirSync(dirname(out), { recursive: true });
    writeFileSync(out, JSON.stringify(rows, null, 1) + "\n");
    const min = Number(opt.min ?? 1.5);
    for (const r of rows) console.log(`${r.id.padEnd(4)} on ${r.on.toFixed(2).padStart(6)} s, last change ${r.last.toFixed(2).padStart(6)} s, holds ${r.hold.toFixed(2).padStart(5)} s${r.hold < min ? "   < " + min : ""}`);
    console.log(`wrote ${out} in ${((Date.now() - started) / 1000).toFixed(0)} s`);
  } else if (opt.times) {
    mkdirSync(out, { recursive: true });
    for (const t of opt.times.split(",").map(Number)) writeFileSync(join(out, `${opt.film ? "film" : "s" + id}_${t.toFixed(2)}.png`), await frame(t));
    console.log(`wrote ${opt.times.split(",").length} stills of ${opt.film ? "the film" : "shot " + id} to ${out}`);
  } else {
    const T0 = Number(opt.from || 0), T1 = opt.to !== undefined ? Number(opt.to) : dur, hold = Number(opt.hold ?? (opt.film ? 0 : 0.6));
    const n = Math.max(1, Math.round((T1 - T0) * FPS) + 1), nHold = Math.round(hold * FPS);
    mkdirSync(dirname(out), { recursive: true });
    const audio = opt.audio ? ["-ss", String(T0), "-i", resolve(opt.audio), "-map", "0:v", "-map", "1:a", "-c:a", "aac", "-b:a", "160k", "-shortest"] : [];
    const ff = spawn(FFMPEG, ["-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", String(FPS), "-c:v", "png", "-i", "-", ...audio,
      "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", String(opt.crf || 20), "-preset", "slow", "-movflags", "+faststart", out], { stdio: ["pipe", "inherit", "inherit"] });
    const done = new Promise((r, j) => { ff.on("close", (c) => (c ? j(new Error(`ffmpeg exited with ${c}`)) : r())); ff.on("error", j); });
    done.catch(() => { /* awaited below; handled here so that an early failure does not end the script before its cleanup */ });
    ff.stdin.on("error", () => { /* ffmpeg stopped reading: done says why */ });
    const write = (png) => Promise.race([new Promise((r) => (ff.stdin.write(png) ? r() : ff.stdin.once("drain", r))), done]);
    let last;
    for (let i = 0; i < n; i++) {
      last = await frame(Math.min(T1, T0 + i / FPS)); await write(last);
      if (opt.film && i % 500 === 0) console.log(`frame ${i} of ${n}, T = ${(T0 + i / FPS).toFixed(1)} s (${((Date.now() - started) / 1000).toFixed(0)} s)`);
    }
    for (let i = 0; i < nHold; i++) await write(last);
    ff.stdin.end(); await done;
    console.log(`wrote ${out}: ${opt.film ? "the film" : "shot " + id}, t ${T0} to ${T1} s, ${n + nHold} frames at ${FPS} fps, in ${((Date.now() - started) / 1000).toFixed(0)} s`);
  }
} finally {
  try { sock?.close(); } catch { /* already closed */ }
  server.close();
  if (!chromeError && chrome.exitCode === null) {                    // Chrome holds its profile until it has exited
    const exited = new Promise((r) => chrome.once("exit", r));
    chrome.kill();
    await Promise.race([exited, sleep(5000)]);
  }
  try { rmSync(profile, { recursive: true, force: true, maxRetries: 10, retryDelay: 200 }); } catch { /* still held: a temp folder */ }
}
