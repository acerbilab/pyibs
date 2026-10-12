"""Write storyboard/storyboard.html: one still per line of the script, with the line and what moves during it.

The lines, captions and notes come from storyboard/shots.json, which scripts/stills.mjs writes from the shot list of
film.html, so the page and the renderer cannot disagree. The previous round's lines, kept in
storyboard/lines-round8.json under that round's line ids, appear struck through under the lines that changed; MOVED
maps a line that changed its id to its previous one. Run from dev/film/:

    python scripts/gen_storyboard.py

The newest animatic, storyboard/animatic/animatic_v<N>.mp4, plays at the top of the page. A line with a clip in
storyboard/motion/s<ID>.mp4, which scripts/record.mjs records, plays it in place of its still,
with the still as the poster. The page is a fragment for the artifact host (no <html>, <head> or <body>) that refers
to its stills and clips by relative paths; a browser opens the folder behind python -m http.server.
"""
import html
import json
from pathlib import Path

shots = json.loads(Path("storyboard/shots.json").read_text(encoding="utf-8"))
assert len(shots) == 28, len(shots)
previous = json.loads(
    Path("storyboard/lines-round8.json").read_text(encoding="utf-8")
)
MOVED = {}  # line id now: its id in the previous round, for a line that moved

chars = sum(len(s["line"]) for s in shots)
animatics = sorted(
    Path("storyboard/animatic").glob("animatic_v*.mp4"),
    key=lambda f: int(f.stem.split("_v")[1]),
)
animatic = animatics[-1].name if animatics else None
chars_before = sum(len(v) for v in previous.values())
secs = lambda n, rate: round(n / rate)  # noqa: E731
mmss = lambda s: f"{s // 60}:{s % 60:02d}"  # noqa: E731
length = f"{chars:,} characters of narration, about {mmss(secs(chars, 13.3))} to {mmss(secs(chars, 12.5))} of speech"

SCENES = {
    "1": (
        "The log-likelihood",
        "Generic trials, drawn for the picture as eight of the 300 trials of a toy model kept off screen. The number at the foot of the column is that toy model’s log-likelihood at its best parameter value. The curve comes from the same toy model, one parameter and 300 trials with a lapse rate, the same that frames 3.5 and 6.2 to 6.4 use; the dashed curve of frame 1.4 is the same model with a larger lapse rate.",
    ),
    "2": (
        "Models you can only simulate",
        "The positions and the simulated moves are the recorded run’s: real positions from human-versus-human games, and the model’s own simulations of them. The search tree of frame 2.3 is drawn for the picture.",
    ),
    "3": (
        "The obvious estimate",
        "The rows of simulations in frames 3.1 to 3.3 are the model’s, on the run’s two positions. The averages in frame 3.3 and the clouds of frames 3.4 and 3.6 are computed from the paper’s fixed-sampling estimator, at the probabilities and numbers of simulations shown. The two curves of frame 3.5 are those of the toy model of scene 1, shown in the window around their tops.",
    ),
    "4": (
        "Inverse binomial sampling",
        "The counts on the positions, 3 and 32, are the recorded run’s. The clouds are computed from the paper’s estimators, and their means are the exact expectations.",
    ),
    "5": (
        "What comes with it",
        "The row at the top of frame 5.1 is the recorded run’s count for the surprising position, with the estimate and the error bar that the paper’s formulas give for it. The twenty summed estimates below it are of the toy model of scene 1 at its best parameter value, one IBS repeat each, with error bars from the paper’s variance formula, computed in the page from counts that are not shown. In frame 5.2 the counts are drawn in the page, and the error bar is drawn to narrow as one over the square root of the number of repeats. Frames 5.3 and 5.4 are the recorded PyIBS run: a hundred real positions, one row each, and its estimate.",
    ),
    "6": (
        "The hand-off, and why bias matters",
        "The curves are those of the toy model of scene 1. Each estimate is the average of ten repeats of its estimator, computed in the page. The parameter values that PyBADS and PyVBMC try are chosen for the picture, and the posterior is computed from the toy’s true log-likelihood.",
    ),
    "7": ("The end card", ""),
}
TAGS = {}  # line id: "revised" | "new" | "proposed", reset at every round

blocks, scene = [], None
for s in shots:
    sc = s["id"].split(".")[0]
    if sc != scene:
        if scene is not None:
            blocks.append("</section>")
        scene = sc
        title, sub = SCENES[sc]
        blocks.append(
            f'<section class="scene" id="scene-{sc}"><h2><span class="num">{sc}</span>{html.escape(title)}</h2>'
            + (f'<p class="scene-sub">{html.escape(sub)}</p>' if sub else "")
        )
    line = s["caption"] or s["line"] or "(No narration.)"
    was = previous.get(MOVED.get(s["id"], s["id"]), "")
    was_html = (
        f'<p class="was"><span class="was-label">Round 8</span> <s>{html.escape(was)}</s></p>'
        if was and was != s["line"]
        else ""
    )
    tag = TAGS.get(s["id"])
    tag_html = (
        f'<span class="tag{" prop" if tag == "proposed" else ""}">{tag}</span>'
        if tag
        else ""
    )
    alt = html.escape(s["line"] or "end card")
    if Path(f'storyboard/motion/s{s["id"]}.mp4').exists():
        tag_html += '<span class="tag motion">motion test</span>'
        media = (
            f'<video controls playsinline preload="metadata" poster="frames/s{s["id"]}.jpg" width="1280" height="720" '
            f'aria-label="Motion test of line {s["id"]}: {alt}"><source src="motion/s{s["id"]}.mp4" type="video/mp4"></video>'
        )
    else:
        media = f'<img src="frames/s{s["id"]}.jpg" alt="Frame {s["id"]}: {alt}" width="1280" height="720" loading="lazy">'
    blocks.append(
        f'<article class="frame">{media}'
        f'<div class="text"><div class="meta">LINE {s["id"]}{tag_html}</div>'
        f'<p class="line">{html.escape(line)}</p>{was_html}<p class="note">{html.escape(s["note"])}</p></div></article>'
    )
blocks.append("</section>")

page = f"""<title>PyIBS Film Storyboard</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:ital,wght@0,400;0,500;1,400&display=swap">
<style>
  /* Layout: a single column of frames, each a still beside its line; stacked on narrow screens. One dark look, as the film. */
  :root {{
    color-scheme: dark;
    --bg: #07090a;
    --panel: #0d1213;
    --rule: #1c2624;
    --ink: #e4ece7;
    --dim: #8f9e98;
    --faint: #56645f;
    --green: #5cf2a6;
    --amber: #ffad55;
    --purple: #b78bff;
    --mono: "IBM Plex Mono", ui-monospace, Menlo, Consolas, monospace;
    --cond: "IBM Plex Sans Condensed", "Arial Narrow", Arial, sans-serif;
    --sans: "IBM Plex Sans", "Helvetica Neue", Arial, sans-serif;
  }}
  body {{ background: var(--bg); color: var(--ink); font-family: var(--sans); font-size: 16px; line-height: 1.55; margin: 0; }}
  .wrap {{ max-width: 1200px; margin-inline: auto; padding-inline: 16px; padding-block: 28px 64px; display: grid; gap: 28px; }}
  header {{ display: grid; gap: 10px; }}
  h1 {{ font-family: var(--cond); font-weight: 600; font-size: 28px; letter-spacing: 0.06em; text-transform: uppercase; margin: 0; text-wrap: balance; }}
  h1 span {{ color: var(--green); }}
  .lede {{ margin: 0; color: var(--dim); max-width: 72ch; }}
  .changes {{ border: 1px solid var(--rule); background: var(--panel); padding: 18px 20px; display: grid; gap: 12px; }}
  .changes h2 {{ font-family: var(--cond); font-weight: 600; font-size: 15px; letter-spacing: 0.14em; text-transform: uppercase; color: var(--green); margin: 0; }}
  .changes ul {{ margin: 0; padding-left: 1.1em; display: grid; gap: 10px; max-width: 92ch; }}
  .changes li::marker {{ color: var(--faint); }}
  .changes .was {{ color: var(--dim); text-decoration: line-through; text-decoration-color: var(--faint); }}
  .changes .now {{ color: var(--ink); }}
  .scene {{ display: grid; gap: 18px; border-top: 1px solid var(--rule); padding-top: 22px; }}
  .scene h2 {{ font-family: var(--cond); font-weight: 600; font-size: 21px; letter-spacing: 0.05em; margin: 0; display: flex; gap: 12px; align-items: baseline; text-wrap: balance; }}
  .scene h2 .num {{ font-family: var(--mono); font-size: 15px; color: var(--green); min-width: 1.4em; }}
  .scene-sub {{ margin: -8px 0 0; color: var(--dim); font-size: 14px; max-width: 80ch; }}
  .animatic {{ display: grid; gap: 10px; }}
  .animatic h2 {{ font-family: var(--cond); font-weight: 600; font-size: 15px; letter-spacing: 0.14em; text-transform: uppercase; color: var(--green); margin: 0; }}
  .animatic video {{ width: 100%; max-width: 100%; height: auto; display: block; border: 1px solid var(--rule); background: #000; }}
  .animatic p {{ margin: 0; color: var(--dim); font-size: 14.5px; max-width: 92ch; }}
  .frame {{ display: grid; grid-template-columns: minmax(0, 1.7fr) minmax(0, 1fr); gap: 20px; align-items: start; }}
  .frame img, .frame video {{ width: 100%; max-width: 100%; height: auto; display: block; border: 1px solid var(--rule); background: #000; }}
  .text {{ display: grid; gap: 8px; min-width: 0; }}
  .meta {{ font-family: var(--mono); font-size: 12px; letter-spacing: 0.12em; color: var(--faint); display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }}
  .tag {{ font-family: var(--mono); font-size: 11px; letter-spacing: 0.1em; color: var(--bg); background: var(--amber); padding: 1px 6px; text-transform: uppercase; }}
  .tag.prop {{ background: var(--dim); }}
  .tag.motion {{ background: var(--green); }}
  .line {{ margin: 0; font-size: 19px; line-height: 1.4; color: var(--ink); text-wrap: pretty; }}
  .text .was {{ margin: 0; font-size: 14.5px; color: var(--faint); }}
  .text .was s {{ text-decoration-color: var(--faint); }}
  .was-label {{ font-family: var(--mono); font-size: 11px; letter-spacing: 0.1em; text-transform: uppercase; }}
  .note {{ margin: 0; font-size: 14.5px; color: var(--dim); max-width: 60ch; }}
  footer {{ border-top: 1px solid var(--rule); padding-top: 18px; color: var(--dim); font-size: 14px; display: grid; gap: 6px; max-width: 92ch; }}
  @media (max-width: 760px) {{ .frame {{ grid-template-columns: 1fr; gap: 10px; }} .line {{ font-size: 17px; }} h1 {{ font-size: 23px; }} }}
</style>

<main class="wrap">
  <header>
    <h1><span>PyIBS</span> film: storyboard</h1>
    <p class="lede">Round 9, 10 October 2026. The masters, after a pass on the look of every shot and with the end card held 8 s. Below the film, one still per line, drawn as its picture leaves the screen.</p>
  </header>

  {f'<section class="animatic" aria-label="The film"><h2>The film, captioned master</h2><video controls playsinline preload="metadata" poster="frames/s1.1.jpg" width="1280" height="720"><source src="animatic/{animatic}" type="video/mp4"></video><p>The master with captions, 1920 x 1080 at 25 fps, 4:07 with the end card, at −16 LUFS. The master without captions, the subtitles and the encodes for the web are made beside it. The voice is Kokoro’s am_michael, kept for the final.</p></section>' if animatic else ""}

  <section class="changes" aria-label="What changed in round 9">
    <h2>What changed in round 9</h2>
    <ul>
      <li><strong>The look.</strong> Lines 3.1 to 4.1 sit 50 px lower, in the middle of the frame, where they had crowded its top half. In 1.1 the name <em>likelihood</em> stands beside the trial it names, not across the frame. In 2.3 the faint trees of every search the model could build stay off the board, the formula and the model’s own tree. In 3.4 the key lists the mean under fixed sampling above the true value, as 3.6 and 4.4 do. In 6.3 the label <em>bias</em> moved right of its bracket, clear of an error bar.</li>
      <li><strong>The end card.</strong> It stands over black, without the faint board that sat off-centre behind its title, and holds 8 s, as the PyVBMC film’s card does. It reads PyIBS 1.5, Inverse Binomial Sampling, pip install --upgrade pyibs, acerbilab.org/model-fitting, the IBS paper, and the lab’s credits with “Voice: Kokoro”.</li>
      <li><strong>The masters and their checks.</strong> Both masters are recorded from the reviewed 1280 x 720 layout at 1.5 times its pixels, 11.3 MB without captions and 12.0 MB with them. They are frame-aligned (63.2 dB above the captions at offset 0, against 46.1 dB one frame off). Their sound measures −16.09 LUFS and −1.33 dBTP. Whisper on the master’s sound hears every word of the narration, with the names spelled as it hears them.</li>
      <li><strong>The subtitles.</strong> The .srt and .vtt come from the page’s own captions, so their 53 cues change exactly when the burned-in captions do.</li>
    </ul>
  </section>

  <section class="changes" aria-label="Still open">
    <h2>Still open</h2>
    <ul>
      <li><strong>The publication waits for PyIBS 1.5.</strong> PyPI has only 0.1.0, so the end card’s install line would install it, and the documentation’s address answers once the release deploys it. Then come the film’s release on the model-fitting site, the YouTube upload, and the links from the README and the documentation.</li>
      <li><strong>Length.</strong> The film runs 4:07 with its end card, against 2:48 and 3:00 for the two previous films. The narration has {chars:,} characters.</li>
    </ul>
  </section>

  {"".join(blocks)}

  <footer>
    <div>What is data and what is not: the positions, the simulated moves and the counts of frames 2.2 to 4.3, the count at the top of frame 5.1, the run of frames 5.3 and 5.4 and the board of frame 6.1 are the recorded PyIBS run’s, on real positions from human-versus-human games. The averages of frame 3.3, the clouds of frames 3.4, 3.6 and 4.4, the twenty estimates of frame 5.1, and the curves and estimates of frames 1.2 to 1.4, 3.5, 4.5 and 6.2 to 6.4 are computed from the paper’s formulas and a toy model, as each scene says. The trials of frames 1.1 to 2.2, 4.5 and 5.2 and the search tree of frame 2.3 are drawn for the picture.</div>
    <div>Next: the director’s review of the masters, then the publication, after the release of PyIBS 1.5.</div>
  </footer>
</main>
"""
Path("storyboard/storyboard.html").write_text(page, encoding="utf-8")
print(
    "wrote storyboard/storyboard.html,",
    len(page),
    "characters,",
    len(shots),
    "frames;",
    length,
)
