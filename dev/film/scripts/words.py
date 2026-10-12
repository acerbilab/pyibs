"""Time each word of the voiced lines, for the cues of film.html.

    python -u scripts/words.py V [--model small.en]

Transcribes each clip V/voice/<line id>.wav that scripts/voice.py wrote with faster-whisper, with word timestamps,
and matches Whisper's words to the words of the line as film.html has it (its `line` in storyboard/shots.json).
Writes film_words.js (window.IBS_WORDS): for each line, a list of [character offset in the line, seconds into the
take] for each word that matched. film.html places a cue between the matched words around it, so a word that Whisper
writes otherwise (a respelled name, a number in digits) falls between its neighbours. Prints, for each line, how many
of its words matched.

The clips change only when a line is voiced again, after which this runs again. Needs faster-whisper, soundfile and
SciPy; faster-whisper fetches its model into HF_HOME.
"""
import argparse
import json
import re
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

FILM = Path(__file__).resolve().parent.parent  # dev/film
WORD = re.compile(r"[A-Za-z0-9’']+")


def norm(w):
    return re.sub(r"[^a-z0-9]", "", w.lower())


def line_words(line):
    """The line's words, split at hyphens, each with its character offset."""
    return [
        (m.start(), norm(m.group()))
        for m in WORD.finditer(line)
        if norm(m.group())
    ]


def heard_words(segments):
    """Whisper's words, split at hyphens and spaces, each with its start; a word split in n parts spreads them over
    its own span."""
    out = []
    for seg in segments:
        for w in seg.words or []:
            parts = [norm(p) for p in re.split(r"[\s\-–]+", w.word.strip())]
            parts = [p for p in parts if p]
            for j, p in enumerate(parts):
                out.append((p, w.start + (w.end - w.start) * j / len(parts)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("V", help="the voice's folder, with voice/<line id>.wav")
    ap.add_argument("--model", default="small.en")
    args = ap.parse_args()

    from faster_whisper import WhisperModel

    shots = json.loads(
        (FILM / "storyboard" / "shots.json").read_text(encoding="utf-8")
    )
    lines = {s["id"]: s["line"] for s in shots if s.get("line")}
    model = WhisperModel(args.model, device="cpu", compute_type="int8")
    voice = Path(args.V) / "voice"
    out = {}
    for lid, text in lines.items():
        clip = voice / f"{lid}.wav"
        if not clip.exists():
            raise SystemExit(f"no take of line {lid}: {clip}")
        audio, sr = sf.read(clip, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = resample_poly(audio, 16000, sr).astype(np.float32)
        segments, _ = model.transcribe(
            audio, language="en", beam_size=5, word_timestamps=True
        )
        heard = heard_words(list(segments))
        mine = line_words(text)
        sm = SequenceMatcher(
            a=[w for _, w in mine], b=[w for w, _ in heard], autojunk=False
        )
        anchors = []
        for a, b, n in sm.get_matching_blocks():
            for j in range(n):
                anchors.append([mine[a + j][0], round(heard[b + j][1], 3)])
        anchors.sort()
        # Keep the anchors in time order: a word matched out of order would make a cue run backwards.
        kept = []
        for c, s in anchors:
            if not kept or s > kept[-1][1]:
                kept.append([c, s])
        out[lid] = kept
        print(f"{lid}: {len(kept)} of {len(mine)} words matched", flush=True)
    js = "// Written by scripts/words.py from the takes' word times; do not edit by hand.\n"
    js += (
        "window.IBS_WORDS = " + json.dumps(out, separators=(",", ":")) + ";\n"
    )
    (FILM / "film_words.js").write_text(js, encoding="utf-8")
    print(f"wrote {FILM / 'film_words.js'}", flush=True)


if __name__ == "__main__":
    main()
