#!/usr/bin/env python3
"""The film's score, and the narration mixed with it.

    python -u scripts/score.py V

The score has two halves, each in the way of one of the lab's earlier
films. While the film explains the problem (scenes 1 to 3), it is scored as
the PyVBMC film is: pads of detuned saws whose harmony changes with the
lines, a glassy ping for each mark the picture shows, clicks for the
simulations that miss, and swells and low booms at the turns. From
inverse binomial sampling to the end card (scenes 4 to 7), it is the
PyBADS film's tracker-style groove: drums, a pulsing bass, a pad, an
arpeggio and a sparse lead, its synthesizer imported from synth.py, with
taps for the misses and bells for the matches.

The piece is in A minor. In scene 3 the bias is a rising inner voice, the
fifth of the chord raised a semitone at a time (Am, F/A, Am6, Am7, then the
same on D). The scene ends on an augmented chord of E, which holds into
line 4.1 and resolves into A minor as the groove comes in on "turns the
obvious approach around". The terms of line 4.3 play as the harmonic
series of A2 over a drone on A. Term k is the k-th harmonic at an
amplitude of 1/k, so that their sum sounds as a sawtooth wave. "This
estimate is unbiased" arrives on F major 7 and the lead enters. The run's
result, line 5.4, holds a dark chord of E under the voice, and a build
leads to the name of PyIBS. The bias returns in line 6.3 with the flat
second (B flat over A). The false conclusion falls into the drone, and
"With IBS" lifts into the end card, which ends on A major.

In the first half every event sounds at the time at which the picture
shows it. In the second, accents move to the nearest thirty-second note of
the beat. The second half runs on a tempo map. Each section starts on a
downbeat at its anchor, the start of a picture or an event, and holds a
fixed number of bars, whose tempo is fitted to its span. The bars were
fitted to the voice of the masters. A new take of the voice moves the
anchors, and the tempi move with them.

V is the folder of a voice's media. Reads V/events.json (``node
scripts/record.mjs V/events.json --film --events``) and V/narration.wav
(scripts/voice.py). Writes into V:

    score.wav       the score as it sits under the voice, ducked
    mix.wav         the narration with the score
    stems/*.wav     the groups of sounds, before the ducking
    score_plan.txt  the first half's chords, and every bar of the second
                    with its time, tempo, chord and patterns and the events
                    that sound in it

``mux.py`` puts mix.wav under the recorded video at -16 LUFS. The first
half's instruments are the PyVBMC film's (docsrc/source/_static/vbmc3d/
scripts/make_score.py of acerbilab/pyvbmc, branch feat-3d-animation). The
second half's machinery, the tempo map, the modes and the mix, is the
PyBADS film's (dev/film/scripts/score.py of acerbilab/pybads, branch
feat-film), and synth.py is that film's copy, unchanged.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import synth as ms
from scipy.signal import butter, fftconvolve, sosfiltfilt

SR = ms.SR = 48000  # the narration's rate

# ══════════════════════════════════════════════════════════════════════════
# The first half: scenes 1 to 3, as the PyVBMC film
# ══════════════════════════════════════════════════════════════════════════

# Chords as MIDI notes; notes below 40 are played as sines (the bass).
VB_CHORDS = {
    "Am": [33, 45, 52],
    "Am9": [33, 45, 57, 60, 64, 67, 71],
    "Fmaj7": [29, 41, 53, 57, 60, 64],
    "Dm9": [38, 50, 53, 57, 60, 64],
    "Cmaj9": [36, 48, 52, 55, 59, 62],
    "G6": [31, 43, 55, 59, 62, 64],
    "E7sus": [28, 40, 52, 57, 59, 62],
    # The bias: the fifth of A minor rising, E, F, F#, G, then that of D
    # minor, A, Bb, B, C.
    "Am5": [33, 45, 57, 60, 64],
    "F/A": [33, 45, 57, 60, 65],
    "Am6": [33, 45, 57, 60, 66],
    "Am7": [33, 45, 57, 60, 67],
    "Dm": [38, 50, 57, 62, 65],
    "Bb/D": [38, 50, 58, 62, 65],
    "Dm6": [38, 50, 59, 62, 65],
    "Dm7": [38, 50, 60, 62, 65],
    "E+": [28, 40, 52, 56, 60, 64],
}
PENTA = [0, 3, 5, 7, 10]  # A C D E G
SCALE = [69 + o + d for o in (0, 12) for d in PENTA]  # A4 to G6


def vb_plan(ev, first, t_end):
    """The first half's harmony: (time, chord or None, gain, cutoff), on
    the lines and the events. None is silence."""
    L = ev["lines"]
    at = lambda line, off=0.0: L[line][0] + off  # noqa: E731
    plan = [
        (0.0, "Am", 0.5, 400),
        (first("bar"), "Am9", 0.5, 1000),
        (at("1.2"), "Fmaj7", 0.5, 1200),
        (at("1.3"), "Dm9", 0.5, 1400),
        (first("top"), "Cmaj9", 0.55, 2000),
        (at("1.4"), "Fmaj7", 0.5, 1600),
        (first("post"), "E7sus", 0.55, 2000),
        (at("2.1"), "Am", 0.4, 350),
        (at("2.2"), "Fmaj7", 0.45, 900),
        (first("board"), "Dm9", 0.45, 1100),
        (at("2.3"), "G6", 0.45, 1200),
        (first("trees"), "E7sus", 0.5, 1400),
        (at("2.4"), "Am9", 0.5, 1500),
    ]
    # The simulations of line 2.4 speed up: F, C, G, Am, one chord every
    # nine simulations from the fourth, so that the harmony quickens with
    # them.
    sims = sorted(
        t
        for t, kind, line in first.all
        if line == "2.4" and kind in ("miss", "match")
    )
    for k, t in enumerate(sims[3::9]):
        plan.append(
            (
                t,
                ["Fmaj7", "Cmaj9", "G6", "Am9"][k % 4],
                0.5 + 0.03 * k,
                1600 + 300 * k,
            )
        )
    plan += [
        (at("3.1"), "Am5", 0.45, 1100),
        (at("3.2"), "Dm9", 0.45, 1000),
        (first("neginf"), None, 0.0, 0),  # minus infinity: silence
        (at("3.3"), "Am5", 0.45, 1200),
        (first("same"), "F/A", 0.5, 1300),
        (at("3.4"), "Am6", 0.5, 1400),
        (first("mean"), "Am7", 0.5, 1500),
        (at("3.5"), "Dm", 0.5, 1500),
        (first("addup"), "Bb/D", 0.5, 1600),
        (first("morph"), "Dm6", 0.5, 1700),
        (first("shift"), "Dm7", 0.5, 1800),
        (at("3.6"), "E7sus", 0.5, 1600),
        (first("still"), "E+", 0.55, 2000),
        (t_end, None, 0.0, 0),
    ]
    plan.sort(key=lambda p: p[0])
    return plan


def vb_rng(name):
    return ms.rng(f"vb {name}")


def midi(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def pan(mono, p):
    a = (np.clip(p, -1, 1) + 1) * np.pi / 4
    return np.stack([mono * np.cos(a), mono * np.sin(a)], axis=1)


def put(bus, t0, x):
    """Add the stereo or mono signal x into bus from time t0."""
    i = int(round(t0 * SR))
    if i >= bus.shape[0] or i + len(x) <= 0:
        return
    if x.ndim == 1:
        x = np.stack([x, x], axis=1)
    a, b = max(i, 0), min(i + len(x), bus.shape[0])
    bus[a:b] += x[a - i : b - i]


def vb_pad(notes, dur, cutoff, random, attack=1.0, release=2.2):
    """Detuned sawtooth pairs through a low-pass, with a sine bass."""
    n = int((dur + release) * SR)
    t = np.arange(n) / SR
    out = np.zeros((n, 2))
    for m in notes:
        f = midi(m)
        if m < 40:
            out += 0.45 * np.sin(2 * np.pi * f * t)[:, None]
            continue
        g = 0.5 / (1 + (m - 40) / 24)
        for c, cents in enumerate((-7, 7)):
            ph = f * 2 ** (cents / 1200) * t + random.uniform()
            out[:, c] += g * (2 * (ph % 1) - 1)
    env = np.minimum(1, t / attack) * np.where(
        t > dur, np.exp(-(t - dur) / (release / 4)), 1
    )
    return ms.lowpass(out * env[:, None], cutoff)


def ping(f, amp, tau, partial=2.76, p=0.0):
    """A glassy ping: a sine and an inharmonic partial."""
    t = np.arange(int(6 * tau * SR)) / SR
    s = np.sin(2 * np.pi * f * t) + 0.22 * np.sin(2 * np.pi * partial * f * t)
    s *= np.exp(-t / tau) * np.minimum(1, t / 0.002)
    return pan(amp * s, p)


def pluck(f, amp, tau=0.35, p=0.0):
    t = np.arange(int(5 * tau * SR)) / SR
    ph = 2 * np.pi * f * t
    s = np.sin(ph) + 0.3 * np.sin(2 * ph) + 0.1 * np.sin(3 * ph)
    s *= np.exp(-t / tau) * np.minimum(1, t / 0.003)
    return pan(amp * s, p)


def vb_boom(f, amp, tau=1.8):
    t = np.arange(int(4 * tau * SR)) / SR
    s = np.sin(2 * np.pi * f * t) + 0.4 * np.sin(4 * np.pi * f * t)
    return amp * s * np.exp(-t / tau) * np.minimum(1, t / 0.01)


def vb_swell(dur, amp, random, lo=600, hi=5000):
    """Noise that rises in level and brightness over dur seconds."""
    n = int(dur * SR)
    t = np.linspace(0, 1, n)
    x = random.standard_normal((n, 2))
    dark, bright = ms.lowpass(ms.highpass(x, lo), lo * 3), ms.highpass(
        x, hi / 3
    )
    return (
        amp
        * (dark * (1 - t)[:, None] + bright * t[:, None])
        * (t**2)[:, None]
    )


def click_kernel():
    kt = np.arange(int(0.004 * SR)) / SR
    return ms.highpass(
        vb_rng("click").standard_normal(kt.size) * np.exp(-kt / 0.0012), 1800
    )


def vb_reverb(x, seconds=3.0, wet=0.3, bright=5000):
    n = int(seconds * SR)
    ir = (
        vb_rng("reverb").standard_normal((n, 2))
        * np.exp(-np.arange(n) / (n / 6.5))[:, None]
    )
    ir = ms.lowpass(ir, bright)
    ir /= np.sqrt((ir**2).sum(axis=0, keepdims=True))
    y = np.stack(
        [fftconvolve(x[:, c], ir[:, c])[: len(x)] for c in range(2)], axis=1
    )
    return x * (1 - wet) + y * wet


class FirstOf:
    """The time of the first event of a kind (in a line) in the first half."""

    def __init__(self, cues):
        self.all = [(c["pic"], c["kind"], c["line"]) for c in cues]

    def __call__(self, kind, line=None):
        ts = [
            t
            for t, k, ln in self.all
            if k == kind and (line is None or ln == line)
        ]
        if not ts:
            raise SystemExit(
                f"the first half's plan names an event the film does not have: {kind}"
            )
        return min(ts)


def first_half(ev, cues_a, t_end, n):
    """The first half's music and effects, each n samples long, and its
    plan."""
    music, fx = np.zeros((n, 2)), np.zeros((n, 2))
    random = vb_rng("events")
    plan = vb_plan(ev, FirstOf(cues_a), t_end)
    pads = vb_rng("pads")
    for k, (t0, ch, g, cut) in enumerate(plan[:-1]):
        if ch:
            put(
                music,
                t0,
                g
                * vb_pad(
                    VB_CHORDS[ch], max(plan[k + 1][0] - t0, 0.3), cut, pads
                ),
            )

    # Line 1.3's knob: a soft tone whose pitch follows its place on the
    # axis, over an octave from A4, and whose level follows its speed.
    knob = [(c["pic"], c["v"]) for c in cues_a if c["kind"] == "knob"]
    if len(knob) > 2:
        tk, fk = np.array(knob).T
        tt = np.arange(int(tk[0] * SR), int(tk[-1] * SR)) / SR
        a = np.interp(
            tt, tk, np.clip(np.abs(np.gradient(fk, tk)) / 0.8, 0, 1) ** 0.6
        )
        a = sosfiltfilt(butter(1, 6.0, fs=SR, output="sos"), a)
        ph = (
            2 * np.pi * np.cumsum(midi(69) * 2.0 ** np.interp(tt, tk, fk)) / SR
        )
        put(
            music,
            tk[0],
            0.05 * np.clip(a, 0, 1) * (np.sin(ph) + 0.15 * np.sin(2 * ph)),
        )

    kernel = click_kernel()
    gaps = np.diff([c["pic"] for c in cues_a], prepend=-1.0)

    def clicks(t0, seconds, count, amp, p):
        imp = np.zeros((int((seconds + 0.01) * SR) + 1, 2))
        idx = (random.uniform(0, seconds, count) * SR).astype(int)
        imp[idx] += pan(np.full(count, amp), p)
        put(
            fx,
            t0,
            np.stack(
                [fftconvolve(imp[:, c], kernel)[: len(imp)] for c in (0, 1)],
                axis=1,
            ),
        )

    def stab(t0, amp):  # the bias: E5 and F5 together
        put(fx, t0, ping(midi(76), amp, 0.6) + ping(midi(77), amp, 0.6))

    for i, c in enumerate(cues_a):
        t, kind, v = c["pic"], c["kind"], c["v"]
        fast = gaps[i] < 0.12
        if kind == "knob":
            continue
        elif kind == "miss":
            clicks(t, 0.0, 1, 0.35 if fast else 0.5, random.uniform(-0.5, 0.5))
        elif kind == "match":
            m = SCALE[random.integers(5, 10)]
            put(
                fx,
                t,
                ping(
                    midi(m),
                    *((0.06, 0.16) if fast else (0.13, 0.3)),
                    p=random.uniform(-0.5, 0.5),
                ),
            )
        elif kind == "data":
            put(
                fx,
                t,
                ping(midi(SCALE[int(v) % len(SCALE)]), 0.06, 0.25, p=-0.4),
            )
        elif kind == "bar":
            put(
                fx,
                t,
                ping(
                    midi(SCALE[int(round(v * (len(SCALE) - 1)))]),
                    0.1,
                    0.3,
                    p=-0.2,
                ),
            )
        elif kind in ("trial", "inv", "move"):
            put(fx, t, ping(midi(81), 0.1, 0.6))
        elif kind == "log":
            put(
                fx,
                t,
                pluck(
                    midi(
                        SCALE[int(round(np.clip(1 + v / 2.8, 0, 1) * 9))] - 12
                    ),
                    0.1,
                    0.3,
                    -0.2,
                ),
            )
        elif kind == "sum":
            put(fx, t, ping(midi(81), 0.11, 0.8) + ping(midi(88), 0.07, 0.8))
        elif kind == "top":
            for k, m in enumerate([69, 72, 76, 81, 84, 88]):
                put(
                    music,
                    t + 0.1 * k,
                    ping(midi(m), 0.06, 1.2, 2.0, 0.5 - 0.2 * k),
                )
        elif kind == "other":
            put(fx, t, pluck(midi(76), 0.09))
            put(fx, t + 0.16, pluck(midi(72), 0.08))
        elif kind == "post":
            put(fx, t, vb_swell(1.2, 0.06, random))
            put(fx, t + 1.2, ping(midi(81), 0.08, 1.0))
        elif kind == "drain":
            for k, m in enumerate([88, 84, 81, 76, 72, 69]):
                put(
                    fx, t + 0.1 * k, ping(midi(m), 0.05, 0.6, p=0.4 - 0.15 * k)
                )
        elif kind == "question":
            put(fx, t, ping(midi(76), 0.08, 0.5))
            put(fx, t + 0.2, ping(midi(83), 0.08, 0.9))
        elif kind == "pieces":
            for _ in range(10):
                f = midi(SCALE[random.integers(5)] - 12)
                put(
                    fx,
                    t + v * random.random(),
                    pluck(f, 0.05, 0.25, random.uniform(-0.5, 0.5)),
                )
        elif kind in ("board", "node", "token", "fix", "row", "push", "zero"):
            put(
                fx,
                t,
                pluck(
                    midi(SCALE[(int(v) + 2) % len(SCALE)]), 0.09, 0.35, -0.3
                ),
            )
        elif kind == "flicker":
            q = vb_rng(f"flicker {int(v)}")
            put(
                fx,
                t,
                ping(
                    midi(SCALE[q.integers(len(SCALE))]),
                    0.06,
                    0.2,
                    q.uniform(2.0, 3.0),
                    q.uniform(-0.6, 0.6),
                ),
            )
        elif kind == "trees":
            for _ in range(24):
                f = midi(SCALE[random.integers(5, 10)])
                put(
                    fx,
                    t + v * random.random(),
                    ping(f, 0.025, 0.25, p=random.uniform(-0.8, 0.8)),
                )
            put(fx, t, vb_swell(v, 0.04, random))
        elif kind == "unlikely":
            put(fx, t, vb_boom(midi(38), 0.12, 1.2))
        elif kind == "count":
            if v:
                put(fx, t, ping(midi(SCALE[0]), 0.08, 0.4))
                put(fx, t + 0.12, ping(midi(SCALE[2]), 0.07, 0.4))
            else:
                put(fx, t, pluck(midi(57), 0.09, 0.2))
        elif kind == "flog":
            put(fx, t, pluck(midi(64), 0.09, 0.25))
        elif kind == "neginf":  # a swell into a low boom, then silence
            put(fx, t - 1.6, vb_swell(1.6, 0.07, random, 400, 8000))
            put(fx, t, vb_boom(midi(33), 0.25, 2.5))
        elif kind == "same":
            stab(t, 0.09)
        elif kind == "truth":
            put(fx, t, ping(midi(76), 0.09, 0.5))
        elif kind == "mean":  # fixed sampling's mean: a semitone too high
            put(fx, t, ping(midi(77), 0.1, 0.5))
        elif kind == "bias":
            g = float(np.clip(np.log1p(abs(v)) / np.log1p(2.0), 0.25, 1.0))
            stab(t, 0.08 * g)
            put(fx, t, vb_boom(midi(33), 0.1 * g, 1.0))
        elif kind == "still":
            stab(t, 0.11)
            put(fx, t, vb_boom(midi(28), 0.15, 1.6))
        elif kind == "rain":
            seconds, column = v
            clicks(t, seconds, 30, 0.3, (column - 1) * 0.5)
        elif kind in ("addup", "morph"):
            put(fx, t, vb_swell(v, 0.05, random))
        elif kind == "shift":
            put(fx, t, ping(midi(76), 0.08, 0.5))
            put(fx, t + 0.18, ping(midi(77), 0.09, 0.7))
        else:
            raise ValueError(
                f"no sound in the first half for the event {kind!r} of line {c['line']}"
            )
    return music, fx, plan


# ══════════════════════════════════════════════════════════════════════════
# The second half: scenes 4 to 7, as the PyBADS film
# ══════════════════════════════════════════════════════════════════════════

# The sections, in order. Each starts at an anchor of the film and ends where
# the next starts: the start of a line's picture, or an event.
ANCHORS = {
    "intro": ("shot", "4.1"),
    "ibs": ("event", "turn"),  # "turns the obvious approach around"
    "terms": ("shot", "4.3"),
    "unbiased": ("shot", "4.4"),
    "errors": ("shot", "5.1"),
    "rounds": ("shot", "5.3"),
    "result": ("shot", "5.4"),
    "handoff": ("event", "name"),  # "PyIBS works"
    "bias": ("shot", "6.3"),
    "false": ("event", "false"),  # "a false conclusion"
    "withibs": ("event", "withibs"),
    "end": ("shot", "7.1"),
}
# The sections whose beats slow down, by this fraction of the first beat per
# beat; the others keep an even tempo.
RITARDANDO = {"end": 0.06}

# The levels that a bar's mode sets (the PyBADS film's): the pad's level and
# the cutoff of its low-pass, the same for the arpeggio, the bass's level and
# how far its filter opens, and the drone's level. A number holds over the
# bar, a pair goes from the first to the second across it, and a list of
# (fraction of the bar, value) places points within it.
MODES = {
    "intro": dict(pad=0.3, pcut=420, arp=0, acut=500, bass=0, bright=0.2, drone=1.0),
    "pulse": dict(pad=0.55, pcut=1000, arp=0.5, acut=1400, bass=0.55, bright=0.35, drone=0.6),
    "search": dict(pad=0.9, pcut=2000, arp=0.9, acut=2900, bass=1.0, bright=0.85, drone=0),
    "fooled": dict(pad=1.0, pcut=2400, arp=1.05, acut=3300, bass=1.0, bright=1.1, drone=0),
    "break": dict(pad=(1.0, 0), pcut=(2400, 160), arp=(1.0, 0), acut=(2900, 250), bass=(1.0, 0), bright=(1.0, 0.1), drone=(0, 1.0)),
    "poll": dict(pad=0, pcut=300, arp=0, acut=450, bass=0, bright=0.1, drone=1.0),
    "build": dict(pad=(0.35, 1.0), pcut=(500, 2600), arp=0, acut=500, bass=(0.55, 1.0), bright=(0.5, 1.3), drone=0),
    "arrive": dict(pad=1.0, pcut=3000, arp=0, acut=500, bass=0.9, bright=0.75, drone=0),
    "lift": dict(pad=1.0, pcut=2800, arp=1.0, acut=3400, bass=1.0, bright=1.05, drone=0),
    "out": dict(pad=(1.0, 0.55), pcut=(2800, 700), arp=(0.8, 0), acut=(3400, 600), bass=0.9, bright=0.7, drone=0),
    "reveal": dict(pad=1.0, pcut=3400, arp=0, acut=500, bass=0.8, bright=0.6, drone=0.3),
    "calm": dict(pad=0.8, pcut=2400, arp=0.6, acut=2400, bass=0.6, bright=0.5, drone=0),
    "final": dict(pad=1.0, pcut=3000, arp=0, acut=500, bass=0.8, bright=0.6, drone=(0, 0.6)),
}  # fmt: skip

# One row per bar: mode, chord, drums, bass, arpeggio (None: silent), and
# what it changes: "beats" when it has fewer than four, levels of MODES.
# fmt: off
SONG = {
    "intro": [  # 4.1, to "turns around": the first half's augmented chord still sounds
        ("intro", None, None, None, None, dict(beats=3, pad=0, drone=0)),
    ],
    "ibs": [  # 4.1-4.2: to the first match, and the counts
        ("pulse", "Am", "pulse", "hint", None, dict(pad=(0.5, 0.8), pcut=(900, 1800), bass=0.5, bright=0.35, drone=(0.6, 0.0))),
        ("search", "Am", "A", "eighths", "roll", dict(pad=(0.8, 0.9), pcut=(1800, 2000), arp=(0.6, 0.85), acut=(1800, 2700), bright=0.7)),
        ("search", "Fmaj7", "B", "eighths", "roll", {}),
        ("search", "C", "A", "eighths", "roll", {}),
        ("search", "Gsus2", "B", "eighths", "roll", {}),
        ("search", "Am", "A16", "sixteenths", "roll", {}),
    ],
    "terms": [  # 4.3: the harmonic series over the drone, and a build
        ("poll", None, None, None, None, {}),
        ("poll", None, "clock", None, None, {}),
        ("build", "E7sus4", "build", "build", None, {}),
    ],
    "unbiased": [  # 4.4-4.5: the arrival, and the lift
        ("arrive", "Fmaj7", "half", "soft", None, {}),
        ("lift", "C", "A16", "sixteenths", "roll", {}),
        ("lift", "Gsus2", "B16", "sixteenths", "roll", {}),
        ("lift", "Am", "A16", "sixteenths", "roll", {}),
        ("out", "Am", "out", "long", "out", {}),
    ],
    "errors": [  # 5.1-5.2: the error bars, calibrated; the repeats
        ("calm", "Am", None, "pedal", "roll", {}),
        ("calm", "Fmaj7", "pulse", "pedal", "roll", {}),
        ("search", "C", "A", "eighths", "roll", dict(pad=0.8, arp=0.8, bass=0.85)),
        ("search", "Gsus2", "B", "eighths", "roll", dict(pad=0.8, arp=0.8, bass=0.85)),
        ("search", "Am", "A", "eighths", "roll", dict(pad=0.8, arp=0.8, bass=0.85)),
        ("search", "Fmaj7", "B", "eighths", "roll", dict(pad=0.8, arp=0.8, bass=0.85)),
    ],
    "rounds": [  # 5.3: the rounds of the run
        ("search", "C", "A16", "sixteenths", "climb", {}),
        ("search", "Gsus2", "B16", "sixteenths", "climb", {}),
        ("break", "E", "break", "sixteenths", "roll", dict(beats=2)),
    ],
    "result": [  # 5.4, a dark E under the voice; 6.1's first sentence, a build
        ("poll", "E", None, "long", None, dict(pad=(0.45, 0.35), pcut=500, bass=0.6, bright=0.3, drone=0)),
        ("build", "E7sus4", "build1", "build", None, dict(pad=(0.3, 0.6), pcut=(450, 1200), bass=(0.4, 0.7), bright=(0.4, 0.8))),
        ("build", "E7sus4", "pickup2", "build", None, dict(beats=2, pad=(0.6, 1.0), pcut=(1200, 2600), bass=(0.7, 1.0), bright=(0.8, 1.3))),
    ],
    "handoff": [  # 6.1-6.2: the name, and PyBADS and PyVBMC
        ("arrive", "Fmaj7", "half", "soft", None, {}),
        ("lift", "C", "A16", "sixteenths", "roll", {}),
        ("lift", "Gsus2", "B16", "sixteenths", "roll", {}),
        ("lift", "Am", "A16", "sixteenths", "roll", {}),
        ("lift", "Fmaj7", "B16", "sixteenths", "roll", {}),
        ("lift", "Gsus2", "A16", "sixteenths", "roll", {}),
    ],
    "bias": [  # 6.3: noisy estimates, then the bias, with the flat second
        ("search", "Am", "A", "eighths", "roll", {}),
        ("search", "C", "B", "eighths", "roll", {}),
        ("fooled", "Bb/A", "A16", "lean", "climb", dict(arp=(0.95, 1.1), acut=(2900, 3400), bright=(1.0, 1.1))),
        ("fooled", "Bbmaj7#11", "B16", "sixteenths", "climb", dict(arp=(1.1, 1.25), acut=(3400, 3900), bright=(1.1, 1.25))),
        ("break", "Am", "break", "sixteenths", "roll", dict(beats=2)),
    ],
    "false": [  # 6.4, "a false conclusion": the drone alone
        ("poll", None, None, None, None, {}),
        ("poll", None, None, None, None, dict(beats=1)),
    ],
    "withibs": [  # "With IBS, the estimate is only noisy": the lift into the end card
        ("reveal", "Fmaj7", None, "long", None, {}),
        ("lift", "Gsus2", "A16", "sixteenths", "roll", {}),
        ("lift", "E7sus4", "pickup1", "build", None, dict(beats=1)),
    ],
    "end": [  # the end card, on A major
        ("final", "A", None, "long", None, {}),
        ("final", "A", None, None, None, {}),
        ("final", "A", None, None, None, {}),
    ],
}
# fmt: on

# The chords of A minor, in the format of synth.py: the bass's root, the
# pad's voicing, the tones that the arpeggio cycles (None: no arpeggio). F
# major 7 is the arrival, C and G sus2 the lift, B flat over A and B flat
# major 7 (#11) the flat second of the bias, E7sus4 and E the dominant, and A
# major the end.
# fmt: off
CHORDS = {
    "Am": ("A1", ["A3", "E4", "A4", "C5", "E5"], ["A4", "C5", "E5", "A5"]),
    "Fmaj7": ("F2", ["C4", "F4", "A4", "C5", "E5"], ["A4", "C5", "E5", "A5", "C6", "E6"]),
    "C": ("C2", ["C4", "G4", "C5", "E5"], ["C5", "E5", "G5", "C6"]),
    "Gsus2": ("G2", ["D4", "G4", "A4", "D5"], ["D5", "G5", "A5", "D6"]),
    "Bb/A": ("A1", ["D4", "F4", "Bb4", "D5"], ["Bb4", "D5", "F5", "A5", "Bb5", "D6"]),
    "Bbmaj7#11": ("Bb1", ["D4", "F4", "A4", "E5"], ["Bb4", "D5", "E5", "A5", "Bb5", "D6"]),
    "E7sus4": ("E2", ["B3", "E4", "A4", "B4", "D5"], None),
    "E": ("E2", ["B3", "E4", "G#4", "B4"], ["E4", "G#4", "B4", "E5"]),
    "A": ("A1", ["A3", "E4", "A4", "C#5", "E5"], None),
}
# fmt: on

# Patterns that synth.py does not have.
DRUMS = dict(
    ms.DRUMS,
    clock=("................", "................", "2...2...2...2..."),
    build1=("3...4...4...5...", "................", "................"),
    pickup2=("6.7.8.9.", "....5567", "........"),
    pickup1=("....", "..57", "..3."),
)
BASS = dict(ms.BASS, pedal="0.......0.......")
ARP = dict(ms.ARP)

# The lead: section, bar, sixteenth, length in sixteenths, note. All its
# notes are in the minor pentatonic scale of A.
LEAD = [
    ("unbiased", 0, 0, 6, "A4"),
    ("unbiased", 0, 6, 2, "G4"),
    ("unbiased", 0, 8, 7, "E4"),
    ("handoff", 0, 0, 6, "E5"),
    ("handoff", 0, 6, 2, "D5"),
    ("handoff", 0, 8, 7, "C5"),
    ("withibs", 0, 6, 2, "D5"),
    ("withibs", 0, 8, 4, "E5"),
    ("withibs", 0, 12, 4, "G5"),
    ("end", 0, 0, 15, "A5"),
]

# Changes to synth.py's levels, for a score under a voice: the drums, the
# arpeggio and the stab of a miss, which compete with speech, are softer.
LEVEL_DB = dict(
    ms.LEVEL_DB,
    kick=-10.0,
    snare=-11.0,
    hat=-20.0,
    arp=-13.0,
    lead=-17.0,
    miss_stab=-16.0,
    tap=-14.0,
    landing=-21.0,
    tick=-27.0,  # a tick of a fast run
    bloom=-27.0,  # a breath of noise
    fooled=-17.0,  # the held stab of the bias
    term=-17.0,  # the first term of line 4.3; term k is 1/k of it
)
# The score's loudness under the voice, in LU from the voice's own (the
# PyBADS film's setting, chosen there by ear), and how far the first half's
# music sits below the second half's.
SCORE_LU = -17.0
FIRST_HALF_LU = -1.5
# How far the score dips while the voice speaks: the music, and the events
# and effects.
DUCK = {"music": 0.45, "events": 0.25}
MUSIC = ("drums", "bass", "pad", "arp", "lead")
FADE_OUT = 1.6  # seconds, to the end of the film

# Accents of the second half that move to the nearest thirty-second note.
SNAP = {
    "turn", "match", "count", "inv", "zero", "imean", "lit", "sum", "repeat", "result", "name",
    "fit", "bias", "false", "withibs", "estimate", "end", "token",
}  # fmt: skip


def anchors(ev):
    """The start of every section of the second half, and the end of the
    film. An anchor names a line, whose picture's start it takes, or a kind
    of event, whose first event it takes."""
    first = {}
    for t, kind, _, _ in ev["events"]:
        first.setdefault(kind, t)
    out = {}
    for name, (kind, key) in ANCHORS.items():
        table = ev["shots"] if kind == "shot" else first
        if key not in table:
            raise SystemExit(f"section {name}: the film has no {kind} {key}")
        out[name] = float(table[key][0] if kind == "shot" else table[key])
    end = float(ev["seconds"])
    starts = list(out.items())
    for (name, t0), t1 in zip(starts, [t for _, t in starts[1:]] + [end]):
        if not t0 < t1:
            raise SystemExit(
                f"section {name} starts at {t0:.3f} s, not before the next at {t1:.3f} s"
            )
    return out, end


class Grid:
    """The beats of the second half: every beat's time, from the sections'
    anchors and their bars. A bar is (section, index, first beat, beats,
    row)."""

    def __init__(self, ev):
        starts, end = anchors(ev)
        names = list(SONG)
        assert names == list(
            ANCHORS
        ), "SONG and ANCHORS name the same sections"
        beats, self.bars, self.section = [], [], {}
        for i, name in enumerate(names):
            t0 = starts[name]
            t1 = starts[names[i + 1]] if i + 1 < len(names) else end
            rows = SONG[name]
            nb = sum(row[5].get("beats", 4) for row in rows)
            lengths = 1.0 + RITARDANDO.get(name, 0.0) * np.arange(nb)
            lengths *= (t1 - t0) / lengths.sum()
            first = len(beats)
            beats.extend(t0 + np.concatenate([[0.0], np.cumsum(lengths)[:-1]]))
            self.section[name] = (first, nb, t0, t1)
            k = first
            for j, row in enumerate(rows):
                n = row[5].get("beats", 4)
                self.bars.append((name, j, k, n, row))
                k += n
        beats.append(end)
        self.beats = np.array(beats)
        self.start, self.end = starts[names[0]], end

    def time(self, b):
        """The time of beat b, a global index that may have a fraction."""
        k = min(int(np.floor(b)), len(self.beats) - 2)
        return self.beats[k] + (b - k) * (self.beats[k + 1] - self.beats[k])

    def at(self, name, bar, beat=0.0):
        """The time of a beat of a bar of a section."""
        first = [b for b in self.bars if b[0] == name][bar][2]
        return self.time(first + beat)

    def snap(self, t, div=8):
        """The time t moved to the nearest 1/div of a beat."""
        k = int(
            np.clip(
                np.searchsorted(self.beats, t, "right") - 1,
                0,
                len(self.beats) - 2,
            )
        )
        d = self.beats[k + 1] - self.beats[k]
        return self.beats[k] + round((t - self.beats[k]) / d * div) / div * d

    def sixteenths(self, bar):
        _, _, first, n, _ = bar
        return np.array([self.time(first + s / 4.0) for s in range(4 * n)])


def lanes(grid):
    """The automation lanes of synth.py's mix, from the bars' levels."""
    names = {"pad": "pad_gain", "pcut": "pad_cutoff", "arp": "arp_gain", "acut": "arp_cutoff",
             "bass": "bass_gain", "bright": "bass_bright", "drone": "drone_gain"}  # fmt: skip
    out = {lane: [] for lane in names.values()}
    out["echo_gate"] = []
    for _, _, first, n, row in grid.bars:
        mode, levels = row[0], dict(
            MODES[row[0]], **{k: v for k, v in row[5].items() if k != "beats"}
        )
        t0, t1 = grid.time(first), grid.time(first + n)
        for key, lane in names.items():
            v = levels[key]
            if isinstance(v, list):
                points = [(t0 + f * (t1 - t0), x) for f, x in v]
            else:
                a, b = v if isinstance(v, tuple) else (v, v)
                points = [(t0 + 0.03, a), (t1 - 0.03, b)]
            out[lane] += [
                (t, max(x, 1.0) if lane.endswith("cutoff") else x)
                for t, x in points
            ]
        # The arpeggio's echoes stop where the drone takes over, so that it
        # starts dry.
        gate = 0.0 if mode == "poll" else 1.0
        out["echo_gate"] += [(t0 + 0.02, gate), (t1 - 0.02, gate)]
    return out


def make_drone():
    """synth.py's drone on A1 (its own is on E1), at RMS 1: the partials of
    synth.DRONE, with a soft swell on every beat."""
    t = np.arange(ms.NW) / SR
    f = ms.hz("A1")
    x = np.zeros(ms.NW)
    for multiple, level, detuning in ms.DRONE:
        x += level * np.sin(ms.TAU * (multiple * f + detuning) * t)
    x *= ms.lowpass(0.72 + 0.28 * np.exp(-np.mod(t, ms.BEAT) / 0.22), 12.0)
    return x / np.sqrt(np.mean(x**2))


def tick(f0):
    """A step of a fast run: a sine of a few milliseconds and a click."""
    n = int(0.045 * SR)
    t = np.arange(n) / SR
    x = np.sin(ms.TAU * f0 * t) * np.exp(-t / 0.008)
    noise = ms.bandpass(ms.rng("tick").standard_normal(n), 2500.0, 7000.0)
    x += 0.4 * ms.unit(noise) * np.exp(-t / 0.0008)
    return ms.unit(x * ms.fade_out(n, 0.01))


def bloom(seconds):
    """A breath of noise that swells and fades, its low-pass opening."""
    n = int(seconds * SR)
    u = np.arange(n) / n
    noise = ms.bandpass(ms.rng("bloom").standard_normal((n, 2)), 600.0, 7000.0)
    x = ms.sweep_lowpass(
        noise, 700.0 * (5000.0 / 700.0) ** np.sin(0.5 * np.pi * u)
    )
    return ms.unit(x * (np.sin(np.pi * u) ** 2)[:, None])


class Stem(ms.Stem):
    """synth.py's stem in single precision: seven of them over four minutes
    at 48 kHz would otherwise take 2.7 GB."""

    def __init__(self):
        self.dry = np.zeros((ms.NW, 2), np.float32)
        self.send = np.zeros((ms.NW, 2), np.float32)


LANDING_SCALE = ("A4", "C5", "D5", "E5", "G5", "A5", "C6", "D6")
NAME_NOTES = ("A5", "C6", "E6")  # the three tokens of line 6.1, rising


def snap_accents(cues, grid):
    """The second half's accents on the nearest thirty-second note; two of
    one kind that would meet there keep the picture's time."""
    taken = set()
    for c in cues:
        if c["kind"] in SNAP:
            s = grid.snap(c["pic"])
            if (c["kind"], round(s, 4)) not in taken:
                c["t"] = s
                taken.add((c["kind"], round(s, 4)))
    return cues


def play_events(cues, grid, events, drums, fx):
    """The second half's events, into the stems of the events, the drums and
    the effects. Returns (time, strength) of the events that duck the pad
    and the arpeggio."""
    crash = ms.make_crash()
    thud, stab = ms.make_miss()
    tap = ms.make_tap()
    tap_dull = ms.unit(ms.lowpass(tap, 1100.0))
    bells = Stem()
    random = ms.rng("film events")
    names = iter(NAME_NOTES)
    ducks = []
    L = LEVEL_DB
    gaps = np.diff([c["pic"] for c in cues], prepend=-1.0)
    points = [c["v"] for c in cues if c["kind"] == "point"]
    lo, hi = (min(points), max(points)) if points else (0.0, 1.0)

    def bell(t0, note, level, tau=0.4, index=0.8):
        bells.put(
            t0, ms.bell(ms.hz(note), tau, index), level, send=ms.SEND["bell"]
        )

    def landing(t0, h, level, p=0.0):
        note = LANDING_SCALE[
            min(int(h * len(LANDING_SCALE)), len(LANDING_SCALE) - 1)
        ]
        events.put(
            t0,
            ms.landing_tick(ms.hz(note), False),
            level,
            pan=p,
            send=ms.SEND["landing"],
        )

    for i, c in enumerate(cues):
        t0, kind, v = c["t"], c["kind"], c["v"]
        fast = gaps[i] < 0.12
        if kind == "miss":
            events.put(
                t0,
                tap_dull,
                L["tap"] - 2.0 - (5.0 if fast else 0.0),
                pan=random.uniform(-0.3, 0.3),
                send=ms.SEND["tap"],
            )
        elif kind == "match":
            events.put(t0, tap, L["tap"], send=ms.SEND["tap"])
            bell(t0, "A5", L["small_hit"] - (2.0 if fast else 0.0), 0.32, 0.7)
            ducks.append((t0, 1.0))
        elif (
            kind == "turn"
        ):  # turned around: a swell into the groove's first downbeat
            fx.put(
                t0 - 2.0,
                ms.make_swell(2.0),
                L["swell"] + 4.0,
                send=ms.SEND["fx"],
            )
            drums.put(t0, crash, L["crash"] - 2.0, send=ms.SEND["crash"])
            bell(t0, "A5", L["big_hit"] - 4.0, 0.9, 0.8)
            bell(t0, "E6", L["big_hit"] - 7.0, 0.9, 0.8)
            ducks.append((t0, 1.0))
        elif kind in ("count", "inv", "digamma", "estimate", "result", "fit"):
            note = "E5" if kind in ("count", "inv") else "A5"
            bell(
                t0,
                note,
                L["bell"] - (3.0 if kind == "result" else 0.0),
                0.5,
                0.6,
            )
        elif kind == "zero":
            events.put(t0, tap, L["tap"], send=ms.SEND["tap"])
        elif kind == "term":  # the k-th harmonic of A2, at 1/k
            k = int(v)
            n = int(3.0 * SR)
            tt = np.arange(n) / SR
            x = np.sin(ms.TAU * k * ms.hz("A2") * tt) * np.exp(-tt / 2.2)
            x *= (1.0 - np.exp(-tt / 0.006)) * ms.fade_out(n, 0.4)
            events.put(t0, x / k, L["term"], pan=0.25 * np.sin(k), send=0.3)
        elif kind == "rain":
            seconds, column = v
            q = ms.rng(f"rain {c['line']} {column}")
            for _ in range(12):
                x = tick(ms.hz(LANDING_SCALE[q.integers(4, 8)]))
                events.put(
                    t0 + seconds * q.random(),
                    x,
                    L["tick"] - 2.0,
                    pan=(column - 1) * 0.5,
                    send=ms.SEND["tap"],
                )
        elif kind == "imean":  # IBS's mean, on the true value
            bell(t0, "E5", L["small_hit"], 0.5, 0.6)
            ducks.append((t0, 0.6))
        elif kind == "lit":
            for name, lower in (("A5", 0.0), ("E6", 3.0)):
                bell(t0, name, L["big_hit"] - 2.0 - lower, 0.9, 0.8)
            drums.put(t0, crash, L["crash"] - 3.0, send=ms.SEND["crash"])
            ducks.append((t0, 1.0))
        elif kind == "row":
            landing(t0, (int(v) % 8) / 8.0, L["landing"])
        elif kind == "sum":
            bell(t0, "A5", L["bell"], 0.6, 0.6)
            bell(t0 + 0.09, "E6", L["bell"] - 3.0, 0.6, 0.6)
        elif kind in ("errbar", "post"):
            fx.put(t0, bloom(2.2), L["bloom"], send=ms.SEND["fx"])
        elif kind == "est":
            landing(
                t0,
                0.7 if v else 0.2,
                L["landing"] - (0.0 if v else 3.0),
                random.uniform(-0.4, 0.4),
            )
        elif kind == "truth":
            landing(t0, 0.0, L["landing"])
        elif kind == "repeat":
            bell(t0, "C6" if v == 2 else "E6", L["small_hit"] - 2.0, 0.4, 0.7)
        elif kind == "round":
            for _ in range(int(min(v, 6))):
                x = tick(ms.hz(LANDING_SCALE[random.integers(3, 8)]))
                level = L["tick"] + 2.0 * np.log2(1 + v) / np.log2(28)
                events.put(
                    t0 + 0.1 * random.random(),
                    x,
                    level,
                    pan=random.uniform(-0.6, 0.6),
                    send=ms.SEND["tap"],
                )
        elif kind == "name":
            for name, lower in (("A5", 0.0), ("E6", 2.0)):
                bell(t0, name, L["big_hit"] - lower, 0.95, 0.9)
            bell(t0, "A4", L["big_hit"] - 6.0, 0.8, 0.5)
            drums.put(t0, crash, L["crash"], send=ms.SEND["crash"])
            ducks.append((t0, 1.0))
        elif kind == "ghost":
            events.put(
                t0,
                tap_dull,
                L["tap"] - 6.0,
                pan=random.uniform(-0.3, 0.3),
                send=ms.SEND["tap"],
            )
        elif kind == "token":
            bell(t0, next(names), L["bell"] + 1.0, 0.6, 0.6)
        elif kind == "point":
            landing(
                t0,
                (v - lo) / max(hi - lo, 1e-9),
                L["landing"],
                random.uniform(-0.4, 0.4),
            )
        elif kind == "fpoint":  # an estimate of fixed sampling: a dull thud
            events.put(t0, thud, L["miss_thud"] - 8.0, send=ms.SEND["miss"])
        elif kind == "bias":
            events.put(t0, stab, L["fooled"], send=0.45)
            events.put(t0, thud, L["miss_thud"] - 3.0, send=ms.SEND["miss"])
            ducks.append((t0, 1.0))
        elif kind == "false":  # the fall into the drone
            fx.put(
                t0 - 1.35,
                ms.make_powerdown(1.35),
                L["powerdown"],
                send=ms.SEND["fx"],
            )
        elif kind == "withibs":
            for name, lower in (("A5", 2.0), ("E6", 0.0), ("A6", 4.0)):
                bell(t0, name, L["big_hit"] - 2.0 - lower, 1.2, 0.6)
            drums.put(t0, crash, L["crash"], send=ms.SEND["crash"])
            ducks.append((t0, 1.0))
        elif kind == "end":
            for name, lower in (
                ("A6", 0.0),
                ("E6", 3.0),
                ("C#6", 5.0),
                ("A5", 6.0),
            ):
                bell(t0, name, L["big_hit"] - 2.0 - lower, 1.0, 0.7)
            drums.put(t0, crash, L["crash"] - 4.0, send=ms.SEND["crash"])
            ducks.append((t0, 1.0))
        else:
            raise ValueError(
                f"no sound in the second half for the event {kind!r} of line {c['line']}"
            )

    events.dry += bells.dry
    events.send += bells.send
    echoes = ms.echoes(bells.dry, "bell")
    events.dry += echoes
    events.send += ms.SEND["bell"] * echoes
    return ducks


def effects(grid, fx, drums):
    """The swells, crashes, risers and falls at the changes of section."""
    at = grid.at
    swell, crash = ms.make_swell(ms.BEAT), ms.make_crash()
    L = LEVEL_DB
    for t0, level in [
        (at("unbiased", 0), 0.0),
        (at("errors", 2), -5.0),
        (at("bias", 2), -4.0),
    ]:
        fx.put(t0 - ms.BEAT, swell, L["swell"] + level, send=ms.SEND["fx"])
    for t0, level in [(at("unbiased", 0), -2.0), (at("errors", 2), -6.0)]:
        drums.put(t0, crash, L["crash"] + level, send=ms.SEND["crash"])
    # Into the drone: the falls before the terms, after the lift of 4.5 and
    # into the false conclusion.
    fx.put(
        at("terms", 0) - 1.35, ms.make_powerdown(1.35), L["powerdown"] - 2.0
    )
    fx.put(
        at("unbiased", 4),
        ms.make_downlifter(2.2),
        L["downlifter"] - 3.0,
        send=ms.SEND["fx"],
    )
    fx.put(
        at("bias", 4),
        ms.make_downlifter(2.2),
        L["downlifter"],
        send=ms.SEND["fx"],
    )
    # The builds: risers into the arrival and into the name.
    fx.put(
        at("terms", 2),
        ms.make_riser(at("unbiased", 0) - at("terms", 2)),
        L["riser"] - 2.0,
        send=ms.SEND["fx"],
    )
    fx.put(
        at("result", 1),
        ms.make_riser(at("handoff", 0) - at("result", 1)),
        L["riser"],
        send=ms.SEND["fx"],
    )


def play_bars(grid, stems):
    """The drums, the bass and the arpeggio, bar by bar. Returns the kicks
    as (time, strength), for the sidechain."""
    kick, snare, hat = ms.make_kick(), ms.make_snare(), ms.make_hat()
    drums, bass, arp = stems["drums"], stems["bass"], stems["arp"]
    bass_ref = np.abs(
        ms.bass_note(ms.hz("E2"), 0.72 * ms.STEP, 1.0, 1.0)
    ).max()
    arp_ref = np.abs(ms.arp_note(ms.hz("E4"), 1.0, 1500.0)).max()
    L = LEVEL_DB
    kicks = []
    for bar in grid.bars:
        _, _, first, n, (mode, chord, drum_name, bass_name, arp_name, _) = bar
        times = grid.sixteenths(bar)
        step = (grid.time(first + n) - grid.time(first)) / (4 * n)
        if drum_name:
            for t0, k, s, h in zip(times, *DRUMS[drum_name]):
                if k != ".":
                    drums.put(t0, kick * (int(k) / 9.0), L["kick"])
                    kicks.append((t0, int(k) / 9.0))
                if s != ".":
                    drums.put(
                        t0,
                        snare * (int(s) / 9.0),
                        L["snare"],
                        send=ms.SEND["snare"],
                    )
                if h != ".":
                    drums.put(
                        t0,
                        hat * (int(h) / 9.0),
                        L["hat"],
                        pan=0.25,
                        send=ms.SEND["hat"],
                    )
        if bass_name:
            root = ms.note_number(CHORDS[chord][0])
            row = BASS[bass_name][: len(times)]
            slow = bass_name == "long"
            for s, (t0, mark) in enumerate(zip(times, row)):
                gain = ms.lane("bass_gain", t0)
                if mark == "." or gain < 0.02:
                    continue
                later = [k for k in range(s + 1, len(row)) if row[k] != "."]
                length = min(later[0] - s if later else len(row) - s, 2)
                gate = (
                    min(1.9, 4 * n * step - 0.1)
                    if slow
                    else 0.72 * length * step
                )
                vel = (1.0, 0.68, 0.84, 0.68)[s % 4]
                f0 = ms.hz(root + {"0": 0, "+": 12, "b": 1}[mark])
                x = ms.bass_note(
                    f0, gate, vel, ms.lane("bass_bright", t0), slow
                )
                bass.put(t0, x * (gain / bass_ref), L["bass"])
        if arp_name:
            tones = CHORDS[chord][2]
            for s, (t0, index) in enumerate(zip(times, ARP[arp_name])):
                gain = ms.lane("arp_gain", t0)
                if index is None or gain < 0.02:
                    continue
                vel = (1.0, 0.6, 0.8, 0.6)[s % 4] * gain
                x = ms.arp_note(
                    ms.hz(tones[index % len(tones)]),
                    vel,
                    ms.lane("arp_cutoff", t0),
                )
                arp.put(
                    t0,
                    x / arp_ref,
                    L["arp"],
                    pan=(-0.35, 0.35, 0.35, -0.35)[s % 4],
                )
    return kicks


def pads(grid):
    """The pad's chords: the bars with one chord in a row make one."""
    out = []
    for _, _, first, n, row in grid.bars:
        chord = row[1]
        if out and out[-1][2] == chord and out[-1][3] == first:
            out[-1][1], out[-1][3] = grid.time(first + n), first + n
        elif chord:
            out.append(
                [grid.time(first), grid.time(first + n), chord, first + n]
            )
    return [(t0, t1, chord) for t0, t1, chord, _ in out]


def second_half(grid, cues):
    """Play the second half. Returns its stems with their reverb."""
    stems = {name: Stem() for name in ms.STEMS}
    drums, bass, pad, arp, lead, events, fx = (
        stems[name] for name in ms.STEMS
    )
    t = np.arange(ms.NW) / SR

    kicks = play_bars(grid, stems)
    bass.dry += ms.stereo(
        make_drone() * ms.lane("drone_gain", t) * ms.db(LEVEL_DB["drone"])
    )

    for t0, t1, chord in pads(grid):
        x = ms.pad_chord(
            CHORDS[chord][1], t1 - t0, 0.4, 0.4, ms.rng(f"pad at {t0:.3f}")
        )
        pad.put(t0, x, LEVEL_DB["pad"])
    pad.dry = ms.sweep_lowpass(pad.dry, ms.lane("pad_cutoff", t))
    pad.dry = ms.highpass(pad.dry, 140.0) * ms.lane("pad_gain", t)[:, None]
    mid = pad.dry.mean(axis=1, keepdims=True)
    pad.dry = mid + ms.PAD_WIDTH * (pad.dry - mid)
    pad.dry /= np.sqrt(0.5 + 0.5 * ms.PAD_WIDTH**2)
    pad.send = ms.SEND["pad"] * pad.dry

    arp.dry += ms.echoes(arp.dry, "arp") * ms.lane("echo_gate", t)[:, None]
    arp.send = ms.SEND["arp"] * arp.dry

    lead_ref = np.abs(ms.lead_note(ms.hz("E5"), 0.5)).max()
    for name, bar, s, length, note in LEAD:
        first = [b for b in grid.bars if b[0] == name][bar][2]
        t0, t1 = grid.time(first + s / 4.0), grid.time(
            first + (s + length) / 4.0
        )
        lead.put(
            t0,
            ms.lead_note(ms.hz(note), t1 - t0 - 0.03) / lead_ref,
            LEVEL_DB["lead"],
        )
    lead.dry += ms.echoes(lead.dry, "lead")
    lead.send = ms.SEND["lead"] * lead.dry

    ducks = play_events(cues, grid, events, drums, fx)
    effects(grid, fx, drums)

    for name, depth in ms.SIDECHAIN.items():
        stems[name].scale(ms.duck_curve(kicks, depth))
    for name, depth in ms.EVENT_DUCK.items():
        stems[name].scale(ms.duck_curve(ducks, depth, 0.004, 0.2))

    ir = ms.reverb_ir()
    out = {}
    for name, stem in stems.items():
        if stem.send.any():
            for c in (0, 1):
                stem.dry[:, c] += fftconvolve(stem.send[:, c], ir[:, c])[
                    : ms.NW
                ]
        out[name] = stem.dry
        stem.send = None
    return out


def check_song():
    """Stop before the synthesis when the form names what does not exist."""
    problems = []
    for name, rows in SONG.items():
        for j, (mode, chord, drums, bass, arp, levels) in enumerate(rows):
            where = f"{name} bar {j}"
            if mode not in MODES:
                problems.append(f"{where}: no mode {mode}")
            if chord is not None and chord not in CHORDS:
                problems.append(f"{where}: no chord {chord}")
            if drums and drums not in DRUMS:
                problems.append(f"{where}: no drum pattern {drums}")
            if drums and any(
                len(p) < 4 * levels.get("beats", 4) for p in DRUMS[drums]
            ):
                problems.append(
                    f"{where}: the drum pattern {drums} is shorter than the bar"
                )
            if bass and (bass not in BASS or chord is None):
                problems.append(f"{where}: no bass {bass} over {chord}")
            if arp and (
                arp not in ARP
                or chord not in CHORDS
                or CHORDS[chord][2] is None
            ):
                problems.append(f"{where}: no arpeggio {arp} over {chord}")
            unknown = set(levels) - set(MODES["intro"]) - {"beats"}
            if unknown:
                problems.append(f"{where}: unknown levels {sorted(unknown)}")
    if problems:
        raise SystemExit("the form does not hold:\n  " + "\n  ".join(problems))


# ══════════════════════════════════════════════════════════════════════════
# The mix with the voice
# ══════════════════════════════════════════════════════════════════════════


def duck(ev, n, depth):
    """A gain that dips by `depth` while the voice speaks, smoothed."""
    d = np.zeros(n)
    for a, b in ev["lines"].values():
        d[max(int((a - 0.15) * SR), 0) : int((b + 0.3) * SR)] = 1.0
    d = sosfiltfilt(butter(1, 2.5, fs=SR, output="sos"), d)
    return 1.0 - depth * np.clip(d, 0.0, 1.0)


def write(path, x):
    sf.write(str(path), x.astype(np.float32), SR, subtype="FLOAT")


def plan_text(vb, grid, cues):
    lines = ["The first half: time, chord, gain, cutoff"]
    lines += [
        f"  {t:7.2f}  {ch or 'silence':8s} {g:.2f} {cut:5.0f}"
        for t, ch, g, cut in vb
    ]
    lines.append("The second half: every bar")
    for bar in grid.bars:
        name, j, first, n, (mode, chord, d, b, a, _) = bar
        t0, t1 = grid.time(first), grid.time(first + n)
        kinds = {}
        for c in cues:
            if t0 <= c["t"] < t1:
                kinds[c["kind"]] = kinds.get(c["kind"], 0) + 1
        what = ", ".join(f"{k} x{v}" if v > 1 else k for k, v in kinds.items())
        bpm = 60.0 * n / (t1 - t0)
        lines.append(
            f"  {name:8s} {j:2d}  {t0:7.2f}-{t1:7.2f}  {bpm:5.1f} BPM  {n}/4  "
            f"{mode:7s} {chord or '-':9s} {d or '-':7s} {b or '-':10s} {a or '-':8s} {what}"
        )
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("v", type=Path, help="the folder of a voice's media")
    out = ap.parse_args().v
    start = time.time()
    ev = json.loads((out / "events.json").read_text(encoding="utf-8"))
    if ev.get("dropped"):
        print(
            f"warning: {len(ev['dropped'])} events fall outside their picture and are left out",
            flush=True,
        )
    check_song()
    grid = Grid(ev)
    N = int(round(grid.end * SR))
    ms.N, ms.NW, ms.DURATION = N, N + int(ms.TAIL * SR), grid.end
    ms.AUTOMATION = lanes(grid)
    ms.LEVEL_DB = LEVEL_DB
    ms.TAP_NOTE = "E5"
    ms.MISS_STAB = (
        "Bb3",
        "E4",
    )  # the flat second of A and the note a tritone above it

    cues = [
        dict(t=t, pic=t, kind=k, v=v, line=line)
        for t, k, v, line in ev["events"]
    ]
    split, turn = grid.start, grid.at("ibs", 0)
    cues_a = [c for c in cues if c["pic"] < split]
    cues_b = snap_accents([c for c in cues if c["pic"] >= split], grid)
    tempi = ", ".join(
        f"{name} {60.0 * nb / (t1 - t0):.1f}"
        for name, (_, nb, t0, t1) in grid.section.items()
    )
    print(
        f"first half to {split:.2f} s; second half {len(grid.bars)} bars, BPM by section: {tempi}",
        flush=True,
    )
    moved = max(abs(c["t"] - c["pic"]) for c in cues_b)
    print(
        f"{len(cues_a)} events in the first half, {len(cues_b)} in the second; accents move by at most {1000 * moved:.0f} ms",
        flush=True,
    )

    print("playing ...", flush=True)
    music_a, fx_a, vb = first_half(ev, cues_a, turn, ms.NW)
    music_a, fx_a = vb_reverb(music_a), vb_reverb(fx_a)
    print(f"  first half: {time.time() - start:.0f} s", flush=True)
    stems = second_half(grid, cues_b)
    print(f"  second half: {time.time() - start:.0f} s", flush=True)
    (out / "score_plan.txt").write_text(
        "\n".join(plan_text(vb, grid, cues_b)) + "\n", encoding="utf-8"
    )

    fades = ms.fade_in(N, 0.02) * ms.fade_out(N, FADE_OUT)
    band = (
        lambda x: ms.lowpass(ms.highpass(x[:N], 25.0), 14000.0)
        * fades[:, None]
    )  # noqa: E731
    for k in list(stems):
        stems[k] = band(stems[k])
    music_a, fx_a = band(music_a), band(fx_a)
    # The first half's music sits FIRST_HALF_LU below the second half's, and
    # its effects 3 dB below its music, as in the PyVBMC film.
    i, j = int(split * SR), int(turn * SR)
    music_b = sum(stems[k] for k in MUSIC)
    music_a *= ms.db(
        ms.loudness(music_b[i:], False)
        + FIRST_HALF_LU
        - ms.loudness(music_a[:j], False)
    )
    fx_a *= ms.db(
        ms.loudness(music_a[:j], False) - 3.0 - ms.loudness(fx_a[:j], False)
    )
    stems["first half music"], stems["first half effects"] = music_a, fx_a

    voice, rate = sf.read(str(out / "narration.wav"), dtype="float64")
    assert rate == SR, rate
    voice = np.pad(voice, (0, max(0, N - len(voice))))[:N]
    voice = np.stack([voice, voice], axis=1)
    music = (music_b + music_a) * duck(ev, N, DUCK["music"])[:, None]
    other = (stems["events"] + stems["fx"] + fx_a) * duck(
        ev, N, DUCK["events"]
    )[:, None]
    score = music + other
    lv = ms.loudness(voice)
    gain = ms.db(lv + SCORE_LU - ms.loudness(score))
    score *= gain
    mix = voice + score
    peak = np.abs(mix).max()
    if peak > 0.89:  # -1 dBFS at most; mux.py sets the loudness
        mix *= 0.89 / peak
    write(out / "score.wav", score)
    write(out / "mix.wav", mix)
    (out / "stems").mkdir(exist_ok=True)
    for old in (out / "stems").glob("*.wav"):
        old.unlink()
    for k, x in stems.items():
        write(out / "stems" / f"{k}.wav", x * gain)

    print(
        f"voice {lv:.1f} LUFS; score {ms.loudness(score):.1f} LUFS ({ms.loudness(score) - lv:+.1f} LU)",
        flush=True,
    )
    for k, x in stems.items():
        print(
            f"  {k:18s} {ms.loudness(x * gain, False):6.1f} LUFS ungated, peak {ms.to_db(np.abs(x * gain).max()):6.1f} dBFS",
            flush=True,
        )
    print(
        f"score: first half {ms.loudness(score[:i], False):.1f}, second half {ms.loudness(score[i:], False):.1f} LUFS ungated",
        flush=True,
    )
    margins = sorted(
        (
            ms.loudness(voice[int(a * SR) : int(b * SR)], False)
            - ms.loudness(score[int(a * SR) : int(b * SR)], False),
            line,
        )
        for line, (a, b) in ev["lines"].items()
    )
    print(
        "voice over score, line by line: "
        + ", ".join(f"{line} {m:.1f}" for m, line in margins[:5])
        + f" ... {margins[-1][1]} {margins[-1][0]:.1f} dB",
        flush=True,
    )
    print(
        f"mix {ms.loudness(mix):.1f} LUFS, peak {ms.to_db(np.abs(mix).max()):.1f} dBFS, {N / SR:.3f} s; wrote {out}; {time.time() - start:.0f} s",
        flush=True,
    )


if __name__ == "__main__":
    main()
