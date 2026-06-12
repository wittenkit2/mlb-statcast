import datetime as dt
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({
    "figure.facecolor": "#ffffff", "axes.facecolor": "#ffffff", "savefig.facecolor": "#ffffff",
    "text.color": "#111418", "axes.labelcolor": "#33373d", "axes.titlecolor": "#0f1722",
    "axes.edgecolor": "#c8ced8", "xtick.color": "#444a52", "ytick.color": "#444a52",
    "legend.facecolor": "#ffffff", "legend.edgecolor": "#d4d9e2", "legend.framealpha": 0.95,
})
import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
from pybaseball import (batting_stats, pitching_stats, playerid_lookup, playerid_reverse_lookup,
                        statcast, statcast_batter, statcast_pitcher, team_batting, team_pitching)

FG_ALIAS = {"CWS": "CHW", "SD": "SDP", "TB": "TBR", "WSH": "WSN", "KC": "KCR", "SF": "SFG"}

st.set_page_config(page_title="MLB Statcast Analysis", page_icon="⚾", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 2.2rem; max-width: 1180px;}
    [data-testid="stMetric"] {
        background: #f7f9fc;
        border: 1px solid #e4e9f0; border-radius: 14px; padding: 14px 18px;
        box-shadow: 0 1px 2px rgba(16,20,30,0.04);
    }
    [data-testid="stMetricValue"] {font-weight: 800; color: #0f1722 !important;}
    [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p {color: #5b6472 !important;}
    [data-testid="stDataFrame"] {border: 1px solid #e4e9f0; border-radius: 12px; overflow: hidden;}
    button[data-baseweb="tab"] {font-size: 0.95rem;}
    h2, h3 {letter-spacing: -0.3px;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

SWING_DESCS = {
    "hit_into_play", "foul", "foul_tip", "swinging_strike",
    "swinging_strike_blocked", "foul_bunt", "missed_bunt",
}
XWOBA = "estimated_woba_using_speedangle"
XBA = "estimated_ba_using_speedangle"
HIT_EVENTS = {"single", "double", "triple", "home_run"}
HIT_MAP = {
    "home_run": "HR", "triple": "3B", "double": "2B", "single": "1B",
    "walk": "BB", "hit_by_pitch": "BB",
    "strikeout": "K", "strikeout_double_play": "K",
}
OUTCOME_ORDER = ["HR", "3B", "2B", "1B", "BB", "K", "Out/other"]
PITCH_SEEN_MAP = {
    "ball": "Balls", "blocked_ball": "Balls", "pitchout": "Balls",
    "called_strike": "Called strikes",
    "swinging_strike": "Swinging strikes (whiffs)",
    "swinging_strike_blocked": "Swinging strikes (whiffs)",
    "missed_bunt": "Swinging strikes (whiffs)",
    "foul": "Fouls", "foul_tip": "Fouls", "foul_bunt": "Fouls",
    "hit_into_play": "In play",
    "hit_by_pitch": "Hit by pitch",
}
PITCH_SEEN_ORDER = ["Balls", "Called strikes", "Swinging strikes (whiffs)",
                    "Fouls", "In play", "Hit by pitch", "Other"]
POS = {1: "P", 2: "C", 3: "1B", 4: "2B", 5: "3B", 6: "SS", 7: "LF", 8: "CF", 9: "RF"}
BASE = 90.0 / np.sqrt(2)
TEAM_IDS = {
    "LAA": 108, "ARI": 109, "AZ": 109, "BAL": 110, "BOS": 111, "CHC": 112, "CIN": 113,
    "CLE": 114, "COL": 115, "DET": 116, "HOU": 117, "KC": 118, "LAD": 119, "WSH": 120,
    "WSN": 120, "NYM": 121, "OAK": 133, "ATH": 133, "PIT": 134, "SD": 135, "SDP": 135,
    "SEA": 136, "SF": 137, "SFG": 137, "STL": 138, "TB": 139, "TBR": 139, "TEX": 140,
    "TOR": 141, "MIN": 142, "PHI": 143, "ATL": 144, "CWS": 145, "CHW": 145, "MIA": 146,
    "NYY": 147, "MIL": 158,
}

GLOSSARY = {
    "Chase rate": "How often a hitter swings at pitches outside the strike zone. Lower is better — it reflects plate discipline.",
    "Strike zone": "The area over the plate (about 1.66 ft wide) between the batter's knees and the midpoint of the torso. Pitches outside it are balls if taken.",
    "Out of zone": "A pitch located outside the strike zone — horizontally off the plate or above/below the batter's zone.",
    "Plate appearance": "One completed turn at bat, ending in a hit, walk, strikeout, out, etc. Abbreviated PA.",
    "Outcome odds": "How often each result (HR, double, single, etc.) actually occurred per plate appearance over the chosen window.",
    "xwOBA": "Expected Weighted On-base Average. Estimates the value of a hitter's contact from exit velocity and launch angle, stripping out luck and defense. About .320 is league average.",
    "xBA": "Expected Batting Average — the probability a batted ball becomes a hit given its exit velocity and launch angle.",
    "Rolling xwOBA": "A 50-batted-ball moving average of xwOBA, used to see whether contact quality is trending up or down over time.",
    "Contact quality": "How hard and at what angle a hitter strikes the ball, summarized here by xwOBA on contact.",
    "Baseline": "A reference period (default: the hitter's 2024 season) to compare current performance against.",
    "Release point": "Where the pitcher's hand releases the ball, measured in feet. A consistent release across pitch types makes pitches harder to tell apart.",
    "Tunneling": "When different pitches travel the same apparent path early, so the hitter can't distinguish them until it's too late to adjust.",
    "Tunnel ratio": "Plate gap divided by release gap. Higher means two pitches start from nearly the same spot but finish far apart — the hallmark of good tunneling.",
    "Standard deviation (SD)": "A measure of spread. A smaller SD here means the pitcher repeats that release point more consistently from pitch to pitch.",
    "Fastball (group)": "The hardest, straightest pitches — 4-seam, sinker, and cutter.",
    "Breaking ball": "A pitch thrown to curve or slide on the way to the plate — sliders, curveballs, sweepers, slurves. It 'breaks' off a straight path using heavy spin.",
    "Offspeed": "A pitch made to look like a fastball but arrive slower to upset timing — changeups, splitters, forkballs.",
    "Batted-ball type": "How the ball was hit: ground ball, line drive, fly ball, or popup.",
    "Fielded": "Credited by Statcast (the hit_location field) with fielding the batted ball — includes grounders and catches, not strictly catches only.",
    "Error": "A misplay by a fielder (bad throw, dropped ball, fumble) that lets a batter or runner advance when ordinary effort would have produced an out.",
    "AVG": "Batting average — hits divided by at-bats. ~.250 is roughly league average.",
    "OBP": "On-base percentage — how often a batter reaches base (hits + walks + HBP per plate appearance). ~.320 is average.",
    "SLG": "Slugging percentage — total bases divided by at-bats; measures power. ~.400 is average.",
    "OPS": "On-base plus slugging (OBP + SLG); a quick all-around hitting number. ~.720 is average.",
    "ISO": "Isolated power — SLG minus AVG; extra-base power only. ~.150 is average.",
    "BABIP": "Batting average on balls in play; high values can signal luck or hard contact. ~.300 is typical.",
    "K%": "Strikeout rate — strikeouts divided by plate appearances. ~22% is average for hitters.",
    "BB%": "Walk rate — walks divided by plate appearances. ~8% is average.",
    "ERA": "Earned run average — earned runs allowed per 9 innings. Lower is better; ~4.00 is roughly average.",
    "WHIP": "Walks plus hits per inning pitched. Lower is better; ~1.30 is roughly average.",
    "HR/9": "Home runs allowed per 9 innings. Lower is better; ~1.2 is roughly average.",
}

PITCH_GLOSSARY = {
    "4-Seam Fastball": "Fastball group. The straightest, hardest fastball (~92–100 mph). Backspin keeps it from dropping much.",
    "Sinker": "Fastball group. A two-seam fastball with arm-side run and extra sink; thrown to induce ground balls.",
    "Cutter": "Fastball group. A fastball that breaks a few inches to the glove side just before the plate.",
    "Slider": "Breaking ball. Tight lateral and downward break, faster and shorter than a curveball (~82–88 mph).",
    "Sweeper": "Breaking ball. A slider variant with big horizontal sweep and less drop.",
    "Slurve": "Breaking ball. A hybrid between a slider and a curveball — sweeping sideways with some drop.",
    "Curveball": "Breaking ball. Big top-to-bottom break from overhand spin; slower (~74–82 mph).",
    "Knuckle Curve": "Breaking ball. A curveball gripped with a knuckle for sharper, later bite.",
    "Slow Curve": "Breaking ball. A curveball thrown with extra loop and lower velocity.",
    "Changeup": "Offspeed. Looks like a fastball but ~8–12 mph slower to disrupt timing; usually fades arm-side.",
    "Split-Finger": "Offspeed. A splitter — looks like a fastball, then drops sharply near the plate.",
    "Forkball": "Offspeed. Like a splitter but with more tumbling drop and lower velocity.",
    "Knuckleball": "Thrown with almost no spin so it flutters unpredictably; very slow.",
    "Screwball": "Breaking ball that moves the opposite way of a curve (toward the pitcher's arm side).",
    "Eephus": "An extremely slow, high-arcing lob pitch meant to surprise the hitter.",
    "Fastball": "A generic fastball classification when Statcast doesn't split it into 4-seam vs. sinker.",
}


def term(label):
    d = GLOSSARY.get(label, "")
    return f'<abbr title="{d}" style="text-decoration:underline dotted;cursor:help">{label}</abbr>'


def glossary_expander():
    with st.expander("📖 Glossary — what the terms mean"):
        for k, v in GLOSSARY.items():
            st.markdown(f"**{k}** — {v}")


def pitch_types_expander():
    with st.expander("🥎 Pitch types — what each pitch is"):
        st.markdown(
            "Pitches fall into three families: **fastballs** (hardest, straightest), "
            "**breaking balls** (curve/slide via spin), and **offspeed** (slower, fastball-looking)."
        )
        for k, v in PITCH_GLOSSARY.items():
            st.markdown(f"**{k}** — {v}")


def pct(x):
    return f"{x:.1%}" if pd.notna(x) else "n/a"


def draw_person(ax, height_ft=6.0, x0=0.0, color="0.8", zorder=1):
    head_r = max(0.22, height_ft * 0.052)
    shoulder = height_ft - 2 * head_r
    ax.plot([x0, x0], [0, shoulder], color=color, lw=7, solid_capstyle="round", zorder=zorder)
    ax.add_patch(plt.Circle((x0, height_ft - head_r), head_r, color=color, zorder=zorder))
    return shoulder


def draw_pitcher(ax, height_ft=6.0, body_x=0.0):
    ax.axhline(0, color="0.6", lw=1)
    ax.plot([-0.85, 0.85], [0, 0], color="0.4", lw=5, solid_capstyle="butt")
    return draw_person(ax, height_ft, body_x, color="0.78", zorder=2)


def draw_field(ax):
    ax.plot([0, 330 / np.sqrt(2)], [0, 330 / np.sqrt(2)], color="0.55", lw=1)
    ax.plot([0, -330 / np.sqrt(2)], [0, 330 / np.sqrt(2)], color="0.55", lw=1)
    th = np.linspace(np.pi / 4, 3 * np.pi / 4, 120)
    ax.plot(400 * np.cos(th), 400 * np.sin(th), color="0.7", lw=1)
    ax.plot([0, BASE, 0, -BASE, 0], [0, BASE, 2 * BASE, BASE, 0], color="0.5", lw=1)
    ax.scatter([BASE, 0, -BASE], [BASE, 2 * BASE, BASE], marker="s", s=22, color="0.45", zorder=3)
    ax.scatter([0], [60.5], marker="^", s=34, color="0.45", zorder=3)
    ax.scatter([0], [0], marker="D", s=26, color="0.3", zorder=3)


# ---------- Pitch guide: ANIMATED original SVG illustrations ----------
PITCHER_SYM = (
    '<symbol id="pp" viewBox="0 0 40 56">'
    '<ellipse cx="20" cy="52" rx="17" ry="5" fill="#000" opacity="0.12"/>'
    '<circle cx="20" cy="9" r="6.5" fill="#e7c8a4"/>'
    '<rect x="16.5" y="2.5" width="7" height="5" rx="2" fill="#1d2b4a"/>'
    '<path d="M14,16 Q20,13 26,16 L25,38 L15,38 Z" fill="#23365f"/>'
    '<path d="M16,16 Q8,8 6,2" stroke="#23365f" stroke-width="4.5" fill="none" stroke-linecap="round"/>'
    '<circle cx="6" cy="2.5" r="3.2" fill="#e7c8a4"/>'
    '<path d="M24,18 Q31,24 30,32" stroke="#23365f" stroke-width="4.5" fill="none" stroke-linecap="round"/>'
    '<path d="M16,38 L14,53" stroke="#1d2b4a" stroke-width="4.5" stroke-linecap="round"/>'
    '<path d="M24,38 L27,53" stroke="#1d2b4a" stroke-width="4.5" stroke-linecap="round"/>'
    '</symbol>'
)

SPIN_LABEL = {"back": "backspin", "top": "topspin", "side": "side spin",
              "fade": "run / fade", "none": "no spin (flutters)"}
# (from, to, dur) for the moving ball's seam rotation; None = knuckleball wobble.
SPIN_ANIM = {
    "back": ("0 0 0", "-360 0 0", "0.5s"),
    "top": ("0 0 0", "360 0 0", "0.55s"),
    "side": ("0 0 0", "360 0 0", "0.9s"),
    "fade": ("0 0 0", "-360 0 0", "0.7s"),
    "none": None,
}
# Finger ellipses (dx, dy, rx, ry, rotation) per grip — original stylized diagrams.
GRIPS = {
    "4seam": [(-8, -6, 5, 13, 0), (8, -6, 5, 13, 0)],
    "2seam": [(-4, -6, 5, 13, 0), (5, -6, 5, 13, 0)],
    "cutter": [(0, -6, 5, 13, 8), (10, -6, 5, 13, 8)],
    "slider": [(4, -6, 5, 13, 20), (13, -3, 5, 13, 20)],
    "curve": [(-3, -7, 5, 13, -10), (6, -7, 5, 13, -10)],
    "change": [(-9, -7, 4, 11, 0), (0, -8, 4, 11, 0), (9, -7, 4, 11, 0)],
    "split": [(-13, -4, 5, 13, -12), (13, -4, 5, 13, 12)],
    "screw": [(-13, -6, 5, 13, -20), (-3, -3, 5, 13, -20)],
    "knuckle": [(-7, -9, 3.5, 6, 0), (0, -10, 3.5, 6, 0), (7, -9, 3.5, 6, 0)],
}


def grip_svg(kind, cx, cy, r):
    seams = (f'<path d="M{cx-r+5},{cy-r+7} Q{cx-2},{cy} {cx-r+5},{cy+r-7}" stroke="#d33" stroke-width="1.4" fill="none"/>'
             f'<path d="M{cx+r-5},{cy-r+7} Q{cx+2},{cy} {cx+r-5},{cy+r-7}" stroke="#d33" stroke-width="1.4" fill="none"/>')
    fcol = 'fill="#e7c8a4" stroke="#caa97f" stroke-width="0.6"'
    # curled ring + pinky wrapping the lower-right side
    side = (f'<ellipse cx="{cx+r-3}" cy="{cy+9}" rx="4" ry="8.5" {fcol} transform="rotate(38 {cx+r-3} {cy+9})"/>'
            f'<ellipse cx="{cx+r+1}" cy="{cy+15}" rx="3.5" ry="7.5" {fcol} transform="rotate(50 {cx+r+1} {cy+15})"/>')
    # index + middle (and sometimes ring) fingers on top — varies by pitch
    top = ""
    for (dx, dy, rx, ry, rot) in GRIPS.get(kind, []):
        top += f'<ellipse cx="{cx+dx}" cy="{cy+dy}" rx="{rx}" ry="{ry}" {fcol} transform="rotate({rot} {cx+dx} {cy+dy})"/>'
    # thumb underneath, peeking at the bottom
    thumb = f'<ellipse cx="{cx-3}" cy="{cy+r-1}" rx="5.5" ry="9" {fcol} transform="rotate(-12 {cx-3} {cy+r-1})"/>'
    return (f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="#fff" stroke="#cbd2d8" stroke-width="1"/>'
            f'{seams}{side}{top}{thumb}'
            f'<text x="{cx}" y="{cy+r+13}" font-size="9" fill="#0d2016" text-anchor="middle">grip</text>')


def pitch_panel(name, velo, desc, path_d, ball, spin, grip):
    """Animated SVG panel: ball travels the path to the plate; includes a grip diagram and per-pitch spin."""
    label = SPIN_LABEL.get(spin, "")
    anim = SPIN_ANIM.get(spin)
    if anim:
        spin_anim = (f'<animateTransform attributeName="transform" type="rotate" from="{anim[0]}" to="{anim[1]}" '
                     f'dur="{anim[2]}" repeatCount="indefinite"/>')
    else:
        spin_anim = ('<animateTransform attributeName="transform" type="rotate" '
                     'values="-9 0 0;9 0 0;-9 0 0" dur="0.45s" repeatCount="indefinite"/>')
    return f'''<svg viewBox="0 0 360 250" width="354" height="246" xmlns="http://www.w3.org/2000/svg" font-family="system-ui,-apple-system,sans-serif">
<defs>
<marker id="ah" markerWidth="7" markerHeight="7" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#e23b2e" opacity="0.85"/></marker>
{PITCHER_SYM}
</defs>
<rect width="360" height="250" rx="12" fill="#4f9d5d"/>
<rect y="150" width="360" height="66" fill="#478f54" opacity="0.5"/>
<ellipse cx="296" cy="86" rx="34" ry="11" fill="#c08a52"/>
<use href="#pp" x="276" y="34" width="40" height="56"/>
<polygon points="62,184 90,184 90,170 76,160 62,170" fill="#fff" stroke="#cbd2d8" stroke-width="1"/>
<path d="{path_d}" stroke="#e23b2e" stroke-width="3" fill="none" opacity="0.7" stroke-dasharray="2,4" marker-end="url(#ah)"/>
<rect x="8" y="8" width="124" height="20" rx="5" fill="#11241a" opacity="0.55"/>
<text x="16" y="22" font-size="12" fill="#fff">Spin: {label}</text>
<g transform="translate(46,76)">{grip_svg(grip, 0, 0, 24)}</g>
<g>
  <animateMotion dur="3.4s" repeatCount="indefinite" calcMode="linear" keyPoints="0;1;1" keyTimes="0;0.62;1" path="{path_d}"/>
  <circle r="8" fill="#fff" stroke="#d33" stroke-width="0.8"/>
  <g>
    <path d="M-6,-3 Q0,3 6,-3 M-5,3 Q0,-2 5,3" stroke="#d33" stroke-width="1.1" fill="none"/>
    {spin_anim}
  </g>
</g>
<rect y="216" width="360" height="34" fill="#11241a" opacity="0.65"/>
<text x="12" y="233" font-size="14" font-weight="700" fill="#fff">{name} · ~{velo} mph</text>
<text x="12" y="246" font-size="11.5" fill="#dfeede">{desc}</text>
</svg>'''


PITCHES = [
    ("4-Seam Fastball", 95, "Straight and hard; backspin fights gravity so it barely drops.",
     "M288,74 Q190,114 86,150", (190, 112), "back", "4seam"),
    ("Sinker", 93, "Two-seam fastball that runs arm-side and sinks — lots of grounders.",
     "M288,74 C236,96 168,122 88,176", (190, 116), "fade", "2seam"),
    ("Cutter", 90, "Fastball with a short, late glove-side cut.",
     "M288,74 C232,92 150,110 84,166", (188, 104), "side", "cutter"),
    ("Slider", 86, "Sweeps sideways and down with a sharp, late break.",
     "M288,74 C246,92 168,108 86,182", (188, 106), "side", "slider"),
    ("Sweeper", 84, "A slider with big horizontal sweep — moves across, less drop.",
     "M288,74 C252,80 150,96 80,194", (196, 92), "side", "slider"),
    ("Curveball", 79, "Topspin makes it tumble — flat at first, then dives down late.",
     "M288,74 C252,76 168,86 92,198", (198, 90), "top", "curve"),
    ("Changeup", 84, "Looks like a fastball but slower; fades arm-side and sinks.",
     "M288,74 C240,92 164,116 90,184", (190, 106), "fade", "change"),
    ("Split-Finger", 86, "Fastball look, then drops off the table just before the plate.",
     "M288,74 C246,80 176,98 96,200", (198, 96), "top", "split"),
    ("Screwball", 80, "Rare — breaks the opposite way of a curve, toward the arm side.",
     "M288,74 C238,90 150,102 84,170", (186, 100), "fade", "screw"),
    ("Knuckleball", 70, "Almost no spin, so it flutters and wobbles unpredictably.",
     "M288,74 Q236,84 214,102 T156,120 T98,176", (196, 104), "none", "knuckle"),
]

# Catcher's-POV: final location in/around the zone + a control point for the break shape.
CATCHER = {
    "4-Seam Fastball": (150, 128, 150, 80),
    "Sinker": (172, 196, 170, 112),
    "Cutter": (128, 150, 150, 100),
    "Slider": (118, 198, 152, 110),
    "Sweeper": (104, 170, 162, 112),
    "Curveball": (150, 214, 150, 88),
    "Changeup": (172, 198, 160, 112),
    "Split-Finger": (150, 224, 150, 110),
    "Screwball": (176, 188, 140, 112),
    "Knuckleball": (150, 172, 150, 120),
}


def pitch_panel_catcher(name, velo, spin):
    """Catcher's-POV animation: the ball flies toward you, growing, then breaks to its spot."""
    ex, ey, cx, cy = CATCHER.get(name, (150, 170, 150, 110))
    label = SPIN_LABEL.get(spin, "")
    anim = SPIN_ANIM.get(spin)
    if anim:
        spin_anim = (f'<animateTransform attributeName="transform" type="rotate" from="{anim[0]}" to="{anim[1]}" '
                     f'dur="{anim[2]}" repeatCount="indefinite"/>')
    else:
        spin_anim = ('<animateTransform attributeName="transform" type="rotate" '
                     'values="-9 0 0;9 0 0;-9 0 0" dur="0.45s" repeatCount="indefinite"/>')
    path = f"M150,42 Q{cx},{cy} {ex},{ey}"
    return f'''<svg viewBox="0 0 300 300" width="300" height="288" xmlns="http://www.w3.org/2000/svg" font-family="system-ui,-apple-system,sans-serif">
<defs>{PITCHER_SYM}</defs>
<rect width="300" height="300" rx="12" fill="#1f3b2e"/>
<use href="#pp" x="138" y="20" width="24" height="34" opacity="0.5"/>
<rect x="110" y="120" width="80" height="92" rx="2" fill="none" stroke="#cfe3d6" stroke-width="2" opacity="0.85"/>
<polygon points="120,252 180,252 180,240 150,230 120,240" fill="#fff" opacity="0.9"/>
<rect x="8" y="8" width="126" height="20" rx="5" fill="#000" opacity="0.4"/>
<text x="16" y="22" font-size="12" fill="#fff">Spin: {label}</text>
<path d="{path}" stroke="#e23b2e" stroke-width="2" fill="none" opacity="0.3" stroke-dasharray="2,5"/>
<g>
  <animateMotion dur="2.6s" repeatCount="indefinite" calcMode="linear" keyPoints="0;1;1" keyTimes="0;0.8;1" path="{path}"/>
  <g>
    <animateTransform attributeName="transform" type="scale" values="0.35;0.35;1.5;1.5" keyTimes="0;0.05;0.8;1" dur="2.6s" repeatCount="indefinite"/>
    <circle r="10" fill="#fff" stroke="#d33" stroke-width="0.8"/>
    <g>
      <path d="M-7,-3 Q0,3 7,-3 M-6,3 Q0,-2 6,3" stroke="#d33" stroke-width="1.2" fill="none"/>
      {spin_anim}
    </g>
  </g>
</g>
<rect y="266" width="300" height="34" fill="#000" opacity="0.5"/>
<text x="12" y="283" font-size="13" font-weight="700" fill="#fff">{name} · ~{velo} mph</text>
<text x="12" y="296" font-size="10.5" fill="#cfe3d6">Coming at you — grows as it nears the plate, then breaks.</text>
</svg>'''


@st.cache_data(show_spinner=False)
def player_bio(mlbam):
    try:
        r = requests.get(f"https://statsapi.mlb.com/api/v1/people/{mlbam}", timeout=10)
        d = r.json()["people"][0]
        hs = d.get("height")
        w = d.get("weight")
        parts = hs.replace('"', "").split("'")
        feet = int(parts[0].strip())
        inches = int(parts[1].strip()) if len(parts) > 1 and parts[1].strip() else 0
        return feet + inches / 12.0, hs, w
    except Exception:
        return None, None, None


@st.cache_data(show_spinner=False)
def batter_hand(mlbam):
    """Batting side from the MLB Stats API: 'R', 'L', or 'S' (switch). None if unknown."""
    try:
        r = requests.get(f"https://statsapi.mlb.com/api/v1/people/{mlbam}", timeout=10)
        return r.json()["people"][0].get("batSide", {}).get("code")
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def pitcher_team(mlbam):
    """The pitcher's current team id, so we can find his game today."""
    try:
        r = requests.get(f"https://statsapi.mlb.com/api/v1/people/{mlbam}", timeout=10)
        return r.json()["people"][0].get("currentTeam", {}).get("id")
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def today_opponent(team_id, today_iso):
    """Today's game for a team → (game_pk, opponent_team_id, opponent_is_home). (None, None, None) if no game."""
    try:
        url = (f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&teamId={team_id}"
               f"&startDate={today_iso}&endDate={today_iso}")
        d = requests.get(url, timeout=10).json()
        for day in d.get("dates", []):
            for g in day.get("games", []):
                h, a = g["teams"]["home"]["team"], g["teams"]["away"]["team"]
                if h["id"] == team_id:
                    return g["gamePk"], a["id"], False
                if a["id"] == team_id:
                    return g["gamePk"], h["id"], True
        return None, None, None
    except Exception:
        return None, None, None


@st.cache_data(show_spinner=False)
def game_lineup(game_pk, opp_is_home):
    """Opposing batting order from the live game feed → [(id, name)]. Empty until lineups post."""
    try:
        r = requests.get(f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live", timeout=10)
        box = r.json()["liveData"]["boxscore"]["teams"]
        side = box["home" if opp_is_home else "away"]
        players = side.get("players", {})
        out = []
        for pid in side.get("battingOrder", []):
            nm = players.get(f"ID{pid}", {}).get("person", {}).get("fullName")
            if nm:
                out.append((int(pid), nm))
        return out
    except Exception:
        return []


@st.cache_data(show_spinner=False)
def team_hitters(team_id, season):
    """Position players for a team → [(id, name)] sorted by last name.
    Tries active roster first, then 40-man, then full season, so it works pre-lineup and off-season."""
    if not team_id:
        return []
    for rtype in ("active", "40Man", "fullSeason"):
        try:
            r = requests.get(f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster"
                             f"?rosterType={rtype}&season={season}", timeout=10)
            out = [(int(p["person"]["id"]), p["person"]["fullName"])
                   for p in r.json().get("roster", [])
                   if p.get("position", {}).get("type") != "Pitcher"]
            if out:
                out.sort(key=lambda x: x[1].split()[-1])
                return out
        except Exception:
            continue
    return []


@st.cache_data(show_spinner=False)
def next_game(team_id, today_iso):
    try:
        s = today_iso
        e = (dt.date.fromisoformat(today_iso) + dt.timedelta(days=21)).isoformat()
        url = (f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&teamId={team_id}"
               f"&startDate={s}&endDate={e}&hydrate=probablePitcher,team")
        d = requests.get(url, timeout=10).json()
        for day in d.get("dates", []):
            for g in day.get("games", []):
                gdate = g.get("gameDate", "")[:10]
                if gdate and gdate < today_iso:
                    continue  # skip anything already in the past
                h, a = g["teams"]["home"], g["teams"]["away"]
                return {
                    "date": gdate,
                    "home": h["team"].get("abbreviation", h["team"].get("name", "?")),
                    "away": a["team"].get("abbreviation", a["team"].get("name", "?")),
                    "home_pp": (h.get("probablePitcher") or {}).get("fullName"),
                    "home_pp_id": (h.get("probablePitcher") or {}).get("id"),
                    "away_pp": (a.get("probablePitcher") or {}).get("fullName"),
                    "away_pp_id": (a.get("probablePitcher") or {}).get("id"),
                }
        return None
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def lookup_id(last, first):
    df = playerid_lookup(last.strip(), first.strip())
    if df is None or len(df) == 0:
        return None, None
    df = df.sort_values("mlb_played_last", ascending=False)
    row = df.iloc[0]
    name = f"{str(row['name_first']).title()} {str(row['name_last']).title()}"
    return int(row["key_mlbam"]), name


@st.cache_data(show_spinner=False)
def pull_batter(start, end, pid):
    return statcast_batter(start, end, pid)


@st.cache_data(show_spinner=False)
def pull_pitcher(start, end, pid):
    return statcast_pitcher(start, end, pid)


@st.cache_data(show_spinner=False)
def pull_team(start, end, team):
    return statcast(start, end, team=team)


@st.cache_data(show_spinner=False)
def pull_day(date):
    return statcast(date, date)


@st.cache_data(show_spinner=False)
def opp_pitching_faced(team_id, start, end, season):
    """Every pitch this team's hitters faced = the opposing pitching. Built per-batter (slow)."""
    try:
        r = requests.get(f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster"
                         f"?rosterType=fullSeason&season={season}", timeout=10)
        roster = r.json().get("roster", [])
        bat_ids = [p["person"]["id"] for p in roster
                   if p.get("position", {}).get("type") != "Pitcher"]
    except Exception:
        return None
    frames = []
    for pid in bat_ids:
        try:
            d = statcast_batter(start, end, pid)
            if d is not None and len(d):
                frames.append(d)
        except Exception:
            continue
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


ID2ABBR = {108: "LAA", 109: "ARI", 110: "BAL", 111: "BOS", 112: "CHC", 113: "CIN", 114: "CLE",
           115: "COL", 116: "DET", 117: "HOU", 118: "KC", 119: "LAD", 120: "WSH", 121: "NYM",
           133: "ATH", 134: "PIT", 135: "SD", 136: "SEA", 137: "SF", 138: "STL", 139: "TB",
           140: "TEX", 141: "TOR", 142: "MIN", 143: "PHI", 144: "ATL", 145: "CWS", 146: "MIA",
           147: "NYY", 158: "MIL"}


def _col(df, name):
    return df[name] if name in df.columns else pd.Series([np.nan] * len(df), index=df.index)


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def _ip_to_float(ip):
    try:
        a = str(ip).split(".")
        return int(a[0]) + (int(a[1]) / 3.0 if len(a) > 1 and a[1] != "" else 0.0)
    except Exception:
        return np.nan


@st.cache_data(show_spinner=False)
def league_table(year, group):
    """All-team season hitting/pitching from the MLB Stats API. Returns (DataFrame, error)."""
    try:
        url = (f"https://statsapi.mlb.com/api/v1/teams/stats?stats=season&group={group}"
               f"&season={year}&sportId=1")
        d = requests.get(url, timeout=15).json()
        splits = d["stats"][0]["splits"]
        rows = []
        for s in splits:
            stt = dict(s["stat"])
            stt["Team"] = ID2ABBR.get(s["team"]["id"], s["team"].get("abbreviation", ""))
            rows.append(stt)
        return pd.DataFrame(rows), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


@st.cache_data(show_spinner=False)
def team_season_batting(year):
    df, err = league_table(year, "hitting")
    if df is None or len(df) == 0:
        return None, err
    out = pd.DataFrame({"Team": df["Team"]})
    out["HR"] = _num(_col(df, "homeRuns"))
    out["3B"] = _num(_col(df, "triples"))
    out["2B"] = _num(_col(df, "doubles"))
    out["H"] = _num(_col(df, "hits"))
    out["BB"] = _num(_col(df, "baseOnBalls"))
    out["SO"] = _num(_col(df, "strikeOuts"))
    out["SB"] = _num(_col(df, "stolenBases"))
    out["R"] = _num(_col(df, "runs"))
    out["RBI"] = _num(_col(df, "rbi"))
    out["AB"] = _num(_col(df, "atBats"))
    out["PA"] = _num(_col(df, "plateAppearances"))
    out["SF"] = _num(_col(df, "sacFlies"))
    out["AVG"] = _num(_col(df, "avg"))
    out["OBP"] = _num(_col(df, "obp"))
    out["SLG"] = _num(_col(df, "slg"))
    out["OPS"] = _num(_col(df, "ops"))
    out["ISO"] = out["SLG"] - out["AVG"]
    denom = (out["AB"] - out["SO"] - out["HR"] + out["SF"]).replace(0, np.nan)
    out["BABIP"] = (out["H"] - out["HR"]) / denom
    out["K%"] = out["SO"] / out["PA"].replace(0, np.nan)
    out["BB%"] = out["BB"] / out["PA"].replace(0, np.nan)
    return out, None


def find_team_row(tb, abbr):
    if tb is None or len(tb) == 0:
        return None
    cands = {abbr.upper(), FG_ALIAS.get(abbr.upper(), abbr.upper())}
    for colname in ["Team", "team", "Tm", "TeamName", "Name"]:
        if colname in tb.columns:
            col = tb[colname].astype(str).str.upper()
            hit = tb[col.isin(cands)]
            if len(hit):
                return hit.iloc[0]
            for cand in cands:
                hit = tb[col.str.contains(cand, na=False, regex=False)]
                if len(hit):
                    return hit.iloc[0]
    return None


@st.cache_data(show_spinner=False)
def player_season_batting(year):
    try:
        return batting_stats(year, qual=1), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def find_player_row(df, full_name):
    if df is None or len(df) == 0 or "Name" not in df.columns:
        return None
    names = df["Name"].astype(str).str.lower()
    m = df[names == full_name.lower()]
    if len(m):
        return m.iloc[0]
    last = full_name.split()[-1].lower() if full_name.split() else full_name.lower()
    m = df[names.str.contains(last, na=False, regex=False)]
    return m.iloc[0] if len(m) else None


@st.cache_data(show_spinner=False)
def fg_team_pitching(year):
    df, err = league_table(year, "pitching")
    if df is None or len(df) == 0:
        return None, err
    out = pd.DataFrame({"Team": df["Team"]})
    out["ERA"] = _num(_col(df, "era"))
    out["WHIP"] = _num(_col(df, "whip"))
    bf = _num(_col(df, "battersFaced")).replace(0, np.nan)
    out["K%"] = _num(_col(df, "strikeOuts")) / bf
    out["BB%"] = _num(_col(df, "baseOnBalls")) / bf
    ipf = _col(df, "inningsPitched").map(_ip_to_float).replace(0, np.nan)
    out["HR/9"] = _num(_col(df, "homeRuns")) * 9.0 / ipf
    return out, None


@st.cache_data(show_spinner=False)
def player_season_row(pid, year, group):
    """A single player's season hitting/pitching line from the MLB Stats API (stat dict or None)."""
    try:
        url = (f"https://statsapi.mlb.com/api/v1/people/{pid}/stats"
               f"?stats=season&group={group}&season={year}")
        d = requests.get(url, timeout=10).json()
        splits = d["stats"][0]["splits"]
        return splits[0]["stat"] if splits else None
    except Exception:
        return None


def hitting_row(stat):
    if not stat:
        return None
    def g(k):
        return pd.to_numeric(stat.get(k), errors="coerce")
    s = {"HR": g("homeRuns"), "R": g("runs"), "RBI": g("rbi"), "H": g("hits"), "SB": g("stolenBases"),
         "3B": g("triples"), "2B": g("doubles"), "BB": g("baseOnBalls"), "SO": g("strikeOuts"),
         "AB": g("atBats"), "PA": g("plateAppearances"), "SF": g("sacFlies"),
         "AVG": g("avg"), "OBP": g("obp"), "SLG": g("slg"), "OPS": g("ops")}
    s["ISO"] = s["SLG"] - s["AVG"] if pd.notna(s["SLG"]) and pd.notna(s["AVG"]) else np.nan
    sf = s["SF"] if pd.notna(s["SF"]) else 0
    denom = (s["AB"] - s["SO"] - s["HR"] + sf) if all(pd.notna(s[k]) for k in ("AB", "SO", "HR")) else np.nan
    s["BABIP"] = (s["H"] - s["HR"]) / denom if pd.notna(denom) and denom else np.nan
    s["K%"] = s["SO"] / s["PA"] if pd.notna(s["PA"]) and s["PA"] else np.nan
    s["BB%"] = s["BB"] / s["PA"] if pd.notna(s["PA"]) and s["PA"] else np.nan
    return pd.Series(s)


def pitching_row(stat):
    if not stat:
        return None
    def g(k):
        return pd.to_numeric(stat.get(k), errors="coerce")
    bf = g("battersFaced")
    ipf = _ip_to_float(stat.get("inningsPitched"))
    s = {"ERA": g("era"), "WHIP": g("whip"),
         "K%": (g("strikeOuts") / bf) if pd.notna(bf) and bf else np.nan,
         "BB%": (g("baseOnBalls") / bf) if pd.notna(bf) and bf else np.nan,
         "HR/9": (g("homeRuns") * 9.0 / ipf) if pd.notna(ipf) and ipf else np.nan}
    return pd.Series(s)


@st.cache_data(show_spinner=False)
def fg_player_pitching(year):
    try:
        return pitching_stats(year, qual=1), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


TEAMS_LIST = sorted({"ARI", "ATL", "BAL", "BOS", "CHC", "CWS", "CIN", "CLE", "COL", "DET",
                     "HOU", "KC", "LAA", "LAD", "MIA", "MIL", "MIN", "NYM", "NYY", "ATH",
                     "PHI", "PIT", "SD", "SF", "SEA", "STL", "TB", "TEX", "TOR", "WSH"})

BAT_SPECS = [("AVG", "AVG", "avg"), ("OBP", "OBP", "avg"), ("SLG", "SLG", "avg"),
             ("OPS", "OPS", "avg"), ("ISO", "ISO", "avg"), ("BABIP", "BABIP", "avg"),
             ("K%", "K%", "pct"), ("BB%", "BB%", "pct")]
PIT_SPECS = [("ERA", "ERA", "f2"), ("WHIP", "WHIP", "f2"), ("K%", "K%", "pct"),
             ("BB%", "BB%", "pct"), ("HR/9", "HR/9", "f2")]


def fmt_stat(v, kind):
    if v is None:
        return "—"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    if pd.isna(v):
        return "—"
    if kind == "avg":
        return (f"{v:.3f}").lstrip("0")
    if kind == "int":
        return f"{int(round(v))}"
    if kind == "pct":
        vv = v * 100 if abs(v) <= 1.5 else v
        return f"{vv:.1f}%"
    if kind == "f2":
        return f"{v:.2f}"
    if kind == "f1":
        return f"{v:.1f}"
    return str(v)


def comparison_df(subject_label, subject_row, team_tbl, comp_abbr, specs):
    comp_row = find_team_row(team_tbl, comp_abbr)
    rows = []
    for label, col, kind in specs:
        pv = subject_row.get(col) if subject_row is not None else None
        mlb = team_tbl[col].mean() if (team_tbl is not None and col in team_tbl.columns) else None
        cv = comp_row.get(col) if comp_row is not None else None
        rows.append({"Stat": label, subject_label: fmt_stat(pv, kind),
                     "MLB avg": fmt_stat(mlb, kind), comp_abbr: fmt_stat(cv, kind)})
    return pd.DataFrame(rows)


NONE_OPT = "— none —"


def compare_table(subject_label, subject_row, team_tbl, comp_team, comp_player_label, comp_player_row, specs):
    """Comparison table: subject vs. MLB avg, plus an optional compare-team and/or compare-player column."""
    comp_team_row = find_team_row(team_tbl, comp_team) if (comp_team and comp_team != NONE_OPT) else None
    rows = []
    for label, col, kind in specs:
        d = {"Stat": label,
             subject_label: fmt_stat(subject_row.get(col) if subject_row is not None else None, kind),
             "MLB avg": fmt_stat(team_tbl[col].mean() if (team_tbl is not None and col in team_tbl.columns) else None, kind)}
        if comp_team_row is not None:
            d[comp_team] = fmt_stat(comp_team_row.get(col), kind)
        if comp_player_label and comp_player_row is not None:
            d[comp_player_label] = fmt_stat(comp_player_row.get(col), kind)
        rows.append(d)
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def projected_mix(pid, start, end):
    df = statcast_pitcher(start, end, pid)
    if df is None or len(df) == 0 or "pitch_name" not in df.columns:
        return None
    d = df.dropna(subset=["pitch_name", "game_date"]).copy()
    if len(d) == 0:
        return None
    d["game_date"] = pd.to_datetime(d["game_date"], errors="coerce")
    d = d.dropna(subset=["game_date"])
    ref = d["game_date"].max()
    age = (ref - d["game_date"]).dt.days
    d["w"] = 0.5 ** (age / 30.0)
    w = d.groupby("pitch_name")["w"].sum()
    return (w / w.sum() * 100).round(1).sort_values(ascending=False)


@st.cache_data(show_spinner=False)
def actual_game_mix(pid, date):
    df = statcast_pitcher(date, date, pid)
    if df is None or len(df) == 0 or "pitch_name" not in df.columns:
        return None
    return (df["pitch_name"].value_counts(normalize=True) * 100).round(1).to_dict()


@st.cache_data(show_spinner=False)
def names_for(ids):
    try:
        lk = playerid_reverse_lookup(list(ids), key_type="mlbam")
        return {int(r["key_mlbam"]): f"{str(r['name_first']).title()} {str(r['name_last']).title()}"
                for _, r in lk.iterrows()}
    except Exception:
        return {}


def log_path():
    try:
        base = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base = os.getcwd()
    return os.path.join(base, "projection_log.json")


def _gsheets_conn():
    """Google Sheets connection if configured (Streamlit secret + package); else None."""
    try:
        from streamlit_gsheets import GSheetsConnection
        return st.connection("gsheets", type=GSheetsConnection)
    except Exception:
        return None


def _load_log_local():
    pth = log_path()
    if os.path.exists(pth):
        try:
            return json.load(open(pth))
        except Exception:
            return []
    return []


def load_log():
    conn = _gsheets_conn()
    if conn is not None:
        try:
            df = conn.read(worksheet="log", ttl=0).dropna(how="all")
            recs = []
            for _, r in df.iterrows():
                if pd.isna(r.get("pitcher_id")):
                    continue
                proj, act, acc = r.get("projected"), r.get("actual"), r.get("accuracy")
                recs.append({
                    "game_date": str(r.get("game_date")) if pd.notna(r.get("game_date")) else None,
                    "pitcher_id": int(float(r["pitcher_id"])),
                    "pitcher": r.get("pitcher"),
                    "team": r.get("team"),
                    "projected": json.loads(proj) if isinstance(proj, str) and proj.strip() else {},
                    "graded": str(r.get("graded")).strip().lower() in ("true", "1", "yes"),
                    "actual": json.loads(act) if isinstance(act, str) and act.strip() else None,
                    "accuracy": float(acc) if pd.notna(acc) and str(acc).strip() != "" else None,
                })
            return recs
        except Exception:
            pass
    return _load_log_local()


def save_log(recs):
    conn = _gsheets_conn()
    if conn is not None:
        try:
            cols = ["game_date", "pitcher_id", "pitcher", "team", "projected", "graded", "actual", "accuracy"]
            rows = [{
                "game_date": r.get("game_date"),
                "pitcher_id": r.get("pitcher_id"),
                "pitcher": r.get("pitcher"),
                "team": r.get("team"),
                "projected": json.dumps(r.get("projected") or {}),
                "graded": bool(r.get("graded")),
                "actual": json.dumps(r.get("actual")) if r.get("actual") is not None else "",
                "accuracy": r.get("accuracy") if r.get("accuracy") is not None else "",
            } for r in recs]
            conn.update(worksheet="log", data=pd.DataFrame(rows, columns=cols))
            return True
        except Exception:
            pass
    try:
        json.dump(recs, open(log_path(), "w"), indent=2)
        return True
    except Exception:
        return False


PITCH_LOG_COLS = ["logged_at", "pitcher", "situation", "predicted", "pred_prob", "actual", "velo", "correct"]


def pitch_log_path():
    try:
        base = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base = os.getcwd()
    return os.path.join(base, "pitch_log.json")


def load_pitch_log():
    conn = _gsheets_conn()
    if conn is not None:
        try:
            df = conn.read(worksheet="pitch_log", ttl=0)
            if "logged_at" in df.columns:
                df = df[df["logged_at"].astype(str).str.strip() != ""]
            else:
                df = df.dropna(how="all")
            return df.to_dict("records")
        except Exception:
            pass
    pth = pitch_log_path()
    if os.path.exists(pth):
        try:
            return json.load(open(pth))
        except Exception:
            return []
    return []


def save_pitch_log(recs):
    conn = _gsheets_conn()
    if conn is not None:
        try:
            try:
                prev_n = int(conn.read(worksheet="pitch_log", ttl=0).shape[0])
            except Exception:
                prev_n = 0
            df = pd.DataFrame(recs, columns=PITCH_LOG_COLS)
            if len(df) < prev_n:  # pad with blank rows so deleted rows get overwritten
                blanks = pd.DataFrame([{c: "" for c in PITCH_LOG_COLS} for _ in range(prev_n - len(df))])
                df = pd.concat([df, blanks], ignore_index=True)
            conn.update(worksheet="pitch_log", data=df)
            return True
        except Exception:
            pass
    try:
        json.dump(recs, open(pitch_log_path(), "w"), indent=2)
        return True
    except Exception:
        return False


# ---- remembered search-form inputs (persist across restarts) ----
# Every search box that should keep its last value across app restarts.
PREF_KEYS = [
    "h_last", "h_first", "h_cs", "h_ce", "h_bs", "h_be", "comp_h", "h_cmpl", "h_cmpf",
    "p_last", "p_first", "p_cs", "p_ce", "comp_p", "p_cmpl", "p_cmpf",
    "f_last", "f_first", "f_team", "f_cs", "f_ce",
    "t_team", "t_cs", "t_ce", "comp_t",
    "ng_team",
    "g_team", "g_date",
    "mu_pl", "mu_pf", "mu_bl", "mu_bf", "mu_cs", "mu_ce",
    "pr_last", "pr_first", "pr_cs", "pr_ce",
]


def prefs_path():
    try:
        base = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base = os.getcwd()
    return os.path.join(base, "prefs.json")


def load_prefs():
    """Last-used search-box values as {key: value}. Google Sheet 'prefs', else local file."""
    conn = _gsheets_conn()
    if conn is not None:
        try:
            df = conn.read(worksheet="prefs", ttl=0).dropna(how="all")
            out = {}
            for _, r in df.iterrows():
                k = r.get("key")
                if isinstance(k, str) and k.strip():
                    v = r.get("value")
                    out[k.strip()] = "" if pd.isna(v) else str(v)
            return out
        except Exception:
            pass
    pth = prefs_path()
    if os.path.exists(pth):
        try:
            return json.load(open(pth))
        except Exception:
            return {}
    return {}


def save_prefs(d):
    d = {k: ("" if d.get(k) is None else str(d.get(k))) for k in PREF_KEYS}
    conn = _gsheets_conn()
    if conn is not None:
        try:
            df = pd.DataFrame([{"key": k, "value": v} for k, v in d.items()], columns=["key", "value"])
            conn.update(worksheet="prefs", data=df)
            return True
        except Exception:
            pass
    try:
        json.dump(d, open(prefs_path(), "w"), indent=2)
        return True
    except Exception:
        return False


def persist_inputs():
    """Snapshot the current value of every remembered search box and save it."""
    save_prefs({k: st.session_state.get(k, "") for k in PREF_KEYS})


def bio_caption(height_ft, hs, w):
    if hs and w:
        return f"Listed at {hs}, {w} lb — figure drawn to scale."
    if hs:
        return f"Listed at {hs} — figure drawn to scale."
    return "Height unavailable; figure uses a 6'0\" default."


def chase_frame(df):
    d = df.dropna(subset=["plate_x", "plate_z", "sz_top", "sz_bot", "description"]).copy()
    out_x = d["plate_x"].abs() > 0.83
    out_z = (d["plate_z"] > d["sz_top"]) | (d["plate_z"] < d["sz_bot"])
    d["out_of_zone"] = out_x | out_z
    d["swung"] = d["description"].isin(SWING_DESCS)
    oz = d[d["out_of_zone"]]
    chase = oz["swung"].mean() if len(oz) else float("nan")
    return d, len(oz), chase


def outcome_counts(df):
    if df is None or "events" not in df.columns:
        return None
    ev = df.dropna(subset=["events"])
    if len(ev) == 0:
        return None
    cat = ev["events"].map(lambda e: HIT_MAP.get(e, "Out/other"))
    counts = cat.value_counts().reindex(OUTCOME_ORDER, fill_value=0)
    return len(ev), counts


def pitches_seen(df):
    if df is None or "description" not in df.columns:
        return None
    d = df.dropna(subset=["description"])
    if len(d) == 0:
        return None
    cat = d["description"].map(lambda x: PITCH_SEEN_MAP.get(x, "Other"))
    counts = cat.value_counts().reindex(PITCH_SEEN_ORDER, fill_value=0)
    return len(d), counts[counts > 0]


def mix_table(series_counts):
    total = series_counts.sum()
    out = series_counts.rename_axis("Pitch").reset_index(name="Count")
    out["Usage %"] = (out["Count"] / total * 100).round(1)
    return out


PITCH_STAT_ORDER = ["Pitches", "Strike %", "Whiff %", "Chase %", "Zone %", "K %", "BB %",
                    "BA against", "Hits", "HR", "PAs", "xwOBA/contact",
                    "Avg exit velo", "Hard-hit %", "Barrel %"]


def pitch_stats(sub):
    """Formatted pitching/contact stats for a subset of Statcast rows (one perspective)."""
    if sub is None or len(sub) == 0:
        return {}
    desc = sub["description"] if "description" in sub.columns else pd.Series(dtype=object)
    ev = sub["events"] if "events" in sub.columns else pd.Series(dtype=object)
    pa = int(ev.notna().sum())
    sw = int(desc.isin(SWING_DESCS).sum())
    wh = int(desc.isin({"swinging_strike", "swinging_strike_blocked"}).sum())
    zone = chase = float("nan")
    zc = {"plate_x", "plate_z", "sz_top", "sz_bot"}
    if zc.issubset(sub.columns):
        z = sub.dropna(subset=list(zc))
        if len(z):
            inz = (z["plate_x"].abs() <= 0.83) & (z["plate_z"] <= z["sz_top"]) & (z["plate_z"] >= z["sz_bot"])
            zone = inz.mean() * 100
            ozm = z[~inz]
            chase = ozm["description"].isin(SWING_DESCS).mean() * 100 if len(ozm) else float("nan")

    def evc(n):
        return int(ev.isin(n).sum())

    k = evc({"strikeout", "strikeout_double_play"})
    bb = evc({"walk"})
    hbp = evc({"hit_by_pitch"})
    sac = evc({"sac_fly", "sac_bunt", "sac_fly_double_play"})
    hits = evc(HIT_EVENTS)
    hr = evc({"home_run"})
    ab = max(pa - bb - hbp - sac, 0)
    out = {"Pitches": f"{len(sub):,}",
           "Strike %": f"{(sub['type'] == 'S').mean() * 100:.1f}%" if "type" in sub.columns else "—",
           "Whiff %": f"{wh / sw * 100:.1f}%" if sw else "—",
           "Chase %": f"{chase:.1f}%" if pd.notna(chase) else "—",
           "Zone %": f"{zone:.1f}%" if pd.notna(zone) else "—",
           "K %": f"{k / pa * 100:.1f}%" if pa else "—",
           "BB %": f"{bb / pa * 100:.1f}%" if pa else "—",
           "BA against": (f"{hits / ab:.3f}").lstrip("0") if ab else "—",
           "Hits": f"{hits}", "HR": f"{hr}", "PAs": f"{pa}"}
    if XWOBA in sub.columns:
        xw = sub[XWOBA].dropna()
        out["xwOBA/contact"] = (f"{xw.mean():.3f}").lstrip("0") if len(xw) else "—"
    if "launch_speed" in sub.columns:
        ls = sub["launch_speed"].dropna()
        if len(ls):
            out["Avg exit velo"] = f"{ls.mean():.1f} mph"
            out["Hard-hit %"] = f"{(ls >= 95).mean() * 100:.1f}%"
    if "launch_speed_angle" in sub.columns:
        la = sub["launch_speed_angle"].dropna()
        if len(la):
            out["Barrel %"] = f"{(la == 6).mean() * 100:.1f}%"
    return out


st.markdown(
    """
    <div style="background:linear-gradient(135deg,#eef3fb 0%,#ffffff 70%);
                border:1px solid #e2e8f2;border-left:5px solid #2563eb;
                border-radius:14px;padding:18px 22px;margin-bottom:8px;">
      <div style="font-size:1.7rem;font-weight:800;letter-spacing:-0.5px;color:#0f1722;">
        ⚾ MLB Statcast Analysis</div>
      <div style="color:#5b6472;font-size:0.92rem;margin-top:4px;">
        Hitter approach · pitcher deception · fielding · team trends · live pitch predictor · pitch guide
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption(
    "Powered by Statcast via pybaseball. First data pull can take 30–60 seconds and may rate-limit; "
    "if it errors, wait a minute and run again."
)
st.markdown(
    "<small><i>Tip: hover over any <abbr title='Like this!' "
    "style='text-decoration:underline dotted;cursor:help'>dotted-underlined</abbr> term for its definition, "
    "or open the Glossary in each tab.</i></small>",
    unsafe_allow_html=True,
)

# Restore last-used search-box values once per session (skips blanks so selectboxes stay valid).
if "_prefs_loaded" not in st.session_state:
    for _k, _v in load_prefs().items():
        if _k in PREF_KEYS and isinstance(_v, str) and _v != "":
            st.session_state.setdefault(_k, _v)
    st.session_state["_prefs_loaded"] = True

tab_h, tab_p, tab_f, tab_t, tab_game, tab_mu, tab_m, tab_pred, tab_g, tab_about = st.tabs(
    ["Hitter diagnosis", "Pitcher deception", "Fielding", "Team stats", "Game overview", "Matchup",
     "Next game", "Pitch predictor", "Pitch guide", "About"]
)

with tab_h:
    st.subheader("Hitter approach diagnosis")
    glossary_expander()
    with st.form("hitter_form"):
        c1, c2 = st.columns(2)
        last = c1.text_input("Hitter last name", key="h_last")
        first = c2.text_input("Hitter first name", key="h_first")
        c3, c4 = st.columns(2)
        cur_start = c3.text_input("Current period start (YYYY-MM-DD)", placeholder="YYYY-MM-DD", key="h_cs")
        cur_end = c4.text_input("Current period end (YYYY-MM-DD)", placeholder="YYYY-MM-DD", key="h_ce")
        c5, c6 = st.columns(2)
        base_start = c5.text_input("Baseline start (YYYY-MM-DD)", placeholder="optional — YYYY-MM-DD", key="h_bs")
        base_end = c6.text_input("Baseline end (YYYY-MM-DD)", placeholder="optional — YYYY-MM-DD", key="h_be")
        cc7, cc8, cc9 = st.columns(3)
        comp_team_h = cc7.selectbox("Compare to team (optional)", [NONE_OPT] + TEAMS_LIST, index=0, key="comp_h")
        cmp_last_h = cc8.text_input("Compare to player — last name (optional)", key="h_cmpl")
        cmp_first_h = cc9.text_input("Compare to player — first name (optional)", key="h_cmpf")
        go_h = st.form_submit_button("Run hitter analysis")

    if go_h:
        persist_inputs()
        if not (cur_start.strip() and cur_end.strip()):
            st.warning("Enter the current-period start and end dates (YYYY-MM-DD).")
            st.stop()
        pid, name = lookup_id(last, first)
        if pid is None:
            st.error(f"No player found for '{first} {last}'. Check the spelling.")
        else:
            try:
                with st.spinner(f"Pulling Statcast data for {name}…"):
                    cur = pull_batter(cur_start, cur_end, pid)
                    base = pull_batter(base_start, base_end, pid)
            except Exception as e:
                st.error("Data pull failed. Often this is Baseball Savant rate-limiting — wait a minute and click Run again. If it keeps failing, expand the details below and share them.")
                with st.expander("Show technical details"):
                    st.exception(e)
                st.stop()

            if cur is None or len(cur) == 0:
                st.warning("No data returned for the current period. Try a wider date range.")
                st.stop()

            has_base = base is not None and len(base) > 0
            bat_h_ft, bat_hs, bat_w = player_bio(pid)
            st.markdown(f"### {name}")

            cur_d, cur_oz, cur_chase = chase_frame(cur)
            base_d, base_oz, base_chase = chase_frame(base) if has_base else (None, 0, float("nan"))

            m1, m2, m3 = st.columns(3)
            m1.metric("Current chase rate", pct(cur_chase), help=GLOSSARY["Chase rate"])
            m2.metric("Baseline chase rate", pct(base_chase), help=GLOSSARY["Baseline"])
            if pd.notna(cur_chase) and pd.notna(base_chase):
                delta = (cur_chase - base_chase) * 100
                m3.metric("Change", f"{delta:+.1f} pp", delta=f"{delta:+.1f} pp", delta_color="inverse",
                          help="Percentage-point change in chase rate vs the baseline. Down (green) is better.")
            else:
                m3.metric("Change", "n/a")
            st.markdown(
                f"<small>{term('Chase rate')} = swings at pitches {term('Out of zone')}, versus the "
                f"{term('Baseline')} period.</small>",
                unsafe_allow_html=True,
            )

            st.markdown("#### Pitches seen (this period)")
            ps = pitches_seen(cur)
            if ps:
                total, pcounts = ps
                ptbl = pd.DataFrame({"Pitch result": pcounts.index, "Count": pcounts.values,
                                     "Share": (pcounts.values / total * 100).round(1)})
                ptbl["Share"] = ptbl["Share"].astype(str) + "%"
                st.dataframe(ptbl, hide_index=True, use_container_width=True)
                st.caption(f"Across {total} total pitches seen. Whiffs are swinging strikes; fouls count as strikes "
                           "(except with two strikes); 'In play' is any batted ball.")
                ev_d = cur.dropna(subset=["description"])
                sw_mask = ev_d["description"].isin(SWING_DESCS)
                ct_mask = ev_d["description"].isin({"hit_into_play", "foul", "foul_tip", "foul_bunt"})
                n_sw = int(sw_mask.sum())
                n_ct = int(ct_mask.sum())
                sm = st.columns(4)
                sm[0].metric("Swings", n_sw, help="Total swings = whiffs + fouls + balls in play.")
                sm[1].metric("Swing %", f"{n_sw / total * 100:.1f}%" if total else "n/a",
                             help="Swings ÷ pitches seen.")
                sm[2].metric("Contact %", f"{n_ct / n_sw * 100:.1f}%" if n_sw else "n/a",
                             help="Of his swings, how often he made contact (foul or ball in play).")
                sm[3].metric("Whiff %", f"{(n_sw - n_ct) / n_sw * 100:.1f}%" if n_sw else "n/a",
                             help="Of his swings, how often he missed entirely.")
                st.caption(f"He swung at {n_sw} of {total} pitches and made contact on {n_ct} of those swings "
                           f"({n_sw - n_ct} swings and misses).")
                n_foul = int(ev_d["description"].isin({"foul", "foul_tip", "foul_bunt"}).sum())
                pa_seen = int(cur["events"].notna().sum()) if "events" in cur.columns else 0
                fm = st.columns(3)
                fm[0].metric("Foul balls", n_foul)
                fm[1].metric("Foul % of swings", f"{n_foul / n_sw * 100:.1f}%" if n_sw else "n/a",
                             help="Fouls ÷ swings — how often he just fouls a pitch off.")
                fm[2].metric("Fouls per PA", f"{n_foul / pa_seen:.2f}" if pa_seen else "n/a")
            else:
                st.info("No pitch-level data to summarize for this period.")

            st.markdown("#### By pitch type — what he swings at & hits")
            if "pitch_name" in cur.columns and cur["pitch_name"].notna().any():
                ptd = cur.dropna(subset=["pitch_name"]).copy()
                ptd["is_swing"] = ptd["description"].isin(SWING_DESCS)
                ptd["is_whiff"] = ptd["description"].isin({"swinging_strike", "swinging_strike_blocked"})
                ptd["is_inplay"] = ptd["description"] == "hit_into_play"
                ptd["is_hit"] = ptd["events"].isin(HIT_EVENTS) if "events" in ptd.columns else False
                ptrows = []
                for nm, g in ptd.groupby("pitch_name"):
                    seen = len(g)
                    sw = int(g["is_swing"].sum())
                    xw = g[XWOBA].dropna() if XWOBA in g.columns else pd.Series(dtype=float)
                    ev = g["events"] if "events" in g.columns else pd.Series(dtype=object)
                    outs = g[g["is_inplay"] & ~g["is_hit"]]
                    bt = outs["bb_type"] if "bb_type" in outs.columns else pd.Series(dtype=object)
                    ptrows.append({
                        "Pitch": nm,
                        "Seen": seen,
                        "Swing %": f"{sw / seen * 100:.0f}%" if seen else "—",
                        "Whiff %": f"{int(g['is_whiff'].sum()) / sw * 100:.0f}%" if sw else "—",
                        "HR": int((ev == "home_run").sum()),
                        "3B": int((ev == "triple").sum()),
                        "2B": int((ev == "double").sum()),
                        "1B": int((ev == "single").sum()),
                        "GB out": int((bt == "ground_ball").sum()),
                        "Fly out": int((bt == "fly_ball").sum()),
                        "Line out": int((bt == "line_drive").sum()),
                        "Popup": int((bt == "popup").sum()),
                        "xwOBA/contact": (f"{xw.mean():.3f}").lstrip("0") if len(xw) else "—",
                    })
                ptdf = pd.DataFrame(ptrows).sort_values("Seen", ascending=False)
                st.dataframe(ptdf, hide_index=True, use_container_width=True)
                st.caption(
                    "Per pitch type he saw this period: **Swing %** = how often he offered, **Whiff %** = misses per swing. "
                    "Hits (**HR / 3B / 2B / 1B**), then how his outs were made (**GB out** = ground out, **Fly out**, "
                    "**Line out**, **Popup**), and **xwOBA/contact** = expected value of his contact. Scroll right to see all columns."
                )
            else:
                st.info("No pitch-type data for this period.")

            st.markdown("#### Outcome odds (this period)")
            cur_oc = outcome_counts(cur)
            if cur_oc:
                pa_n, counts = cur_oc
                tbl = pd.DataFrame({"Outcome": OUTCOME_ORDER, "Current": (counts / pa_n).values})
                if has_base:
                    boc = outcome_counts(base)
                    if boc:
                        b_pa, b_counts = boc
                        tbl["Baseline"] = (b_counts / b_pa).values
                show = tbl.copy()
                for col in [c for c in ("Current", "Baseline") if c in show.columns]:
                    show[col] = (show[col] * 100).round(1).astype(str) + "%"
                st.dataframe(show, hide_index=True, use_container_width=True)
                st.markdown(
                    f"<small>Per {term('Plate appearance')} over {pa_n} PAs in the current period. These are this "
                    "hitter's actual rates over the window you chose — descriptive odds, not a context-aware "
                    "prediction (true odds depend on the pitcher, count, and park).</small>",
                    unsafe_allow_html=True,
                )
                xb = cur[XBA].dropna() if XBA in cur.columns else pd.Series(dtype=float)
                xw = cur[XWOBA].dropna() if XWOBA in cur.columns else pd.Series(dtype=float)
                xc1, xc2 = st.columns(2)
                if len(xb):
                    xc1.metric("xBA on contact", f"{xb.mean():.3f}", help=GLOSSARY["xBA"])
                if len(xw):
                    xc2.metric("xwOBA on contact", f"{xw.mean():.3f}", help=GLOSSARY["xwOBA"])
            else:
                st.info("No completed plate appearances in this period to compute outcome odds.")

            st.markdown("#### Situational hitting (runners on base)")
            sit = cur.dropna(subset=["events"]).copy() if "events" in cur.columns else cur.iloc[0:0]
            if len(sit):
                def on(colp):
                    return sit[colp].notna() if colp in sit.columns else pd.Series(False, index=sit.index)
                any_on = on("on_1b") | on("on_2b") | on("on_3b")
                risp = on("on_2b") | on("on_3b")

                def split_stats(maskv, labelname):
                    s = sit[maskv]
                    pa_ = len(s)
                    h_ = int(s["events"].isin(HIT_EVENTS).sum())
                    bb_ = int((s["events"] == "walk").sum())
                    hbp_ = int((s["events"] == "hit_by_pitch").sum())
                    sac_ = int(s["events"].isin({"sac_fly", "sac_bunt", "sac_fly_double_play"}).sum())
                    ab_ = max(pa_ - bb_ - hbp_ - sac_, 0)
                    hr_ = int((s["events"] == "home_run").sum())
                    avg = (f"{h_ / ab_:.3f}").lstrip("0") if ab_ else "—"
                    return {"Situation": labelname, "PA": pa_, "AB": ab_, "H": h_, "HR": hr_, "AVG": avg}

                rows_ = [split_stats(~any_on, "Bases empty"),
                         split_stats(any_on, "Runners on"),
                         split_stats(risp, "RISP (2nd/3rd)")]
                st.dataframe(pd.DataFrame(rows_), hide_index=True, use_container_width=True)
                st.caption("How he hit by base state in the current period. RISP = runners in scoring position; "
                           "AVG = hits ÷ at-bats in that split.")
            else:
                st.info("No completed plate appearances to split by base state.")

            yr_b = int(cur_start[:4]) if cur_start[:4].isdigit() else dt.date.today().year
            prow = hitting_row(player_season_row(pid, yr_b, "hitting"))
            st.markdown(f"#### Season counting stats ({yr_b})")
            if prow is not None:
                def pgi(col):
                    v = prow.get(col)
                    return f"{int(v)}" if pd.notna(v) else "—"
                rr = st.columns(5)
                rr[0].metric("RBI", pgi("RBI"))
                rr[1].metric("Runs", pgi("R"))
                rr[2].metric("HR", pgi("HR"))
                rr[3].metric("Hits", pgi("H"))
                rr[4].metric("Stolen bases", pgi("SB"))
                st.caption("Full-season totals from the MLB Stats API. RBI isn't in pitch-level Statcast, so it comes from "
                           "season stats.")
            else:
                st.caption(f"Season RBI/counting totals unavailable for {name} in {yr_b}.")

            tb_cmp, tb_cmp_err = team_season_batting(yr_b)
            cmp_row_h, cmp_label_h = None, None
            if cmp_last_h.strip():
                cpid, cpname = lookup_id(cmp_last_h, cmp_first_h)
                if cpid:
                    cmp_row_h = hitting_row(player_season_row(cpid, yr_b, "hitting"))
                    cmp_label_h = cpname
                else:
                    st.caption(f"Couldn't find a player named '{cmp_first_h} {cmp_last_h}' to compare.")
            st.markdown("#### Rate-stat comparison")
            if prow is not None and tb_cmp is not None:
                st.dataframe(compare_table(name, prow, tb_cmp, comp_team_h, cmp_label_h, cmp_row_h, BAT_SPECS),
                             hide_index=True, use_container_width=True)
                bits = ["the MLB average"]
                if comp_team_h != NONE_OPT:
                    bits.append(comp_team_h)
                if cmp_label_h:
                    bits.append(cmp_label_h)
                st.caption(f"{name}'s {yr_b} rate stats vs. {', '.join(bits)}. From the MLB Stats API.")
            else:
                st.info("Season comparison stats unavailable for this player/season.")

            sw = cur_d[cur_d["swung"]]
            if len(sw) >= 5:
                bh = bat_h_ft if bat_h_ft else 6.0
                fig, ax = plt.subplots(figsize=(4.4, 4.8))
                hb = ax.hexbin(sw["plate_x"], sw["plate_z"], gridsize=25, cmap="Reds", mincnt=1)
                zt, zb = cur_d["sz_top"].mean(), cur_d["sz_bot"].mean()
                ax.plot([-0.83, 0.83, 0.83, -0.83, -0.83],
                        [zb, zb, zt, zt, zb], "b-", linewidth=2, label="Strike zone")
                draw_person(ax, bh, x0=-2.1, color="0.82", zorder=1)
                ax.set_xlabel("Horizontal location (ft, catcher view)")
                ax.set_ylabel("Height (ft)")
                ax.set_title("Swings by location (red outside box = chases)", fontsize=10)
                ax.set_xlim(-2.8, 2.5)
                ax.set_ylim(0, max(5.2, bh + 0.5))
                ax.set_aspect("equal")
                ax.legend(fontsize=8)
                fig.colorbar(hb, label="swings")
                st.pyplot(fig, use_container_width=False)
                st.caption("Gray figure = the batter, " + bio_caption(bat_h_ft, bat_hs, bat_w)
                           + " The blue box is the strike zone, at his knees-to-chest height.")
            else:
                st.info("Not enough swings in the current period to draw a swing map.")

            st.markdown("#### Spray chart — where he hit the ball")
            bip = cur.dropna(subset=["hc_x", "hc_y"]) if {"hc_x", "hc_y"}.issubset(cur.columns) else cur.iloc[0:0]
            if "events" in cur.columns and len(bip):
                bip = bip[bip["events"].notna()]
            if len(bip) >= 3:
                EVMAP = {"home_run": "HR", "triple": "3B", "double": "2B", "single": "1B"}
                HITCOLOR = {"HR": "#d62728", "3B": "#9467bd", "2B": "#1f77b4", "1B": "#2ca02c", "Out/other": "#9aa0a6"}
                X = (bip["hc_x"] - 125.42) * 2.5
                Y = (198.27 - bip["hc_y"]) * 2.5
                labels = bip["events"].map(lambda e: EVMAP.get(e, "Out/other"))
                figs, axs = plt.subplots(figsize=(4.4, 4.4))
                draw_field(axs)
                for lab in ["Out/other", "1B", "2B", "3B", "HR"]:
                    sel = labels == lab
                    axs.scatter(X[sel], Y[sel], s=22, alpha=0.75, label=lab, color=HITCOLOR[lab], zorder=4)
                xmax = max(180, float(np.nanmax(np.abs(X))) + 25)
                ymax = max(260, float(np.nanmax(Y)) + 25)
                axs.set_xlim(-xmax, xmax)
                axs.set_ylim(-20, ymax)
                axs.set_aspect("equal")
                axs.set_xticks([])
                axs.set_yticks([])
                axs.set_title(f"Spray chart — where {name} hit the ball", fontsize=10)
                axs.legend(fontsize=7, loc="upper right")
                st.pyplot(figs, use_container_width=False)
                st.caption("To-scale field in feet: home plate at bottom. Each dot is a batted ball in the current "
                           "period, colored by result — so you can see his pull/oppo tendencies and where the damage lands.")
            else:
                st.info("Not enough batted balls with location data for a spray chart.")

            bb = cur.dropna(subset=[XWOBA]).sort_values("game_date").copy()
            if len(bb) >= 20:
                bb["roll"] = bb[XWOBA].rolling(50, min_periods=20).mean()
                fig2, ax2 = plt.subplots(figsize=(5.2, 2.3))
                ax2.plot(range(len(bb)), bb["roll"], linewidth=2)
                ax2.axhline(0.320, color="gray", ls="--", label="~league-average xwOBA")
                ax2.set_xlabel("Batted balls (chronological)")
                ax2.set_ylabel("rolling xwOBA")
                ax2.set_title("Contact-quality trend", fontsize=10)
                ax2.legend(fontsize=8)
                st.pyplot(fig2, use_container_width=False)
                st.markdown(
                    f"<small>{term('Rolling xwOBA')} tracks {term('Contact quality')} over time.</small>",
                    unsafe_allow_html=True,
                )
            else:
                st.info("Not enough batted balls in the current period for a rolling xwOBA trend.")

            st.markdown("#### What to work on")
            recs = []
            if pd.notna(cur_chase) and pd.notna(base_chase):
                cd = (cur_chase - base_chase) * 100
                if cd > 2:
                    ch = cur_d[(cur_d["out_of_zone"]) & (cur_d["swung"])]
                    up = int((ch["plate_z"] > ch["sz_top"]).sum())
                    down = int((ch["plate_z"] < ch["sz_bot"]).sum())
                    side = int(((ch["plate_z"] <= ch["sz_top"]) & (ch["plate_z"] >= ch["sz_bot"])
                                & (ch["plate_x"].abs() > 0.83)).sum())
                    region = max([("up", up), ("down", down), ("to the sides", side)], key=lambda t: t[1])[0]
                    recs.append(
                        f"**Tighten zone discipline.** Chase rate is up {cd:+.1f} pp vs baseline, and those swings "
                        f"are leaking mostly **{region}** out of the zone. Laying off there cuts weak contact and strikeouts."
                    )
                elif cd < -2:
                    recs.append(f"**Approach is sharp** — chase rate is down {cd:+.1f} pp vs baseline, so plate discipline isn't the problem.")
                else:
                    recs.append("**Approach is stable** — chase rate is essentially unchanged from baseline, so swing decisions aren't the issue.")

            cur_xw = cur[XWOBA].dropna().mean() if XWOBA in cur.columns else float("nan")
            base_xw = base[XWOBA].dropna().mean() if has_base and XWOBA in base.columns else float("nan")
            if pd.notna(cur_xw) and pd.notna(base_xw):
                cq = cur_xw - base_xw
                if cq < -0.015:
                    recs.append(
                        f"**Contact quality is down** ({cur_xw:.3f} vs {base_xw:.3f} xwOBA on contact). With discipline holding, "
                        "this points to timing/mechanics or pitch selection inside the zone — worth a swing/launch-angle look."
                    )
                elif cq > 0.015:
                    recs.append(
                        f"**Contact quality is up** ({cur_xw:.3f} vs {base_xw:.3f}). The bat is producing; if results lag, that's likely variance that should correct."
                    )
                else:
                    recs.append(f"**Contact quality is steady** (~{cur_xw:.3f} xwOBA on contact vs {base_xw:.3f} baseline).")

            if not recs:
                recs.append("Add a baseline period to generate comparison-based guidance.")
            for r in recs:
                st.markdown("- " + r)
            st.caption(
                "Heuristic guidance derived from the metrics above (chase-rate change, where swings leave the zone, "
                "and contact-quality change). A data-driven read, not professional coaching."
            )

with tab_p:
    st.subheader("Pitcher deception")
    glossary_expander()
    pitch_types_expander()
    with st.form("pitcher_form"):
        c1, c2 = st.columns(2)
        plast = c1.text_input("Pitcher last name", key="p_last")
        pfirst = c2.text_input("Pitcher first name", key="p_first")
        c3, c4 = st.columns(2)
        p_start = c3.text_input("Start date (YYYY-MM-DD)", placeholder="YYYY-MM-DD", key="p_cs")
        p_end = c4.text_input("End date (YYYY-MM-DD)", placeholder="YYYY-MM-DD", key="p_ce")
        pc7, pc8, pc9 = st.columns(3)
        comp_team_p = pc7.selectbox("Compare to team (optional)", [NONE_OPT] + TEAMS_LIST, index=0, key="comp_p")
        cmp_last_p = pc8.text_input("Compare to pitcher — last name (optional)", key="p_cmpl")
        cmp_first_p = pc9.text_input("Compare to pitcher — first name (optional)", key="p_cmpf")
        go_p = st.form_submit_button("Run pitcher analysis")

    if go_p:
        persist_inputs()
        if not (p_start.strip() and p_end.strip()):
            st.warning("Enter the start and end dates (YYYY-MM-DD).")
            st.stop()
        pid, name = lookup_id(plast, pfirst)
        if pid is None:
            st.error(f"No player found for '{pfirst} {plast}'. Check the spelling.")
        else:
            try:
                with st.spinner(f"Pulling Statcast data for {name}…"):
                    pdf = pull_pitcher(p_start, p_end, pid)
            except Exception as e:
                st.error("Data pull failed. Often this is Baseball Savant rate-limiting — wait a minute and click Run again. If it keeps failing, expand the details below and share them.")
                with st.expander("Show technical details"):
                    st.exception(e)
                st.stop()

            if pdf is None or len(pdf) == 0:
                st.warning("No data returned for this range. Try different dates.")
                st.stop()

            pit_h_ft, pit_hs, pit_w = player_bio(pid)
            st.markdown(f"### {name}")

            yr_p = int(p_start[:4]) if p_start[:4].isdigit() else dt.date.today().year
            tp_tbl, _tp_err = fg_team_pitching(yr_p)
            prow_p = pitching_row(player_season_row(pid, yr_p, "pitching"))
            cmp_row_p, cmp_label_p = None, None
            if cmp_last_p.strip():
                cppid, cppname = lookup_id(cmp_last_p, cmp_first_p)
                if cppid:
                    cmp_row_p = pitching_row(player_season_row(cppid, yr_p, "pitching"))
                    cmp_label_p = cppname
                else:
                    st.caption(f"Couldn't find a pitcher named '{cmp_first_p} {cmp_last_p}' to compare.")
            st.markdown("#### Rate-stat comparison")
            if prow_p is not None and tp_tbl is not None:
                st.dataframe(compare_table(name, prow_p, tp_tbl, comp_team_p, cmp_label_p, cmp_row_p, PIT_SPECS),
                             hide_index=True, use_container_width=True)
                bits = ["the MLB average"]
                if comp_team_p != NONE_OPT:
                    bits.append(comp_team_p)
                if cmp_label_p:
                    bits.append(cmp_label_p)
                st.caption(f"{name}'s {yr_p} rate stats vs. {', '.join(bits)}. From the MLB Stats API.")
            else:
                st.info("Season comparison stats unavailable for this pitcher/season.")

            cols = ["pitch_name", "release_pos_x", "release_pos_z", "plate_x", "plate_z", "release_speed"]
            p = pdf[cols].dropna().copy()
            counts = p["pitch_name"].value_counts(normalize=True)
            keep = counts[counts >= 0.05].index
            p = p[p["pitch_name"].isin(keep)]

            if p["pitch_name"].nunique() < 2:
                st.info("Not enough pitch-type variety in this range for a deception comparison.")
            else:
                ph = pit_h_ft if pit_h_ft else 6.0
                colL, colR = st.columns(2)
                with colL:
                    fig, ax = plt.subplots(figsize=(4.0, 4.4))
                    sh = draw_pitcher(ax, height_ft=ph, body_x=0.0)
                    mrx, mrz = p["release_pos_x"].mean(), p["release_pos_z"].mean()
                    ax.plot([0, mrx], [sh, mrz], color="0.6", lw=1, ls="--", zorder=1)
                    for nm, grp in p.groupby("pitch_name"):
                        ax.scatter(grp["release_pos_x"], grp["release_pos_z"], alpha=0.3, s=14, label=nm, zorder=4)
                    ax.set_xlabel("Horizontal release (ft, catcher view)")
                    ax.set_ylabel("Release height (ft)")
                    ax.set_title("Release vs. pitcher on the mound", fontsize=10)
                    ax.set_xlim(-4, 4)
                    ax.set_ylim(0, max(7, ph + 0.8))
                    ax.set_aspect("equal")
                    ax.legend(fontsize=7, loc="upper right")
                    st.pyplot(fig, use_container_width=False)
                    st.caption("On-mound view — " + bio_caption(pit_h_ft, pit_hs, pit_w))
                with colR:
                    fig_s, ax_s = plt.subplots(figsize=(4.0, 4.0))
                    for nm, grp in p.groupby("pitch_name"):
                        ax_s.scatter(grp["release_pos_x"], grp["release_pos_z"], alpha=0.3, s=14, label=nm)
                    ax_s.set_xlabel("Horizontal release (ft, catcher view)")
                    ax_s.set_ylabel("Release height (ft)")
                    ax_s.set_title("Release spread (zoomed)", fontsize=10)
                    ax_s.set_aspect("equal")
                    ax_s.legend(fontsize=7)
                    st.pyplot(fig_s, use_container_width=False)
                    st.caption("Zoomed to the release cluster so you can see the spread between pitch types. "
                               "(See the Pitch guide tab for animated, illustrated pitches.)")

                summary = p.groupby("pitch_name").agg(
                    n=("release_pos_x", "size"),
                    rel_x_mean=("release_pos_x", "mean"), rel_z_mean=("release_pos_z", "mean"),
                    rel_x_std=("release_pos_x", "std"), rel_z_std=("release_pos_z", "std"),
                ).round(3)
                st.markdown("**Release-point summary by pitch type**")
                summary_disp = summary.rename(columns={
                    "n": "Pitches", "rel_x_mean": "Avg horiz. release (ft)", "rel_z_mean": "Avg release height (ft)",
                    "rel_x_std": "Horiz. consistency (SD, ft)", "rel_z_std": "Height consistency (SD, ft)",
                })
                summary_disp.index.name = "Pitch type"
                st.dataframe(summary_disp, use_container_width=True)
                st.caption(
                    "Pitches = number thrown. Avg horiz. release / release height = where the ball leaves the hand "
                    "(feet; horizontal is catcher's view, height is off the ground). The two consistency columns are "
                    "standard deviations of those release points — smaller = more repeatable."
                )

                centers = summary[["rel_x_mean", "rel_z_mean"]].values
                max_sep = max((np.hypot(*(centers[i] - centers[j]))
                               for i in range(len(centers)) for j in range(i + 1, len(centers))),
                              default=0.0)
                st.metric("Max release separation across pitch types", f"{max_sep*12:.1f} in",
                          help="Largest distance between the average release points of any two pitch types, in inches. Smaller = more deceptive.")

                names = list(summary.index)
                pairs = []
                for i in range(len(names)):
                    for j in range(i + 1, len(names)):
                        a, b = names[i], names[j]
                        rel = np.hypot(summary.loc[a, "rel_x_mean"] - summary.loc[b, "rel_x_mean"],
                                       summary.loc[a, "rel_z_mean"] - summary.loc[b, "rel_z_mean"])
                        pa, pb = p[p.pitch_name == a], p[p.pitch_name == b]
                        plate = np.hypot(pa["plate_x"].mean() - pb["plate_x"].mean(),
                                         pa["plate_z"].mean() - pb["plate_z"].mean())
                        pairs.append({"pair": f"{a} vs {b}",
                                      "release_sep_in": round(rel * 12, 2),
                                      "plate_sep_in": round(plate * 12, 2),
                                      "tunnel_ratio": round(plate / rel, 2) if rel > 0 else np.nan})
                tunnel = pd.DataFrame(pairs).sort_values("tunnel_ratio", ascending=False)
                st.markdown("**Approximate tunneling (release vs. plate separation)**")
                tunnel_disp = tunnel.rename(columns={
                    "pair": "Pitch pair", "release_sep_in": "Release gap (in)",
                    "plate_sep_in": "Plate gap (in)", "tunnel_ratio": "Tunnel ratio",
                })
                st.dataframe(tunnel_disp, hide_index=True, use_container_width=True)
                st.caption(
                    "Release gap = how far apart two pitches leave the hand. Plate gap = how far apart they end up. "
                    "Tunnel ratio = plate gap ÷ release gap; higher means they start together but separate late. "
                    "Approximate — true tunneling needs the ball's position at the commit point (~23 ft)."
                )

                st.markdown("#### What to work on")
                precs = []
                maxsep_in = max_sep * 12
                if maxsep_in > 4:
                    precs.append(f"**Tighten release consistency.** Release points differ by up to {maxsep_in:.1f} in across "
                                 "pitch types — enough for hitters to read the pitch early. Work toward a common release slot.")
                elif maxsep_in < 2:
                    precs.append(f"**Release is tightly clustered** (max {maxsep_in:.1f} in apart) — pitches come from essentially "
                                 "the same slot, which is hard to pick up. A real asset.")
                else:
                    precs.append(f"**Release is fairly consistent** (max {maxsep_in:.1f} in apart across pitch types).")
                spread = np.hypot(summary["rel_x_std"], summary["rel_z_std"])
                if len(spread) and pd.notna(spread.max()):
                    worst = spread.idxmax()
                    precs.append(f"**Steady up the {worst}.** Its release is the most scattered "
                                 f"(~{spread.loc[worst]*12:.1f} in spread); a more repeatable slot there improves command and deception.")
                if len(tunnel):
                    best = tunnel.iloc[0]
                    precs.append(f"**Lean on the {best['pair']} pairing.** It tunnels best — starts close, ends "
                                 f"{best['plate_sep_in']:.1f} in apart at the plate. A strong two-strike weapon.")
                for r in precs:
                    st.markdown("- " + r)
                st.caption("Heuristic guidance from release separation, per-pitch spread, and the tunneling proxy above. "
                           "A data-driven read, not professional coaching.")

with tab_f:
    st.subheader("Fielding — where the ball comes to them")
    glossary_expander()
    pitch_types_expander()
    st.caption(
        "Enter a fielder, their team, and a date range. This pulls the team's Statcast data and filters to balls "
        "that player fielded — a heavier pull, so give it time and rerun if it rate-limits. Team uses standard "
        "abbreviations (e.g., BAL, NYY, LAD, KC)."
    )
    with st.form("field_form"):
        c1, c2, c3 = st.columns(3)
        flast = c1.text_input("Fielder last name", key="f_last")
        ffirst = c2.text_input("Fielder first name", key="f_first")
        team = c3.text_input("Team abbreviation", key="f_team")
        c4, c5 = st.columns(2)
        f_start = c4.text_input("Start date (YYYY-MM-DD)", placeholder="YYYY-MM-DD", key="f_cs")
        f_end = c5.text_input("End date (YYYY-MM-DD)", placeholder="YYYY-MM-DD", key="f_ce")
        go_f = st.form_submit_button("Run fielding analysis")

    if go_f:
        persist_inputs()
        if not (f_start.strip() and f_end.strip()):
            st.warning("Enter the start and end dates (YYYY-MM-DD).")
            st.stop()
        pid, name = lookup_id(flast, ffirst)
        if pid is None:
            st.error(f"No player found for '{ffirst} {flast}'. Check the spelling.")
        else:
            try:
                with st.spinner(f"Pulling {team.upper()} Statcast data… (this one is slow)"):
                    df = pull_team(f_start, f_end, team.strip().upper())
            except Exception as e:
                st.error("Data pull failed. Often this is Baseball Savant rate-limiting — wait a minute and click Run again. If it keeps failing, expand the details below and share them.")
                with st.expander("Show technical details"):
                    st.exception(e)
                st.stop()

            if df is None or len(df) == 0:
                st.warning("No data returned. Double-check the team abbreviation (e.g., BAL, NYY, LAD) and dates.")
                st.stop()

            appear = {}
            for n in range(1, 10):
                col = f"fielder_{n}"
                if col in df.columns:
                    appear[n] = int((df[col] == pid).sum())
            if sum(appear.values()) == 0:
                st.warning(f"{name} doesn't appear as a fielder in {team.upper()}'s data for this window. Check the name and team.")
                st.stop()

            mask = pd.Series(False, index=df.index)
            if "hit_location" in df.columns:
                for n in range(1, 10):
                    col = f"fielder_{n}"
                    if col in df.columns:
                        mask = mask | ((df["hit_location"] == n) & (df[col] == pid))
            fb = df[mask].copy()

            st.markdown(f"### {name} — {team.upper()}")
            primary = max(appear, key=appear.get)
            mc1, mc2 = st.columns(2)
            mc1.metric("Primary position", POS.get(primary, "?"))
            mc2.metric("Balls fielded", len(fb), help=GLOSSARY["Fielded"])

            st.markdown("**Where they played** (pitches on the field by position)")
            posrows = [{"Position": POS.get(n, str(n)), "Pitches on field": appear[n]}
                       for n in sorted(appear) if appear[n] > 0]
            st.dataframe(pd.DataFrame(posrows), hide_index=True, use_container_width=True)

            if len(fb) >= 1:
                loc = fb.dropna(subset=["hc_x", "hc_y"]) if {"hc_x", "hc_y"}.issubset(fb.columns) else fb.iloc[0:0]
                if len(loc) >= 3:
                    X = (loc["hc_x"] - 125.42) * 2.5
                    Y = (198.27 - loc["hc_y"]) * 2.5
                    fig, ax = plt.subplots(figsize=(4.4, 4.4))
                    draw_field(ax)
                    if "bb_type" in loc.columns and loc["bb_type"].notna().any():
                        bt = loc["bb_type"].fillna("unknown")
                        for t in bt.unique():
                            sel = bt == t
                            ax.scatter(X[sel], Y[sel], s=20, alpha=0.6, label=t, zorder=4)
                        ax.legend(title="batted-ball type", fontsize=8)
                    else:
                        ax.scatter(X, Y, s=20, alpha=0.6, zorder=4)
                    xmax = max(180, float(np.nanmax(np.abs(X))) + 25)
                    ymax = max(250, float(np.nanmax(Y)) + 25)
                    ax.set_xlim(-xmax, xmax)
                    ax.set_ylim(-20, ymax)
                    ax.set_aspect("equal")
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_title(f"Where {name} fielded balls (to scale)", fontsize=10)
                    st.pyplot(fig, use_container_width=False)
                    st.caption("To-scale field in feet: home plate at bottom, diamond and ~400 ft wall for reference.")

                st.markdown("**Who hit to them most**")
                top = fb["batter"].value_counts().head(10)
                nm = names_for(tuple(int(i) for i in top.index))
                who = pd.DataFrame({"Batter": [nm.get(int(i), str(int(i))) for i in top.index],
                                    "Balls fielded": top.values})
                st.dataframe(who, hide_index=True, use_container_width=True)

                cc1, cc2 = st.columns(2)
                with cc1:
                    st.markdown("**Pitch type hit to them**")
                    if "pitch_name" in fb.columns:
                        pt = fb["pitch_name"].value_counts().rename_axis("Pitch").reset_index(name="Balls")
                        st.dataframe(pt, hide_index=True, use_container_width=True)
                with cc2:
                    st.markdown(f"**{term('Batted-ball type')}**", unsafe_allow_html=True)
                    if "bb_type" in fb.columns:
                        btv = fb["bb_type"].value_counts().rename_axis("Type").reset_index(name="Balls")
                        st.dataframe(btv, hide_index=True, use_container_width=True)

            st.markdown(f"**{term('Error')}s**", unsafe_allow_html=True)
            if "des" in df.columns:
                des = df["des"].fillna("")
                emask = des.str.contains("error", case=False)
                if "events" in df.columns:
                    emask = emask | (df["events"].fillna("") == "field_error")
                lastl = flast.strip().lower()
                err = df[emask & des.str.lower().str.contains(lastl)]
                ecols = [c for c in ["game_date", "des"] if c in err.columns]
                if len(err) and ecols:
                    ed = err[ecols].drop_duplicates().rename(columns={"game_date": "Date", "des": "Play"})
                    if "Date" in ed.columns:
                        ed = ed.sort_values("Date")
                    st.metric("Errors in window", len(ed))
                    st.dataframe(ed, hide_index=True, use_container_width=True)
                    st.caption("Plays whose description mentions an error involving this player. Heuristic — based on the play text.")
                else:
                    st.info("No errors found for this player in the window.")
            else:
                st.info("Error descriptions aren't available in this data.")

with tab_t:
    st.subheader("Team stats")
    glossary_expander()
    pitch_types_expander()
    st.caption("A team's pitching mix and hitting over a window. Heavy pull (whole team), so give it time and rerun if it rate-limits.")
    with st.form("team_form"):
        c1, c2, c3 = st.columns(3)
        t_team = c1.text_input("Team abbreviation", key="t_team")
        t_start = c2.text_input("Start date (YYYY-MM-DD)", placeholder="YYYY-MM-DD", key="t_cs")
        t_end = c3.text_input("End date (YYYY-MM-DD)", placeholder="YYYY-MM-DD", key="t_ce")
        comp_team_t = st.selectbox("Compare to team (optional)", [NONE_OPT] + TEAMS_LIST,
                                   index=0, key="comp_t")
        go_t = st.form_submit_button("Run team analysis")

    if go_t:
        persist_inputs()
        if not (t_team.strip() and t_start.strip() and t_end.strip()):
            st.warning("Enter a team abbreviation and the start and end dates (YYYY-MM-DD).")
        else:
            st.session_state["team_run"] = (t_team.strip().upper(), t_start, t_end)
    if st.session_state.get("team_run"):
        team_u, t_start, t_end = st.session_state["team_run"]

        season_yr = int(t_start[:4]) if t_start[:4].isdigit() else dt.date.today().year
        tb, tb_err = team_season_batting(season_yr)
        st.markdown(f"### {team_u} — batting ({season_yr} season totals)")
        row = find_team_row(tb, team_u)
        if row is not None:
            def gi(col):
                v = row.get(col)
                return f"{int(v)}" if pd.notna(v) else "—"

            def gf(col):
                v = row.get(col)
                return (f"{v:.3f}").lstrip("0") if pd.notna(v) else "—"

            h, d2, t3, hr = row.get("H"), row.get("2B"), row.get("3B"), row.get("HR")
            singles = f"{int(h - d2 - t3 - hr)}" if all(pd.notna(x) for x in (h, d2, t3, hr)) else gi("1B")
            ops = row.get("OPS")
            if pd.isna(ops):
                o, s = row.get("OBP"), row.get("SLG")
                ops = (o + s) if (pd.notna(o) and pd.notna(s)) else float("nan")

            r1 = st.columns(4)
            r1[0].metric("Home runs", gi("HR"))
            r1[1].metric("Triples (3B)", gi("3B"))
            r1[2].metric("Doubles (2B)", gi("2B"))
            r1[3].metric("Singles (1B)", singles)
            r2 = st.columns(4)
            r2[0].metric("Hits", gi("H"))
            r2[1].metric("Walks (BB)", gi("BB"))
            r2[2].metric("Strikeouts (SO)", gi("SO"))
            r2[3].metric("Stolen bases", gi("SB"))
            r3 = st.columns(4)
            r3[0].metric("AVG", gf("AVG"))
            r3[1].metric("OBP", gf("OBP"))
            r3[2].metric("SLG", gf("SLG"))
            r3[3].metric("OPS", (f"{ops:.3f}").lstrip("0") if pd.notna(ops) else "—")
            r4 = st.columns(4)
            r4[0].metric("Runs", gi("R"))
            r4[1].metric("RBI", gi("RBI"))
            r4[2].metric("ISO", gf("ISO"))
            r4[3].metric("BABIP", gf("BABIP"))
            st.caption(f"Full-season {season_yr} team batting totals from the MLB Stats API — the team's own offense.")
            _bat_lbl = "" if comp_team_t == NONE_OPT else f" & {comp_team_t}"
            st.markdown(f"**Batting rates vs. MLB average{_bat_lbl}**")
            st.dataframe(compare_table(team_u, row, tb, comp_team_t, None, None, BAT_SPECS),
                         hide_index=True, use_container_width=True)
        else:
            st.info(f"Couldn't load {team_u} batting totals for {season_yr} from the MLB Stats API.")
            with st.expander("Show details"):
                if tb_err:
                    st.write("FanGraphs error:", tb_err)
                elif tb is not None:
                    st.write("Columns returned:", list(tb.columns))
                    st.dataframe(tb.head(8), use_container_width=True)
                else:
                    st.write("No data returned.")

        st.markdown("---")
        try:
            with st.spinner(f"Pulling {team_u} Statcast data… (slow)"):
                df = pull_team(t_start, t_end, team_u)
        except Exception as e:
            st.error("Data pull failed. Often this is Baseball Savant rate-limiting — wait a minute and click Run again. If it keeps failing, expand the details below and share them.")
            with st.expander("Show technical details"):
                st.exception(e)
            st.stop()

        if df is None or len(df) == 0:
            st.warning("No data returned. Check the team abbreviation and dates.")
            st.stop()
        st.markdown(f"### {team_u} pitching — your staff vs. the pitching you faced")
        st.markdown(
            f"- **{team_u} pitchers** = how {team_u}'s own pitching staff did (facing opposing hitters).\n"
            f"- **Opp pitchers faced** = how the opposing pitchers did against {team_u}'s hitters — i.e., the pitching "
            f"{team_u}'s offense saw."
        )
        st.caption("Comparing the columns shows whether your pitching was tougher or easier than the pitching you faced. "
                   "Lower K% / BA against / xwOBA = better pitching.")
        if {"home_team", "away_team", "inning_topbot"}.issubset(df.columns):
            bal_mask = (((df["home_team"] == team_u) & (df["inning_topbot"] == "Top"))
                        | ((df["away_team"] == team_u) & (df["inning_topbot"] == "Bot")))
            involved = (df["home_team"] == team_u) | (df["away_team"] == team_u)
            bal_df = df[bal_mask]
            opp_df = df[involved & ~bal_mask]
        else:
            bal_df = df
            opp_df = df.iloc[0:0]

        bal_s = pitch_stats(bal_df)
        tid_t = TEAM_IDS.get(team_u)
        load_key = f"loadopp_{team_u}_{t_start}_{t_end}"
        if (opp_df is None or len(opp_df) == 0):
            if not st.session_state.get(load_key):
                st.caption(
                    f"The `team={team_u}` Statcast pull only contains {team_u}'s own pitches, so the **Opp pitchers "
                    "faced** column is empty by default. Load it below — it pulls every Baltimore hitter's faced "
                    "pitches and stitches them together (slower, and may rate-limit)."
                )
                if tid_t and st.button("⬇︎ Load opponents' pitching (slower)"):
                    st.session_state[load_key] = True
                    st.rerun()
            if st.session_state.get(load_key) and tid_t:
                with st.spinner("Pulling every Baltimore hitter's pitches faced… this takes a bit."):
                    loaded = opp_pitching_faced(tid_t, t_start, t_end, season_yr)
                if loaded is not None and len(loaded):
                    opp_df = loaded
                    st.caption("✓ Opponents' pitching loaded (stitched from Baltimore's hitters).")
                else:
                    st.session_state[load_key] = False
                    st.warning("Couldn't assemble opponents' pitching (roster or rate-limit issue). Try the button again in a minute.")

        opp_s = pitch_stats(opp_df)
        comp_rows = [{"Stat": s, f"{team_u} pitchers": bal_s.get(s, "—"),
                      "Opp pitchers faced": opp_s.get(s, "—")} for s in PITCH_STAT_ORDER]
        st.dataframe(pd.DataFrame(comp_rows), hide_index=True, use_container_width=True)

        mcol1, mcol2 = st.columns(2)
        with mcol1:
            st.markdown(f"**{team_u} pitch mix**")
            if "pitch_name" in bal_df.columns and len(bal_df):
                st.dataframe(mix_table(bal_df["pitch_name"].value_counts()), hide_index=True, use_container_width=True)
            if "pitcher" in bal_df.columns and len(bal_df):
                st.markdown(f"**{team_u} pitchers used**")
                tb1 = bal_df["pitcher"].value_counts().head(10)
                n1 = names_for(tuple(int(i) for i in tb1.index))
                st.dataframe(pd.DataFrame({"Pitcher": [n1.get(int(i), str(int(i))) for i in tb1.index],
                                           "Pitches": tb1.values}), hide_index=True, use_container_width=True)
        with mcol2:
            st.markdown("**Opponents' pitch mix**")
            if "pitch_name" in opp_df.columns and len(opp_df):
                st.dataframe(mix_table(opp_df["pitch_name"].value_counts()), hide_index=True, use_container_width=True)
                if "pitcher" in opp_df.columns:
                    st.markdown("**Opposing pitchers faced**")
                    tb2 = opp_df["pitcher"].value_counts().head(10)
                    n2 = names_for(tuple(int(i) for i in tb2.index))
                    st.dataframe(pd.DataFrame({"Pitcher": [n2.get(int(i), str(int(i))) for i in tb2.index],
                                               "Pitches": tb2.values}), hide_index=True, use_container_width=True)
            else:
                st.caption("Use the **Load opponents' pitching** button above to fill this in.")
        st.caption("'BA against' / 'xwOBA' describe how the batters facing that pitching performed. Baltimore's side is "
                   "from the team pull; the opponents' side (once loaded) is stitched from each Baltimore hitter's data.")

        tp_cmp, _tp_cmp_err = fg_team_pitching(season_yr)
        tp_row = find_team_row(tp_cmp, team_u)
        if tp_row is not None:
            _pit_lbl = "" if comp_team_t == NONE_OPT else f" & {comp_team_t}"
            st.markdown(f"**Pitching rates vs. MLB average{_pit_lbl}**")
            st.dataframe(compare_table(team_u, tp_row, tp_cmp, comp_team_t, None, None, PIT_SPECS),
                         hide_index=True, use_container_width=True)
            st.caption(f"Full-season {season_yr} team pitching rates vs. the MLB average (mean of all 30 teams)"
                       f"{('' if comp_team_t == NONE_OPT else ' and ' + comp_team_t)}, from the MLB Stats API.")

with tab_m:
    st.subheader("Next game — pitch probability & calibration")
    st.caption(
        "Projected pitch PROBABILITY for each probable starter (recency-weighted from this season), which you can "
        "save and later grade against what was actually thrown — building an accuracy record over time. A data-informed "
        "projection, not a guarantee: starters and game plans change. Saved history lives on your computer; on the "
        "hosted version it resets when the app restarts."
    )
    with st.form("matchup_form"):
        m_team = st.text_input("Team abbreviation", key="ng_team")
        go_m = st.form_submit_button("Project next game")

    if go_m:
        persist_inputs()
        abbr = m_team.strip().upper()
        tid = TEAM_IDS.get(abbr)
        if tid is None:
            st.error(f"Unknown team abbreviation '{abbr}'. Try BAL, NYY, LAD, KC, etc.")
        else:
            with st.spinner("Looking up the schedule…"):
                g = next_game(tid, dt.date.today().isoformat())
            if not g:
                st.warning("Couldn't find an upcoming game in the next three weeks (or the schedule API was unreachable).")
            else:
                st.session_state["mg"] = g
                st.markdown(f"### {g['away']} @ {g['home']} — {g['date']}")
                season = (g["date"][:4] if g.get("date") else str(dt.date.today().year))
                s_start, s_end = f"{season}-03-01", dt.date.today().isoformat()
                projections = {}
                for side in ("away", "home"):
                    ppname = g.get(f"{side}_pp")
                    ppid = g.get(f"{side}_pp_id")
                    st.markdown(f"#### {g[side]} probable starter")
                    if not ppid:
                        st.info("No probable starter posted yet (usually announced 1–2 days out).")
                        continue
                    st.markdown(f"**{ppname}**")
                    try:
                        with st.spinner(f"Projecting {ppname}…"):
                            prob = projected_mix(int(ppid), s_start, s_end)
                    except Exception:
                        prob = None
                    if prob is not None and len(prob):
                        tdf = prob.rename_axis("Pitch").reset_index(name="Probability %")
                        st.dataframe(tdf, hide_index=True, use_container_width=True)
                        projections[str(int(ppid))] = {"pitcher": ppname, "side": g[side], "projected": prob.to_dict()}
                    else:
                        st.info("No pitch data yet for this pitcher this season.")
                st.session_state["mg_proj"] = projections
                st.caption("Probability = recency-weighted share of each pitch this season (recent starts count more); sums to 100%.")

    if st.session_state.get("mg_proj"):
        if st.button("💾 Save these projections to my log"):
            g = st.session_state.get("mg", {})
            recs = load_log()
            existing = {(r["game_date"], r["pitcher_id"]) for r in recs}
            added = 0
            for pid_s, info in st.session_state["mg_proj"].items():
                if (g.get("date"), int(pid_s)) in existing:
                    continue
                recs.append({"game_date": g.get("date"), "pitcher_id": int(pid_s), "pitcher": info["pitcher"],
                             "team": info["side"], "projected": info["projected"], "graded": False,
                             "actual": None, "accuracy": None})
                added += 1
            if save_log(recs):
                st.success(f"Saved {added} projection(s).")
            else:
                st.warning("Couldn't write the log file here (read-only filesystem?).")

    st.markdown("---")
    st.markdown("#### Projection log & accuracy")
    log = load_log()
    if not log:
        st.info("No saved projections yet. Project a game and click Save to start building your log.")
    else:
        if st.button("✅ Grade finished games"):
            today = dt.date.today().isoformat()
            graded_now = 0
            for r in log:
                if r.get("graded") or (r.get("game_date") or "9999") >= today:
                    continue
                try:
                    actual = actual_game_mix(int(r["pitcher_id"]), r["game_date"])
                except Exception:
                    actual = None
                if not actual:
                    continue
                proj = r.get("projected") or {}
                keys = set(proj) | set(actual)
                tvd = 0.5 * sum(abs(proj.get(k, 0) - actual.get(k, 0)) for k in keys) / 100.0
                r["actual"] = actual
                r["accuracy"] = round((1 - tvd) * 100, 1)
                r["graded"] = True
                graded_now += 1
            save_log(log)
            st.success(f"Graded {graded_now} game(s).")
            log = load_log()

        rows = [{"Date": r.get("game_date"), "Pitcher": r.get("pitcher"),
                 "Status": "graded" if r.get("graded") else "pending",
                 "Accuracy %": r.get("accuracy") if r.get("accuracy") is not None else "—"} for r in log]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        gr = [r for r in log if r.get("graded") and r.get("accuracy") is not None]
        if gr:
            st.metric("Average projection accuracy", f"{np.mean([r['accuracy'] for r in gr]):.1f}%",
                      help="100% = projected pitch mix exactly matched what was actually thrown (via total-variation distance).")
            with st.expander("Projected vs. actual for graded games"):
                for r in gr:
                    st.markdown(f"**{r['pitcher']} — {r['game_date']}** (accuracy {r['accuracy']}%)")
                    comp = pd.DataFrame({"Projected %": pd.Series(r["projected"]),
                                         "Actual %": pd.Series(r["actual"])}).fillna(0).round(1)
                    st.dataframe(comp, use_container_width=True)
        st.caption("Accuracy = 100% − total-variation distance between projected and actual pitch mix. "
                   "Grade games the day after they're played, once Statcast has the data.")

with tab_g:
    st.subheader("Pitch guide — what each pitch looks like")
    st.caption(
        "Original animated illustrations (a learning aid, not live data): watch the ball leave the pitcher and travel "
        "to the plate. Pitches group into fastballs, breaking balls, and offspeed."
    )
    pg_view = st.radio("View", ["Side view", "Catcher's POV"], horizontal=True, key="pg_view")
    for i in range(0, len(PITCHES), 2):
        cols = st.columns(2)
        for col, p in zip(cols, PITCHES[i:i + 2]):
            with col:
                if pg_view == "Catcher's POV":
                    components.html(pitch_panel_catcher(p[0], p[1], p[5]), height=300)
                else:
                    components.html(pitch_panel(*p), height=252)
    st.caption("Speeds are typical big-league averages; individual pitchers vary. Break directions are drawn for a "
               "right-handed pitcher — a lefty's arm-side pitches mirror left-to-right. Side view = from the third-base "
               "side; Catcher's POV = the ball coming toward home plate.")

with tab_pred:
    st.subheader("Pitch predictor — live")
    pitch_types_expander()
    st.caption(
        "Load a pitcher once, then tap the count, outs, runners, and batter side as the at-bat unfolds — the "
        "likelihoods update instantly (no re-loading). Descriptive tendency from his history, not a guaranteed call."
    )

    with st.form("pred_load"):
        c1, c2 = st.columns(2)
        prlast = c1.text_input("Pitcher last name", key="pr_last")
        prfirst = c2.text_input("Pitcher first name", key="pr_first")
        c3, c4 = st.columns(2)
        pr_start = c3.text_input("From (YYYY-MM-DD)", placeholder="YYYY-MM-DD", key="pr_cs")
        pr_end = c4.text_input("To (YYYY-MM-DD)", placeholder="YYYY-MM-DD", key="pr_ce")
        if st.form_submit_button("Load pitcher"):
            persist_inputs()
            if not (pr_start.strip() and pr_end.strip()):
                st.warning("Enter the From and To dates (YYYY-MM-DD).")
            else:
                lid, lname = lookup_id(prlast, prfirst)
                if lid is None:
                    st.error(f"No pitcher found for '{prfirst} {prlast}'. Check the spelling.")
                else:
                    st.session_state["pred"] = {"pid": lid, "name": lname, "start": pr_start, "end": pr_end}

    pred = st.session_state.get("pred")
    if pred:
        try:
            with st.spinner(f"Loading {pred['name']}…"):
                pdf = pull_pitcher(pred["start"], pred["end"], pred["pid"])
        except Exception as e:
            st.error("Data pull failed — Baseball Savant may be rate-limiting. Wait a minute and click Load again.")
            with st.expander("Show technical details"):
                st.exception(e)
            st.stop()
        if pdf is None or len(pdf) == 0 or "pitch_name" not in pdf.columns:
            st.warning("No pitch data for this pitcher and date range.")
            st.stop()

        st.markdown(f"### {pred['name']}  ·  live situation")

        # ---- who's at the plate: scope the list to today's game so you're not scrolling the whole league ----
        today_iso = dt.date.today().isoformat()
        season = dt.date.today().year
        ptid = pitcher_team(pred["pid"])
        auto_pk, auto_opp, auto_home = today_opponent(ptid, today_iso) if ptid else (None, None, None)

        opp_choices = ["Auto — today's opponent"] + sorted(ID2ABBR.values())
        opp_pick = st.selectbox(
            "Opponent / lineup", opp_choices, index=0, key="lopp",
            help="Defaults to whoever your pitcher's team plays today. Override if you're watching a different game.")
        if opp_pick == "Auto — today's opponent":
            opp_id_final, game_pk, opp_home = auto_opp, auto_pk, auto_home
        else:
            opp_id_final = {v: k for k, v in ID2ABBR.items()}.get(opp_pick)
            game_pk, opp_home = None, None

        my_abbr = ID2ABBR.get(ptid, "his team")
        opp_abbr = ID2ABBR.get(opp_id_final, "opponent")

        batter_list, src, lineup_posted = [], "", False
        if game_pk is not None:
            batter_list = game_lineup(game_pk, opp_home)
            if batter_list:
                src, lineup_posted = "today's posted lineup", True
        if not batter_list and opp_id_final:
            batter_list = team_hitters(opp_id_final, season)
            src = f"{opp_abbr} roster"

        faced_counts = (pdf["batter"].dropna().astype(int).value_counts().to_dict()
                        if "batter" in pdf.columns else {})

        if opp_pick == "Auto — today's opponent":
            if opp_id_final:
                st.caption(f"Detected today's game: **{my_abbr} vs {opp_abbr}**.")
            else:
                st.caption(f"Couldn't auto-detect a game today for **{my_abbr}** (off-day, or his team didn't "
                           "resolve from the API). Pick the opponent above to load their batters.")

        if batter_list:
            blabel_to_id, bopts = {}, ["Any (all batters)"]
            for bid, nm in batter_list:
                tag = f"{faced_counts[bid]} seen" if bid in faced_counts else "new — uses handedness"
                lab = f"{nm}  ·  {tag}"
                bopts.append(lab)
                blabel_to_id[lab] = bid
            batter_sel = st.selectbox(
                "Facing batter (optional)", bopts, index=0, key="lbat",
                help="Scoped to the current game. 'seen' = pitches he's already thrown this batter (head-to-head); "
                     "'new' = no history, so it falls back to the batter's handedness.")
            if lineup_posted:
                st.caption(f"Batter list from **{src}** — the 9 in order. Pick who's at the plate.")
            else:
                st.caption(f"⚠︎ Lineup not posted yet — showing the **{src}** ({len(batter_list)} hitters). "
                           "Lineups usually drop ~2–4 hrs before first pitch; re-open then for the exact 9.")
        else:
            blabel_to_id, batter_sel = {}, "Any (all batters)"
            if opp_id_final:
                st.info(f"Found **{my_abbr} vs {opp_abbr}**, but couldn't load batters (no posted lineup and the roster "
                        "fetch came back empty). Type the batter below, or use the situation filters alone.")
            else:
                st.info("Pick an opponent above to load their batters, type a batter below, or just use the situation filters.")

        with st.expander("Batter not listed? (pinch-hitter, call-up) — type a name"):
            nb1, nb2 = st.columns(2)
            new_last = nb1.text_input("Last name", "", key="lbat_new_l")
            new_first = nb2.text_input("First name", "", key="lbat_new_f")

        st.write("Quick set:")
        qp = st.columns(6)
        if qp[0].button("First pitch"):
            st.session_state.update({"lb": "0", "ls": "0"}); st.rerun()
        if qp[1].button("2 strikes"):
            st.session_state.update({"ls": "2"}); st.rerun()
        if qp[2].button("Full count"):
            st.session_state.update({"lb": "3", "ls": "2"}); st.rerun()
        if qp[3].button("RISP"):
            st.session_state.update({"lr": "Runner in scoring position"}); st.rerun()
        if qp[4].button("Bases loaded"):
            st.session_state.update({"lr": "Bases loaded"}); st.rerun()
        if qp[5].button("Reset"):
            st.session_state.update({"lb": "Any", "ls": "Any", "lo": "Any", "lh": "Any", "lr": "Any", "li": "Any", "lv": "Any", "lbat": "Any (all batters)", "lbat_new_l": "", "lbat_new_f": ""}); st.rerun()
        st.caption(
            "Quick set — **First pitch**: 0-0 count. **2 strikes**: any count with two strikes (put-away spot). "
            "**Full count**: 3 balls and 2 strikes (3-2). **RISP**: runner in scoring position (2nd and/or 3rd). "
            "**Bases loaded**: runners on 1st, 2nd, and 3rd. **Reset**: clear all filters back to Any."
        )

        r1, r2, r3 = st.columns(3)
        balls_sel = r1.radio("Balls", ["Any", "0", "1", "2", "3"], horizontal=True, key="lb")
        strikes_sel = r2.radio("Strikes", ["Any", "0", "1", "2"], horizontal=True, key="ls")
        outs_sel = r3.radio("Outs", ["Any", "0", "1", "2"], horizontal=True, key="lo")
        r4, r5 = st.columns([1, 2])
        hand_sel = r4.radio("Batter hits", ["Any", "R", "L"], horizontal=True, key="lh")
        base_sel = r5.radio("Runners", ["Any", "Bases empty", "Runner(s) on", "Runner in scoring position", "Bases loaded"],
                            horizontal=True, key="lr")
        r6, r7 = st.columns(2)
        inning_sel = r6.selectbox("Inning", ["Any", "1", "2", "3", "4", "5", "6", "7", "8", "9", "Extra (10+)"],
                                  index=0, key="li")
        venue_sel = r7.selectbox("Pitching at", ["Any", "Home", "Away"], index=0, key="lv")
        st.caption(
            "**Runners** — **Any**: every situation. **Bases empty**: no runners on. **Runner(s) on**: at least one "
            "runner anywhere. **Runner in scoring position (RISP)**: a runner on 2nd and/or 3rd — close enough to score "
            "on a single. **Bases loaded**: runners on 1st, 2nd, and 3rd. "
            "**Pitching at** — **Home** = his home games, **Away** = his road games."
        )

        overall = pdf["pitch_name"].value_counts(normalize=True) * 100
        d = pdf.copy()
        if balls_sel != "Any" and "balls" in d.columns:
            d = d[d["balls"] == int(balls_sel)]
        if strikes_sel != "Any" and "strikes" in d.columns:
            d = d[d["strikes"] == int(strikes_sel)]
        if outs_sel != "Any" and "outs_when_up" in d.columns:
            d = d[d["outs_when_up"] == int(outs_sel)]
        if inning_sel != "Any" and "inning" in d.columns:
            inn = pd.to_numeric(d["inning"], errors="coerce")
            d = d[inn >= 10] if inning_sel == "Extra (10+)" else d[inn == int(inning_sel)]
        if venue_sel != "Any" and "inning_topbot" in d.columns:
            d = d[d["inning_topbot"] == ("Top" if venue_sel == "Home" else "Bot")]
        if hand_sel != "Any" and "stand" in d.columns:
            d = d[d["stand"] == hand_sel]
        for col in ("on_1b", "on_2b", "on_3b"):
            if col not in d.columns:
                d[col] = np.nan
        on1, on2, on3 = d["on_1b"].notna(), d["on_2b"].notna(), d["on_3b"].notna()
        if base_sel == "Bases empty":
            d = d[~on1 & ~on2 & ~on3]
        elif base_sel == "Runner(s) on":
            d = d[on1 | on2 | on3]
        elif base_sel == "Runner in scoring position":
            d = d[on2 | on3]
        elif base_sel == "Bases loaded":
            d = d[on1 & on2 & on3]
        def _hand_of(bid):
            bh = batter_hand(bid)
            if bh == "S":
                ph = pdf["p_throws"].dropna().mode() if "p_throws" in pdf.columns else pd.Series(dtype=object)
                bh = "L" if (len(ph) and ph.iloc[0] == "R") else "R"
            return bh

        nb_note = None
        sel_bid = blabel_to_id.get(batter_sel)
        if new_last.strip():
            nbid, nbname = lookup_id(new_last, new_first)
            if nbid is None:
                nb_note = ("warning", f"No player found for '{new_first} {new_last}'. Last name alone usually works.")
            elif nbid in faced_counts and "batter" in d.columns:
                d = d[d["batter"] == nbid]
                nb_note = ("info", f"{pred['name']} has faced {nbname} — using their head-to-head history.")
            else:
                bh = _hand_of(nbid)
                if bh in ("R", "L") and "stand" in d.columns:
                    d = d[d["stand"] == bh]
                    nb_note = ("info", f"No matchups vs {nbname} — showing {pred['name']}'s mix vs **{bh}-handed batters**.")
                else:
                    nb_note = ("warning", f"Couldn't determine {nbname}'s handedness; showing all batters.")
        elif sel_bid is not None and "batter" in d.columns:
            bname = batter_sel.split("  ·  ")[0]
            if sel_bid in faced_counts:
                d = d[d["batter"] == sel_bid]
            else:
                bh = _hand_of(sel_bid)
                if bh in ("R", "L") and "stand" in d.columns:
                    d = d[d["stand"] == bh]
                    nb_note = ("info", f"No matchups vs {bname} — showing {pred['name']}'s mix vs **{bh}-handed batters**.")

        n = len(d)
        if nb_note:
            (st.info if nb_note[0] == "info" else st.warning)(nb_note[1])
        cstr = f"{balls_sel if balls_sel != 'Any' else 'x'}-{strikes_sel if strikes_sel != 'Any' else 'x'}"
        if n == 0:
            st.warning("No pitches match this exact situation. Loosen a filter.")
        else:
            prob = (d["pitch_name"].value_counts(normalize=True) * 100).round(1)
            top = prob.index[0]
            m1, m2 = st.columns(2)
            m1.metric(f"Most likely ({cstr})", f"{top}", f"{prob.iloc[0]:.0f}%")
            m2.metric("Matching pitches", n)
            if n < 10:
                st.warning(f"Only {n} pitches match — too few to trust. Loosen a filter for a steadier read.")
            tbl = prob.rename_axis("Pitch").reset_index(name="Likelihood %")
            tbl["Overall %"] = tbl["Pitch"].map(lambda p: round(float(overall.get(p, 0)), 1))
            colA, colB = st.columns([2, 3])
            with colA:
                st.dataframe(tbl, hide_index=True, use_container_width=True)
            with colB:
                fig, ax = plt.subplots(figsize=(4.6, 2.8))
                ax.barh(list(prob.index[::-1]), list(prob.values[::-1]), color="#1f77b4")
                ax.set_xlabel("likelihood %")
                ax.set_title(f"In {cstr} counts" + ("" if base_sel == "Any" else f", {base_sel.lower()}"), fontsize=9)
                st.pyplot(fig, use_container_width=False)
            st.caption(f"From {n} pitches in this situation. 'Overall %' is his all-situations usage. Tendency, not a guarantee.")

            st.markdown("#### Log the actual pitch")
            st.caption("After the pitch is thrown, record what it actually was and how hard — it saves so you can track "
                       "how often your read was right.")
            base_opts = ["Any", "Bases empty", "Runner(s) on", "Runner in scoring position", "Bases loaded"]
            with st.form("logpitch", clear_on_submit=True):
                lc1, lc2, lc3, lc4 = st.columns([1.8, 1, 1.8, 0.9])
                arsenal = list(dict.fromkeys(list(prob.index) + ["Other"]))
                actual_pitch = lc1.selectbox("Actual pitch thrown", arsenal)
                actual_velo = lc2.number_input("Velocity (mph)", min_value=40.0, max_value=110.0, value=92.0, step=0.5)
                log_base = lc3.selectbox("Runners (this pitch)", base_opts,
                                         index=base_opts.index(base_sel) if base_sel in base_opts else 0)
                lc4.markdown("&nbsp;")
                log_btn = lc4.form_submit_button("✅ Log")
            if log_btn:
                sit_parts = []
                if balls_sel != "Any" or strikes_sel != "Any":
                    sit_parts.append(f"{cstr} count")
                if outs_sel != "Any":
                    sit_parts.append(f"{outs_sel} out")
                if log_base != "Any":
                    sit_parts.append(log_base.lower())
                if inning_sel != "Any":
                    sit_parts.append(f"inning {inning_sel}")
                if hand_sel != "Any":
                    sit_parts.append(f"vs {hand_sel}HB")
                if venue_sel != "Any":
                    sit_parts.append(venue_sel.lower())
                if new_last.strip():
                    sit_parts.append("vs " + (new_first + " " + new_last).strip())
                elif batter_sel in blabel_to_id:
                    sit_parts.append("vs " + batter_sel.split("  ·  ")[0])
                sit_txt = ", ".join(sit_parts) if sit_parts else "any situation"
                rec = {"logged_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
                       "pitcher": pred["name"], "situation": sit_txt,
                       "predicted": top, "pred_prob": float(prob.iloc[0]),
                       "actual": actual_pitch, "velo": float(actual_velo),
                       "correct": bool(actual_pitch == top)}
                plog = load_pitch_log()
                plog.append(rec)
                if save_pitch_log(plog):
                    st.success(f"Logged: {actual_pitch} at {actual_velo:.0f} mph — "
                               + ("✅ matched your prediction!" if rec["correct"] else f"you predicted {top}."))
                else:
                    st.warning("Couldn't save the log (add a 'pitch_log' tab to your Google Sheet — see note below).")

            plog = load_pitch_log()
            if plog:
                pdf_log = pd.DataFrame(plog)
                graded = pdf_log[pdf_log["predicted"].notna()] if "predicted" in pdf_log.columns else pdf_log
                if len(graded):
                    hit = graded["correct"].apply(lambda x: str(x).lower() in ("true", "1", "yes", "true")).mean() \
                        if "correct" in graded.columns else float("nan")
                    lm1, lm2 = st.columns(2)
                    lm1.metric("Pitches logged", len(pdf_log))
                    if pd.notna(hit):
                        lm2.metric("Prediction hit rate", f"{hit * 100:.0f}%",
                                   help="How often the most-likely pitch matched what was actually thrown.")
                show_cols = [c for c in ["logged_at", "pitcher", "situation", "predicted", "actual", "velo", "correct"]
                             if c in pdf_log.columns]
                st.markdown("**Your logged pitches (most recent first)**")
                st.dataframe(pdf_log[show_cols].iloc[::-1].head(25), hide_index=True, use_container_width=True)
                st.caption("Saved to your Google Sheet's 'pitch_log' tab (add that tab once) so it persists on your phone.")

                if "actual" in pdf_log.columns and "velo" in pdf_log.columns:
                    vv = pdf_log.copy()
                    vv["velo"] = pd.to_numeric(vv["velo"], errors="coerce")
                    av = vv.dropna(subset=["velo"]).groupby("actual")["velo"].agg(["count", "mean", "max"])
                    if len(av):
                        av = (av.round(1).rename(columns={"count": "Pitches", "mean": "Avg mph", "max": "Top mph"})
                              .rename_axis("Pitch").reset_index().sort_values("Pitches", ascending=False))
                        st.markdown("**Avg velocity by pitch (from your logs)**")
                        st.dataframe(av, hide_index=True, use_container_width=True)

                with st.expander("✏️ Edit / delete logged pitches"):
                    labels = [f"{i}: {r.get('logged_at', '')} — {r.get('pitcher', '')} — "
                              f"{r.get('actual', '')} {r.get('velo', '')} mph" for i, r in enumerate(plog)]
                    to_del = st.multiselect("Select entries to delete", labels, key="pitch_del")
                    dc1, dc2 = st.columns(2)
                    if dc1.button("🗑 Delete selected") and to_del:
                        idxs = {int(lbl.split(":")[0]) for lbl in to_del}
                        save_pitch_log([r for i, r in enumerate(plog) if i not in idxs])
                        st.success(f"Deleted {len(idxs)} entr{'y' if len(idxs) == 1 else 'ies'}.")
                        st.rerun()
                    if dc2.button("Clear ALL logged pitches"):
                        save_pitch_log([])
                        st.success("Cleared all logged pitches.")
                        st.rerun()
    else:
        st.info("Enter a pitcher above and click **Load pitcher** to start.")

with tab_game:
    st.subheader("Game overview — pitch-by-pitch")
    st.caption(
        "Pick a team and a game date. Shows that game's at-bats — what the team's pitchers threw to each batter, "
        "and what its hitters saw — so you can spot pitcher-vs-batter patterns. It pulls the whole day's pitches, so give it a few seconds."
    )
    with st.form("game_form"):
        gc1, gc2 = st.columns(2)
        g_team = gc1.text_input("Team abbreviation", placeholder="e.g. BAL", key="g_team")
        g_date = gc2.text_input("Game date (YYYY-MM-DD)", placeholder="YYYY-MM-DD", key="g_date")
        go_game = st.form_submit_button("Load game")

    if go_game:
        persist_inputs()
        if not (g_team.strip() and g_date.strip()):
            st.warning("Enter a team abbreviation and a game date (YYYY-MM-DD).")
            st.stop()
        team_g = g_team.strip().upper()
        try:
            with st.spinner("Pulling that day's pitches…"):
                day = pull_day(g_date.strip())
        except Exception as e:
            st.error("Data pull failed — Baseball Savant may be rate-limiting. Wait a minute and try again.")
            with st.expander("Show technical details"):
                st.exception(e)
            st.stop()
        need = {"home_team", "away_team", "inning_topbot", "at_bat_number", "pitch_number",
                "pitcher", "batter", "inning"}
        if day is None or len(day) == 0 or not need.issubset(day.columns):
            st.warning("No usable data returned for that date.")
            st.stop()
        g = day[(day["home_team"] == team_g) | (day["away_team"] == team_g)].copy()
        if len(g) == 0:
            st.warning(f"No {team_g} game found on {g_date.strip()}. Check the date and the abbreviation.")
            st.stop()

        home, away = g["home_team"].iloc[0], g["away_team"].iloc[0]
        opp = away if home == team_g else home
        st.markdown(f"### {away} @ {home} — {g_date.strip()}")
        st.caption("Pitch codes: FF=4-seam, SI=sinker, FC=cutter, SL=slider, ST=sweeper, CU=curve, "
                   "KC=knuckle-curve, CH=change, FS=splitter, KN=knuckleball.")

        bal_pitch = (((g["home_team"] == team_g) & (g["inning_topbot"] == "Top"))
                     | ((g["away_team"] == team_g) & (g["inning_topbot"] == "Bot")))
        pit_rows, bat_rows = g[bal_pitch], g[~bal_pitch]

        ids = set()
        for col in ("pitcher", "batter"):
            ids |= {int(x) for x in g[col].dropna().astype(int).unique()}
        nm = names_for(tuple(ids))

        def pa_table(sub):
            if sub is None or len(sub) == 0:
                return pd.DataFrame()
            sub = sub.dropna(subset=["at_bat_number"]).copy()
            keys = ["game_pk", "at_bat_number"] if "game_pk" in sub.columns else ["at_bat_number"]
            rows = []
            for _, grp in sub.sort_values(keys + ["pitch_number"]).groupby(keys):
                grp = grp.sort_values("pitch_number")
                codes = grp["pitch_type"] if "pitch_type" in grp.columns else grp.get("pitch_name")
                velos = grp["release_speed"] if "release_speed" in grp.columns else [None] * len(grp)
                seq = ", ".join(
                    f"{(str(c) if pd.notna(c) else '?')}{(' ' + str(int(v))) if pd.notna(v) else ''}"
                    for c, v in zip(list(codes), list(velos))
                )
                last = grp.iloc[-1]
                res = last.get("events")
                if pd.isna(res) or not res:
                    res = last.get("description")
                res = str(res).replace("_", " ") if pd.notna(res) else ""
                rows.append({
                    "Inn": f"{str(last['inning_topbot'])[0]}{int(last['inning'])}",
                    "Pitcher": nm.get(int(last["pitcher"]), str(int(last["pitcher"]))),
                    "Batter": nm.get(int(last["batter"]), str(int(last["batter"]))),
                    "# pitches": len(grp),
                    "Pitches (type + mph)": seq,
                    "Result": res,
                })
            return pd.DataFrame(rows)

        st.markdown(f"#### {team_g} pitching — what each pitcher threw to each batter")
        pt_tbl = pa_table(pit_rows)
        if len(pt_tbl):
            st.dataframe(pt_tbl.sort_values(["Pitcher", "Inn"]), hide_index=True, use_container_width=True)
        else:
            st.info("No pitching at-bats found for this team in that game.")

        st.markdown(f"#### {team_g} hitting — what their batters saw (vs {opp})")
        bt_tbl = pa_table(bat_rows)
        if len(bt_tbl):
            st.dataframe(bt_tbl.sort_values(["Batter", "Inn"]), hide_index=True, use_container_width=True)
        else:
            st.info("No hitting at-bats found for this team in that game.")
        st.caption("Each row is one plate appearance: the pitch sequence (type + velocity, in order) and how it ended. "
                   "The pitching table is sorted by pitcher so his pattern against each batter is easy to scan.")

with tab_mu:
    st.subheader("Matchup — pitcher vs. batter")
    pitch_types_expander()
    st.caption(
        "Every pitch a pitcher has thrown to one specific batter over a window — his pitch mix, the results, and the "
        "full sequence of each plate appearance. The most direct way to see how he attacks that hitter."
    )
    with st.form("mu_form"):
        m1, m2 = st.columns(2)
        mu_plast = m1.text_input("Pitcher last name", key="mu_pl")
        mu_pfirst = m2.text_input("Pitcher first name", key="mu_pf")
        m3, m4 = st.columns(2)
        mu_blast = m3.text_input("Batter last name", key="mu_bl")
        mu_bfirst = m4.text_input("Batter first name", key="mu_bf")
        m5, m6 = st.columns(2)
        mu_start = m5.text_input("From (YYYY-MM-DD)", placeholder="YYYY-MM-DD", key="mu_cs")
        mu_end = m6.text_input("To (YYYY-MM-DD)", placeholder="YYYY-MM-DD", key="mu_ce")
        go_mu = st.form_submit_button("Run matchup")

    if go_mu:
        persist_inputs()
        if not (mu_start.strip() and mu_end.strip()):
            st.warning("Enter the From and To dates (YYYY-MM-DD).")
            st.stop()
        ppid, pname = lookup_id(mu_plast, mu_pfirst)
        bbid, bname = lookup_id(mu_blast, mu_bfirst)
        if ppid is None:
            st.error(f"No pitcher found for '{mu_pfirst} {mu_plast}'. Check the spelling.")
            st.stop()
        if bbid is None:
            st.error(f"No batter found for '{mu_bfirst} {mu_blast}'. Check the spelling.")
            st.stop()
        try:
            with st.spinner(f"Pulling {pname}'s pitches…"):
                pdf = pull_pitcher(mu_start, mu_end, ppid)
        except Exception as e:
            st.error("Data pull failed — Baseball Savant may be rate-limiting. Wait a minute and run again.")
            with st.expander("Show technical details"):
                st.exception(e)
            st.stop()
        if pdf is None or len(pdf) == 0 or "batter" not in pdf.columns:
            st.warning("No pitch data for this pitcher and date range.")
            st.stop()

        mu = pdf[pdf["batter"] == bbid].copy()
        st.markdown(f"### {pname} vs. {bname}")
        if len(mu) == 0:
            st.warning(f"{pname} didn't throw a pitch to {bname} in this window. Try a wider date range.")
            st.stop()

        pa = mu.dropna(subset=["events"]) if "events" in mu.columns else mu.iloc[0:0]
        mc = st.columns(3)
        mc[0].metric("Pitches", len(mu))
        mc[1].metric("Plate appearances", len(pa))
        if "release_speed" in mu.columns and mu["release_speed"].notna().any():
            mc[2].metric("Avg velo", f"{mu['release_speed'].dropna().mean():.1f} mph")

        if "pitch_name" in mu.columns:
            mix = (mu["pitch_name"].value_counts(normalize=True) * 100).round(1)
            overall = (pdf["pitch_name"].value_counts(normalize=True) * 100).round(1)
            t = mix.rename_axis("Pitch").reset_index(name="vs this batter %")
            t["Overall %"] = t["Pitch"].map(lambda p: round(float(overall.get(p, 0)), 1))
            st.markdown("**Pitch mix to this batter** (vs. his overall usage)")
            st.dataframe(t, hide_index=True, use_container_width=True)

            try:
                with st.spinner(f"Loading how {bname} hits each pitch…"):
                    bdf = pull_batter(mu_start, mu_end, bbid)
            except Exception:
                bdf = None
            if bdf is not None and len(bdf) and "pitch_name" in bdf.columns:
                bb = bdf.dropna(subset=["pitch_name"]).copy()
                bb["sw"] = bb["description"].isin(SWING_DESCS)
                bb["wh"] = bb["description"].isin({"swinging_strike", "swinging_strike_blocked"})
                prof = []
                for nm_, grp in bb.groupby("pitch_name"):
                    seen = len(grp)
                    sw = int(grp["sw"].sum())
                    xw = grp[XWOBA].dropna() if XWOBA in grp.columns else pd.Series(dtype=float)
                    prof.append({
                        "Pitch": nm_,
                        "Seen": seen,
                        "Whiff %": f"{int(grp['wh'].sum()) / sw * 100:.0f}%" if sw else "—",
                        "xwOBA/contact": (f"{xw.mean():.3f}").lstrip("0") if len(xw) else "—",
                        "HR": int((grp["events"] == "home_run").sum()) if "events" in grp.columns else 0,
                    })
                profdf = pd.DataFrame(prof).sort_values("Seen", ascending=False)
                st.markdown(f"**How {bname} does by pitch type** — vs. all pitchers (the 'why')")
                st.dataframe(profdf, hide_index=True, use_container_width=True)
                st.caption("Low whiff % + high xwOBA = pitches he handles, so pitchers avoid them. "
                           "High whiff % + low xwOBA = pitches that beat him, so pitchers lean on them. "
                           "Compare this to the mix above to read the reasoning.")

        oc = outcome_counts(mu)
        if oc:
            _n, counts = oc
            ot = pd.DataFrame({"Outcome": OUTCOME_ORDER, "Count": counts.values})
            ot = ot[ot["Count"] > 0]
            if len(ot):
                st.markdown("**Outcomes of those plate appearances**")
                st.dataframe(ot, hide_index=True, use_container_width=True)

        if {"at_bat_number", "pitch_number"}.issubset(mu.columns):
            st.markdown("**Every plate appearance (pitch sequence)**")
            keys = ["game_pk", "at_bat_number"] if "game_pk" in mu.columns else ["at_bat_number"]
            rows = []
            for _, grp in mu.sort_values(keys + ["pitch_number"]).groupby(keys):
                grp = grp.sort_values("pitch_number")
                codes = grp["pitch_type"] if "pitch_type" in grp.columns else grp.get("pitch_name")
                velos = grp["release_speed"] if "release_speed" in grp.columns else [None] * len(grp)
                seq = ", ".join(
                    f"{(str(c) if pd.notna(c) else '?')}{(' ' + str(int(v))) if pd.notna(v) else ''}"
                    for c, v in zip(list(codes), list(velos))
                )
                last = grp.iloc[-1]
                res = last.get("events")
                if pd.isna(res) or not res:
                    res = last.get("description")
                date = str(last.get("game_date"))[:10] if "game_date" in grp.columns else ""
                rows.append({"Date": date, "Pitches (type + mph)": seq,
                             "Result": str(res).replace("_", " ") if pd.notna(res) else ""})
            st.dataframe(pd.DataFrame(rows).iloc[::-1], hide_index=True, use_container_width=True)
            st.caption("Pitch codes: FF=4-seam, SI=sinker, FC=cutter, SL=slider, ST=sweeper, CU=curve, "
                       "KC=knuckle-curve, CH=change, FS=splitter.")
        st.caption(f"From {len(mu)} pitches {pname} threw to {bname} in the window. Head-to-head samples are small, so read trends, not certainties.")

with tab_about:
    st.markdown(
        """
**What this does**

*Hitter diagnosis* — chase rate vs. a baseline, pitches seen, outcome odds with expected-stat context, a swing map
with the batter to scale, a spray chart on a to-scale field, rolling xwOBA, and guidance.

*Pitcher deception* — release-point consistency vs. a to-scale pitcher, a movement profile, an approximate tunneling
proxy, and guidance.

*Fielding* — where a fielder fielded balls on a to-scale field, who hit to them, pitch and batted-ball types, and errors.

*Team stats* — pitching mix, balls/strikes, hits allowed, pitchers used, plus hitting outcomes and most-used hitters.

*Next game* — the team's next opponent, probable starters, recency-weighted pitch probabilities, and a saved
calibration log that grades projections against actual results over time.

*Pitch guide* — original animated illustrations of how each pitch moves.

Player bio and schedule data come from the MLB Stats API; pitch data comes live from Baseball Savant via `pybaseball`.
        """
    )
    glossary_expander()
    pitch_types_expander()
