#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════╗
║         ADVANCED GESTURE CONTROLLER  v2.0               ║
╠══════════════════════════════════════════════════════════╣
║  Features:                                               ║
║  1. Virtual Mouse  – Index finger moves cursor           ║
║                      Pinch thumb+index  → Click          ║
║                      Pinch thumb+middle → Right-Click    ║
║  2. Volume Control – Spread/close thumb & index          ║
║  3. Brightness     – Same pinch on left hand             ║
║  4. Scroll         – Move index finger up / down         ║
║  5. Presentation   – Open-palm swipe left / right        ║
║  6. Gesture Train  – Record & save custom gestures       ║
║  7. Multi-Hand     – Both hands tracked simultaneously   ║
╠══════════════════════════════════════════════════════════╣
║  Keys:  M / Tab = next mode                             ║
║         Q       = quit                                   ║
║  TRAIN: R=start recording  Enter=confirm name           ║
║         S=save  Esc=cancel  D=delete last               ║
╚══════════════════════════════════════════════════════════╝
"""

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
import math
import json
import os
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import screen_brightness_control as sbc

# ──────────────────────────── CONFIG ─────────────────────────────────────────
pyautogui.FAILSAFE = False
pyautogui.PAUSE    = 0

SCREEN_W, SCREEN_H = pyautogui.size()
CAM_W, CAM_H       = 1280, 720
GESTURE_FILE        = "custom_gestures.json"

SMOOTH_FACTOR = 7        # Mouse smoothing (higher = smoother / slower)
SWIPE_DIST    = 120      # Min horizontal px for presentation swipe
SWIPE_DUR     = 0.65     # Max seconds allowed to complete swipe

# ──────────────────────────── COLOR PALETTE (BGR) ────────────────────────────
P = {
    "bg":       (15,  15,  22),
    "panel":    (22,  26,  35),
    "mouse":    (40, 210, 255),   # amber-gold
    "volume":   (60, 210,  90),   # green
    "bright":   (30, 180, 255),   # sky blue
    "scroll":   (190, 80, 255),   # violet
    "present":  (30, 150, 255),   # blue
    "train":    (0,  165, 255),   # orange
    "accent":   (110,255, 180),   # mint
    "white":    (235,238, 242),
    "gray":     (95,  100,112),
    "darkgray": (40,  43,  52),
    "red":      (50,  50, 220),
    "green":    (60, 200,  80),
}

MODES = ["MOUSE", "VOLUME", "BRIGHTNESS", "SCROLL", "PRESENTATION", "TRAIN"]
MODE_CLR = {
    "MOUSE":        P["mouse"],
    "VOLUME":       P["volume"],
    "BRIGHTNESS":   P["bright"],
    "SCROLL":       P["scroll"],
    "PRESENTATION": P["present"],
    "TRAIN":        P["train"],
}

# ──────────────────────────── AUDIO INIT ─────────────────────────────────────
try:
    _spk = AudioUtilities.GetSpeakers()
    # pycaw >= 20230104 wraps IMMDevice inside AudioDevice._dev
    _raw = _spk._dev if hasattr(_spk, '_dev') else _spk
    _iface    = _raw.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    _vol_ctrl = cast(_iface, POINTER(IAudioEndpointVolume))
    _vrange   = _vol_ctrl.GetVolumeRange()
    VOL_MIN, VOL_MAX = _vrange[0], _vrange[1]
    AUDIO_OK  = True
except Exception as _e:
    AUDIO_OK  = False
    print(f"[warn] Audio unavailable: {_e}")

# ──────────────────────────── MEDIAPIPE INIT ──────────────────────────────────
_mp_hands   = mp.solutions.hands
_mp_draw    = mp.solutions.drawing_utils

_detector   = _mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
)

LM_STYLE  = _mp_draw.DrawingSpec(color=(0, 255, 140), thickness=3, circle_radius=4)
CON_STYLE = _mp_draw.DrawingSpec(color=(40, 200, 100), thickness=2)

# ──────────────────────────── CAMERA INIT ────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
cap.set(cv2.CAP_PROP_FPS, 60)

ACT_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
ACT_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# ──────────────────────────── RUNTIME STATE ──────────────────────────────────
mode_idx      = 0
last_act_t    = 0

# Mouse
pm_x, pm_y   = 0.0, 0.0

# Scroll
last_scroll_y = None

# Swipe
swipe_x0      = None
swipe_t0      = None

# Training state machine
T_IDLE, T_NAMING, T_RECORD = 0, 1, 2
train_state   = T_IDLE
name_buf      = ""
train_name    = ""
train_samples = []

# FPS
_fps_t  = time.time()
_fps_c  = 0
fps_val = 0

# Load gestures
custom_gestures = {}
if os.path.exists(GESTURE_FILE):
    try:
        with open(GESTURE_FILE) as _f:
            custom_gestures = json.load(_f)
    except Exception:
        pass

# ──────────────────────────── UTILITY FUNCTIONS ───────────────────────────────

def get_landmarks(hlms, frame):
    """Extract pixel-space (id, x, y) landmarks for one hand."""
    h, w = frame.shape[:2]
    return [(i, int(l.x * w), int(l.y * h))
            for i, l in enumerate(hlms.landmark)]


def fingers_up(lm, hand="Right"):
    """
    Return [thumb, index, middle, ring, pinky] as 0/1.
    Thumb uses X-axis (mirrored), others use Y-axis.
    """
    out = []
    # Thumb
    if hand == "Right":
        out.append(1 if lm[4][1] < lm[3][1] else 0)
    else:
        out.append(1 if lm[4][1] > lm[3][1] else 0)
    # Fingers: tip y < pip y → extended
    for tip, pip in [(8,6), (12,10), (16,14), (20,18)]:
        out.append(1 if lm[tip][2] < lm[pip][2] else 0)
    return out


def pt_dist(p1, p2):
    return math.hypot(p2[0]-p1[0], p2[1]-p1[1])


def lerp(v, a, b, c, d):
    """Linear map v from [a,b] → [c,d]."""
    if b == a:
        return c
    return c + (v-a) * (d-c) / (b-a)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def save_gestures():
    with open(GESTURE_FILE, "w") as _f:
        json.dump(custom_gestures, _f, indent=2)


# ──────────────────────────── DRAWING HELPERS ────────────────────────────────

def draw_pill_bg(img, x1, y1, x2, y2, color, alpha=0.72):
    """Draw a rounded-rect pill with alpha blend."""
    ov  = img.copy()
    r   = (y2-y1)//2
    mid = (y1+y2)//2
    cv2.rectangle(ov, (x1+r, y1), (x2-r, y2), color, -1)
    cv2.circle(ov, (x1+r, mid), r, color, -1)
    cv2.circle(ov, (x2-r, mid), r, color, -1)
    cv2.addWeighted(ov, alpha, img, 1-alpha, 0, img)


def status_tag(img, text, color, y=72, x=12):
    """Draw a pill-shaped status tag."""
    font  = cv2.FONT_HERSHEY_SIMPLEX
    (tw,th),_ = cv2.getTextSize(text, font, 0.62, 2)
    draw_pill_bg(img, x, y, x+tw+22, y+th+12, P["bg"], alpha=0.78)
    cv2.putText(img, text, (x+11, y+th+4), font, 0.62, color, 2)


def draw_v_bar(img, x, y, w, h, pct, color, label=""):
    """Draw a vertical progress bar."""
    cv2.rectangle(img, (x, y), (x+w, y+h), P["darkgray"], -1)
    cv2.rectangle(img, (x, y), (x+w, y+h), P["gray"], 1)
    fill = int(clamp(pct,0,100) / 100 * h)
    if fill > 0:
        cv2.rectangle(img, (x+1, y+h-fill), (x+w-1, y+h-1), color, -1)
        # Highlight stripe
        cv2.rectangle(img, (x+1, y+h-fill), (x+4, y+h-1), tuple(min(c+60,255) for c in color), -1)
    if label:
        cv2.putText(img, label, (x-2, y-9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    cv2.putText(img, f"{int(pct)}%", (x-2, y+h+16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)


def draw_cursor(img, x, y, color, clicking=False):
    """Draw a crosshair cursor at (x,y)."""
    r  = 20 if clicking else 13
    cl = P["red"] if clicking else color
    cv2.circle(img, (x, y), r, cl, 2)
    cv2.circle(img, (x, y), 3, cl, -1)
    for dx, dy in [(-r-8,0),(r+8,0),(0,-r-8),(0,r+8)]:
        cv2.line(img, (x+dx//3, y+dy//3), (x+dx, y+dy), cl, 2)


def draw_top_bar(img, mode):
    """Draw the mode-selection top bar."""
    h, w   = img.shape[:2]
    bar_h  = 60
    ov     = img.copy()
    cv2.rectangle(ov, (0,0), (w, bar_h), (8,10,16), -1)
    cv2.addWeighted(ov, 0.88, img, 0.12, 0, img)
    cv2.line(img, (0, bar_h), (w, bar_h), (45,50,62), 1)

    tab_w = w // len(MODES)
    for i, m in enumerate(MODES):
        x1    = i * tab_w
        color = MODE_CLR[m]
        active= (m == mode)
        if active:
            ov2 = img.copy()
            cv2.rectangle(ov2, (x1+3, 5), (x1+tab_w-3, bar_h-3), color, -1)
            cv2.addWeighted(ov2, 0.22, img, 0.78, 0, img)
            cv2.rectangle(img, (x1+3, 5), (x1+tab_w-3, bar_h-3), color, 2)
            cv2.putText(img, m, (x1+10, 37),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 2)
        else:
            cv2.putText(img, m, (x1+10, 37),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, P["gray"], 1)


def draw_pinch_line(img, p1, p2, color):
    """Draw the thumb-index pinch indicator."""
    mid = ((p1[0]+p2[0])//2, (p1[1]+p2[1])//2)
    cv2.line(img,  p1, p2, color, 3)
    cv2.circle(img, p1,  10, color, -1)
    cv2.circle(img, p2,  10, color, -1)
    cv2.circle(img, mid,  7, P["white"], -1)
    return mid


def hint(img, text):
    """Bottom-of-frame hint text."""
    h = img.shape[0]
    cv2.putText(img, text, (10, h-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.41, P["gray"], 1)


# ──────────────────────────── MODE HANDLERS ──────────────────────────────────

def mode_mouse(img, lm, fingers):
    """Virtual mouse: index → move, pinch index → click, pinch middle → right-click."""
    global pm_x, pm_y, last_act_t

    ix, iy = lm[8][1], lm[8][2]
    thumb  = (lm[4][1], lm[4][2])
    index  = (lm[8][1], lm[8][2])
    middle = (lm[12][1], lm[12][2])

    if fingers[1]:   # Index finger up → move cursor
        mx = lerp(ix, 80, ACT_W-80, 0, SCREEN_W)
        my = lerp(iy, 80, ACT_H-80, 0, SCREEN_H)
        mx, my = clamp(mx,0,SCREEN_W-1), clamp(my,0,SCREEN_H-1)

        sx = pm_x + (mx-pm_x)/SMOOTH_FACTOR
        sy = pm_y + (my-pm_y)/SMOOTH_FACTOR
        pyautogui.moveTo(int(sx), int(sy))
        pm_x, pm_y = sx, sy

        d_click = pt_dist(thumb, index)
        clicking = d_click < 38
        draw_cursor(img, ix, iy, MODE_CLR["MOUSE"], clicking)

        if clicking:
            t = time.time()
            if t - last_act_t > 0.5:
                pyautogui.click()
                last_act_t = t
            status_tag(img, "LEFT CLICK", P["red"])
        else:
            status_tag(img, f"Mouse  {int(sx)}, {int(sy)}", MODE_CLR["MOUSE"])

        d_rclick = pt_dist(thumb, middle)
        if d_rclick < 38:
            t = time.time()
            if t - last_act_t > 0.6:
                pyautogui.rightClick()
                last_act_t = t
            status_tag(img, "RIGHT CLICK", P["train"], y=110)
    else:
        status_tag(img, "Raise index to activate", P["gray"])

    hint(img, "Index: move cursor | Pinch index-thumb: click | Pinch middle-thumb: right-click")


def mode_volume(img, lm, fingers):
    """Volume: thumb-index pinch distance → system volume."""
    global last_act_t
    if not AUDIO_OK:
        status_tag(img, "Audio API unavailable", P["red"])
        return

    thumb = (lm[4][1], lm[4][2])
    index = (lm[8][1], lm[8][2])
    d     = pt_dist(thumb, index)
    draw_pinch_line(img, thumb, index, MODE_CLR["VOLUME"])

    d_c     = clamp(d, 25, 230)
    vol_db  = lerp(d_c, 25, 230, VOL_MIN, VOL_MAX)
    vol_pct = lerp(d_c, 25, 230, 0, 100)
    _vol_ctrl.SetMasterVolumeLevel(vol_db, None)

    w = img.shape[1]
    draw_v_bar(img, w-58, 68, 28, 210, vol_pct, MODE_CLR["VOLUME"], "VOL")
    status_tag(img, f"Volume  {int(vol_pct)}%", MODE_CLR["VOLUME"])

    if d_c <= 30:
        status_tag(img, "MUTED", P["red"], y=110)

    hint(img, "Spread thumb & index to raise volume | Close to lower")


def mode_brightness(img, lm, fingers):
    """Brightness: thumb-index pinch distance → screen brightness."""
    thumb = (lm[4][1], lm[4][2])
    index = (lm[8][1], lm[8][2])
    d     = pt_dist(thumb, index)
    draw_pinch_line(img, thumb, index, MODE_CLR["BRIGHTNESS"])

    d_c    = clamp(d, 25, 230)
    bright = int(lerp(d_c, 25, 230, 0, 100))

    try:
        sbc.set_brightness(bright)
    except Exception:
        pass

    w = img.shape[1]
    draw_v_bar(img, w-58, 68, 28, 210, bright, MODE_CLR["BRIGHTNESS"], "BRIGHT")
    status_tag(img, f"Brightness  {bright}%", MODE_CLR["BRIGHTNESS"])

    hint(img, "Spread thumb & index to raise brightness | Close to lower")


def mode_scroll(img, lm, fingers):
    """Scroll: index finger vertical movement → page scroll."""
    global last_scroll_y

    h, w = img.shape[:2]
    iy   = lm[8][2]
    ix   = lm[8][1]

    # Draw scroll track
    tx = w // 2
    cv2.rectangle(img, (tx-30, 75), (tx+30, h-65), P["darkgray"], -1)
    cv2.rectangle(img, (tx-30, 75), (tx+30, h-65), (55,58,72), 1)

    # Indicator dot on track
    track_pct = clamp(lerp(iy, 80, h-80, 0, 100), 0, 100)
    dot_y     = int(lerp(track_pct, 0, 100, 80, h-70))
    cv2.rectangle(img, (tx-28, dot_y-6), (tx+28, dot_y+6), MODE_CLR["SCROLL"], -1)

    # Up/down arrows
    cv2.arrowedLine(img, (tx, 90), (tx, 72), MODE_CLR["SCROLL"], 2, tipLength=0.5)
    cv2.arrowedLine(img, (tx, h-80), (tx, h-65), MODE_CLR["SCROLL"], 2, tipLength=0.5)

    # Finger dot
    cv2.circle(img, (ix, iy), 10, MODE_CLR["SCROLL"], -1)
    cv2.circle(img, (ix, iy),  4, P["white"], -1)

    if last_scroll_y is not None:
        delta = last_scroll_y - iy
        if abs(delta) > 7:
            scroll_amt = int(delta / 7)
            pyautogui.scroll(scroll_amt)
            label = "^ Scroll UP" if delta > 0 else "v Scroll DOWN"
            status_tag(img, label, MODE_CLR["SCROLL"])

    last_scroll_y = iy
    hint(img, "Move index finger up to scroll up | Move down to scroll down")


def mode_presentation(img, lm, fingers):
    """Presentation: open-palm horizontal swipe → arrow keys for slides."""
    global swipe_x0, swipe_t0, last_act_t

    h, w   = img.shape[:2]
    wrist_x = lm[0][1]

    # Draw slide arrows
    arrow_color = MODE_CLR["PRESENTATION"]
    cv2.arrowedLine(img, (55, h//2), (165, h//2), arrow_color, 3, tipLength=0.35)
    cv2.putText(img, "PREV", (30, h//2-14), cv2.FONT_HERSHEY_SIMPLEX, 0.52, arrow_color, 1)
    cv2.arrowedLine(img, (w-55, h//2), (w-165, h//2), arrow_color, 3, tipLength=0.35)
    cv2.putText(img, "NEXT", (w-80, h//2-14), cv2.FONT_HERSHEY_SIMPLEX, 0.52, arrow_color, 1)

    is_open = sum(fingers) >= 4

    if is_open:
        cv2.circle(img, (lm[0][1], lm[0][2]), 22, arrow_color, 2)
        status_tag(img, "Open palm - Swipe!", arrow_color)

        if swipe_x0 is None:
            swipe_x0 = wrist_x
            swipe_t0 = time.time()
        else:
            elapsed = time.time() - swipe_t0
            diff    = wrist_x - swipe_x0

            if abs(diff) > SWIPE_DIST and elapsed < SWIPE_DUR:
                t = time.time()
                if t - last_act_t > 0.8:
                    if diff > 0:          # swipe right -> prev
                        pyautogui.press("left")
                        status_tag(img, "<< PREV SLIDE", arrow_color, y=110)
                    else:                 # swipe left -> next
                        pyautogui.press("right")
                        status_tag(img, "NEXT SLIDE >>", arrow_color, y=110)
                    last_act_t = t
                swipe_x0 = None

            elif elapsed >= SWIPE_DUR:
                swipe_x0 = None

            # Progress indicator
            if swipe_x0 is not None:
                prog = abs(diff) / SWIPE_DIST
                bar_w = int(prog * 200)
                bar_x = w//2 - 100
                cv2.rectangle(img, (bar_x, h-50), (bar_x+200, h-38), P["darkgray"], -1)
                if bar_w:
                    c = P["red"] if diff < 0 else P["green"]
                    cv2.rectangle(img, (bar_x, h-50), (bar_x+bar_w, h-38), c, -1)
    else:
        swipe_x0 = None
        status_tag(img, "Open all 5 fingers then swipe", P["gray"])

    hint(img, "Open palm + swipe right -> prev slide | swipe left -> next slide")


def mode_train(img, lm, fingers):
    """Custom gesture training: record, name, and save gesture patterns."""
    global train_state, name_buf, train_name, train_samples

    h, w        = img.shape[:2]
    finger_str  = "".join(str(f) for f in fingers)
    font        = cv2.FONT_HERSHEY_SIMPLEX

    # ── Panel background ────────────────────────────────────────────────────
    panel_h = 28 + 24 * max(len(custom_gestures), 1) + 30
    ov = img.copy()
    cv2.rectangle(ov, (8, 68), (370, 68+panel_h), (12,15,22), -1)
    cv2.rectangle(ov, (8, 68), (370, 68+panel_h), (45,50,65), 1)
    cv2.addWeighted(ov, 0.82, img, 0.18, 0, img)

    # ── Saved gestures list ─────────────────────────────────────────────────
    cv2.putText(img, "Saved Gestures", (16, 88), font, 0.54, MODE_CLR["TRAIN"], 1)
    if not custom_gestures:
        cv2.putText(img, "  (none yet — press R to record)", (16, 112),
                    font, 0.43, P["gray"], 1)
    for i, (gname, gdata) in enumerate(list(custom_gestures.items())):
        matched = (fingers == gdata["fingers"])
        col     = P["accent"] if matched else P["white"]
        bullet  = "*" if matched else "-"
        pat     = "".join(str(x) for x in gdata["fingers"])
        cv2.putText(img, f"  {bullet} {gname}: [{pat}]",
                    (16, 112 + 23*i), font, 0.46, col, 1)

    # ── Current finger readout ──────────────────────────────────────────────
    fw = 46
    for fi, fval in enumerate(fingers):
        fx  = 15 + fi * (fw+6)
        fy  = h-105
        col = MODE_CLR["TRAIN"] if fval else P["darkgray"]
        cv2.rectangle(img, (fx, fy), (fx+fw, fy+40), col, -1)
        cv2.rectangle(img, (fx, fy), (fx+fw, fy+40), P["gray"], 1)
        label = ["T","I","M","R","P"][fi]
        tc    = P["bg"] if fval else P["gray"]
        cv2.putText(img, label, (fx+14, fy+27), font, 0.65, tc, 2)
    cv2.putText(img, "T  I  M  R  P  (Thumb/Index/Middle/Ring/Pinky)",
                (15, h-58), font, 0.38, P["gray"], 1)

    # ── State-specific UI ───────────────────────────────────────────────────
    if train_state == T_IDLE:
        cv2.putText(img, "R: record new gesture  |  D: delete last gesture",
                    (10, h-28), font, 0.46, P["gray"], 1)

    elif train_state == T_NAMING:
        cv2.putText(img, "Gesture name:", (10, h-130), font, 0.55, MODE_CLR["TRAIN"], 1)
        disp = name_buf + "|"
        (tw,th),_ = cv2.getTextSize(disp, font, 0.72, 2)
        cv2.rectangle(img, (8, h-120), (tw+36, h-93), P["panel"], -1)
        cv2.rectangle(img, (8, h-120), (tw+36, h-93), MODE_CLR["TRAIN"], 1)
        cv2.putText(img, disp, (14, h-100), font, 0.72, P["white"], 2)
        cv2.putText(img, "Enter: confirm   Esc: cancel",
                    (10, h-28), font, 0.45, P["gray"], 1)

    elif train_state == T_RECORD:
        blink = int(time.time()*3) % 2
        if blink:
            cv2.circle(img, (18, h-133), 9, P["red"], -1)
        cv2.putText(img, f"Recording  '{train_name}'", (32, h-126),
                    font, 0.62, P["red"], 2)

        prog = min(len(train_samples), 60)
        cv2.rectangle(img, (8, h-112), (360, h-96), P["darkgray"], -1)
        if prog:
            cv2.rectangle(img, (8, h-112),
                          (8 + int(352*prog/60), h-96), MODE_CLR["TRAIN"], -1)
        cv2.putText(img, f"Samples: {prog}/60   hold still!",
                    (10, h-80), font, 0.5, P["white"], 1)
        cv2.putText(img, "S: save gesture   Esc: cancel",
                    (10, h-28), font, 0.46, P["gray"], 1)
        train_samples.append(fingers[:])

    # ── Live match indicator ─────────────────────────────────────────────────
    for gname, gdata in custom_gestures.items():
        if fingers == gdata["fingers"]:
            status_tag(img, f"[MATCH] {gname} detected", MODE_CLR["TRAIN"])
            break

    hint(img, "Finger pattern: [Thumb Index Middle Ring Pinky]  (1=up, 0=down)")


# ──────────────────────────── MAIN LOOP ──────────────────────────────────────

def main():
    global mode_idx, last_act_t
    global pm_x, pm_y
    global last_scroll_y
    global swipe_x0, swipe_t0
    global train_state, name_buf, train_name, train_samples
    global custom_gestures
    global _fps_t, _fps_c, fps_val

    print("=" * 56)
    print("  Advanced Gesture Controller  -  starting")
    print(f"  Screen : {SCREEN_W} x {SCREEN_H}")
    print(f"  Camera : {ACT_W} x {ACT_H}")
    print(f"  Audio  : {'[OK] ready' if AUDIO_OK else '[--] unavailable'}")
    print("=" * 56)
    print("  M / Tab  -> cycle modes           Q -> quit")
    print("  TRAIN:  R=record  Enter=confirm  S=save  Esc=cancel  D=delete")
    print("=" * 56)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res   = _detector.process(rgb)

        mode  = MODES[mode_idx]

        # ── Detect hands ────────────────────────────────────────────────────
        hand_list = []
        if res.multi_hand_landmarks and res.multi_handedness:
            for hlms, hinfo in zip(res.multi_hand_landmarks,
                                    res.multi_handedness):
                side = hinfo.classification[0].label  # "Left" / "Right"
                lm   = get_landmarks(hlms, frame)
                fup  = fingers_up(lm, side)

                _mp_draw.draw_landmarks(frame, hlms,
                                         _mp_hands.HAND_CONNECTIONS,
                                         LM_STYLE, CON_STYLE)

                # Hand label
                label_col = MODE_CLR["MOUSE"] if side == "Right" else MODE_CLR["TRAIN"]
                cv2.putText(frame, side, (lm[0][1]-16, lm[0][2]-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, label_col, 2)

                hand_list.append({"lm": lm, "fingers": fup, "side": side})

        # ── Dispatch to active mode ──────────────────────────────────────────
        if hand_list:
            pri  = hand_list[0]
            lm   = pri["lm"]
            fing = pri["fingers"]

            if   mode == "MOUSE":        mode_mouse(frame, lm, fing)
            elif mode == "VOLUME":       mode_volume(frame, lm, fing)
            elif mode == "BRIGHTNESS":   mode_brightness(frame, lm, fing)
            elif mode == "SCROLL":       mode_scroll(frame, lm, fing)
            elif mode == "PRESENTATION": mode_presentation(frame, lm, fing)
            elif mode == "TRAIN":        mode_train(frame, lm, fing)

            # ── Dual-hand: volume + brightness simultaneously ────────────────
            if len(hand_list) == 2 and mode in ("VOLUME", "BRIGHTNESS"):
                sec = hand_list[1]
                if mode == "VOLUME":
                    mode_brightness(frame, sec["lm"], sec["fingers"])
                else:
                    mode_volume(frame, sec["lm"], sec["fingers"])

            # ── Custom gesture overlay (non-train modes) ─────────────────────
            if mode != "TRAIN":
                for gname, gdata in custom_gestures.items():
                    if fing == gdata["fingers"]:
                        status_tag(frame, f"[{gname}]", P["accent"], y=145)
                        break

        else:
            # No hands detected — reset stateful vars
            last_scroll_y = None
            swipe_x0      = None
            if mode not in ("TRAIN",):
                status_tag(frame, "No hand detected", P["gray"])

        # ── Multi-hand count indicator ───────────────────────────────────────
        h_img, w_img = frame.shape[:2]
        hand_col = P["accent"] if len(hand_list) > 0 else P["gray"]
        cv2.putText(frame, f"Hands: {len(hand_list)}",
                    (w_img-115, h_img-28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, hand_col, 1)

        # ── FPS ──────────────────────────────────────────────────────────────
        _fps_c += 1
        if time.time() - _fps_t >= 1.0:
            fps_val = _fps_c
            _fps_c  = 0
            _fps_t  = time.time()
        cv2.putText(frame, f"FPS: {fps_val}",
                    (w_img-115, h_img-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, P["gray"], 1)

        # ── Top bar ──────────────────────────────────────────────────────────
        draw_top_bar(frame, mode)

        cv2.imshow("Advanced Gesture Controller", frame)

        # ── Key handling ─────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            # Show quit confirmation overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (frame.shape[1]//2-160, frame.shape[0]//2-40),
                          (frame.shape[1]//2+160, frame.shape[0]//2+50), (10,12,20), -1)
            cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
            cv2.putText(frame, "Press Q again to quit",
                        (frame.shape[1]//2-145, frame.shape[0]//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80,80,220), 2)
            cv2.putText(frame, "Press any other key to continue",
                        (frame.shape[1]//2-145, frame.shape[0]//2+32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, P["gray"], 1)
            cv2.imshow("Advanced Gesture Controller", frame)
            confirm = cv2.waitKey(2000) & 0xFF
            if confirm == ord('q'):
                break

        elif key in (ord('m'), 9):   # M or Tab → next mode
            mode_idx       = (mode_idx + 1) % len(MODES)
            last_scroll_y  = None
            swipe_x0       = None

        # Training-specific keys
        elif mode == "TRAIN":

            if train_state == T_IDLE:
                if key == ord('r'):
                    train_state = T_NAMING
                    name_buf    = ""
                elif key == ord('d') and custom_gestures:
                    last_k = list(custom_gestures.keys())[-1]
                    del custom_gestures[last_k]
                    save_gestures()
                    print(f"[train] Deleted gesture '{last_k}'")

            elif train_state == T_NAMING:
                if key == 13:             # Enter → confirm name
                    stripped = name_buf.strip()
                    if stripped:
                        train_name    = stripped
                        train_state   = T_RECORD
                        train_samples = []
                elif key == 27:           # Esc → cancel
                    train_state = T_IDLE
                elif key == 8:            # Backspace
                    name_buf = name_buf[:-1]
                elif 32 <= key <= 126:    # Printable ASCII
                    name_buf += chr(key)

            elif train_state == T_RECORD:
                if key == ord('s'):       # S → save
                    if train_samples:
                        voted = []
                        for fi in range(5):
                            v = [s[fi] for s in train_samples]
                            voted.append(1 if sum(v) > len(v)//2 else 0)
                        custom_gestures[train_name] = {"fingers": voted}
                        save_gestures()
                        print(f"[train] Saved '{train_name}': {voted}")
                    train_state   = T_IDLE
                    train_name    = ""
                    train_samples = []
                    name_buf      = ""
                elif key == 27:           # Esc → cancel
                    train_state   = T_IDLE
                    train_name    = ""
                    train_samples = []

    cap.release()
    cv2.destroyAllWindows()
    print("Gesture Controller stopped.")


if __name__ == "__main__":
    main()
