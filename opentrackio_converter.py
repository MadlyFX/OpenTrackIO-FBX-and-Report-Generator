#!/usr/bin/env python3
"""
OpenTrackIO to FBX Converter + PDF Report Generator
====================================================
Converts a folder of OpenTrackIO JSON frame files into:
  1. An animated FBX camera file (binary FBX 7.4, compatible with Maya,
     Blender, Unreal Engine, Nuke, and other DCCs)
  2. A PDF report with summary statistics, charts, and lens/sensor metadata

Single sequence:
    python opentrackio_converter.py <sequence_folder> [options]

Batch mode (multiple sequences):
    python opentrackio_converter.py <top_level_folder> --batch [options]

    In batch mode each immediate sub-folder of <top_level_folder> that
    contains .json files is treated as an independent sequence.  Outputs
    are written to a matching sub-folder inside --output-dir (or alongside
    each sequence folder when --output-dir is omitted).

    Batch mode is also triggered automatically when the input folder
    contains no .json files directly but does contain sub-folders that do.

Options:
    --fps FPS           Override frames per second (default: auto-detect)
    --output-dir DIR    Output directory (default: same as input folder)
    --fbx-name NAME     FBX output filename  (default: camera_animation.fbx)
    --pdf-name NAME     PDF output filename  (default: camera_report.pdf)
    --no-fbx            Skip FBX generation
    --no-pdf            Skip PDF report generation
    --batch             Force batch mode (process sub-folders as sequences)

Requirements:
    pip install matplotlib numpy reportlab

OpenTrackIO spec: https://www.opentrackio.org
"""

import os
import sys
import json
import math
import argparse
import tempfile
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, HRFlowable, KeepTogether,
)

# ═══════════════════════════════════════════════════════════════
#  OPENTRACKIO PARSER
# ═══════════════════════════════════════════════════════════════

class OTFrame:
    """One parsed OpenTrackIO sample."""

    __slots__ = [
        "frame_number", "timecode", "fps_num", "fps_denom",
        # Transform
        "tx", "ty", "tz",          # metres
        "pan", "tilt", "roll",     # degrees (Y / X / Z rotation)
        # Lens
        "focal_length",            # mm
        "focus_distance",          # metres
        "f_stop",                  # f-number or None
        "nominal_focal_length",    # mm or None
        "entrance_pupil_offset",   # mm or None
        # Camera body
        "sensor_width",            # mm
        "sensor_height",           # mm
        "sensor_pixel_width",      # px or None
        "sensor_pixel_height",     # px or None
        "camera_make",
        "camera_model",
        "camera_serial",
        "camera_label",
        "anamorphic_squeeze",
        "_raw",
    ]

    def __init__(self):
        self.frame_number = 0
        self.timecode = None
        self.fps_num = 24
        self.fps_denom = 1
        self.tx = self.ty = self.tz = 0.0
        self.pan = self.tilt = self.roll = 0.0
        self.focal_length = 35.0
        self.focus_distance = 2.0
        self.f_stop = None
        self.nominal_focal_length = None
        self.entrance_pupil_offset = None
        self.sensor_width = 36.0
        self.sensor_height = 24.0
        self.sensor_pixel_width = None
        self.sensor_pixel_height = None
        self.camera_make = None
        self.camera_model = None
        self.camera_serial = None
        self.camera_label = None
        self.anamorphic_squeeze = 1.0
        self._raw = {}

    @property
    def fps(self):
        return self.fps_num / max(self.fps_denom, 1)


def _g(d, *keys, default=None):
    """Safe nested dict getter: _g(d, 'a', 'b', 'c') → d['a']['b']['c']."""
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k)
        if d is None:
            return default
    return d if d is not None else default


def _flt(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def parse_frame(data: dict) -> OTFrame:
    """
    Parse one OpenTrackIO JSON dict into an OTFrame.

    Handles both the idealised schema and the real-world layout where
    static camera metadata lives under 'static.camera' and frame rate
    lives under 'timing.timecode.format.frameRate'.
    """
    f = OTFrame()
    f._raw = data

    timing = data.get("timing") or {}

    # ── Frame rate ─────────────────────────────────────────────
    # Real-world location: timing.timecode.format.frameRate
    # Alternative:         timing.frameRate  /  timing.sampleRate
    tc_block = timing.get("timecode") or {}
    fps_raw = (
        _g(tc_block, "format", "frameRate") or
        timing.get("frameRate") or
        timing.get("sampleRate") or {}
    )
    if isinstance(fps_raw, dict):
        f.fps_num   = int(_g(fps_raw, "num")   or _g(fps_raw, "numerator")   or 24)
        f.fps_denom = int(_g(fps_raw, "denom") or _g(fps_raw, "denominator") or 1)
    elif isinstance(fps_raw, (int, float)):
        f.fps_num, f.fps_denom = int(fps_raw), 1

    # ── Timecode & frame number ────────────────────────────────
    tc = tc_block or _g(data, "timecode")
    if isinstance(tc, dict):
        h  = int(tc.get("hours")   or tc.get("h")  or 0)
        m  = int(tc.get("minutes") or tc.get("m")  or 0)
        s  = int(tc.get("seconds") or tc.get("s")  or 0)
        fr = int(tc.get("frames")  or tc.get("f")  or 0)
        f.timecode = f"{h:02d}:{m:02d}:{s:02d}:{fr:02d}"
        # Derive absolute frame number from timecode
        fps_int = f.fps_num // f.fps_denom if f.fps_denom else 24
        f.frame_number = h * 3600 * fps_int + m * 60 * fps_int + s * fps_int + fr
    elif isinstance(tc, str):
        f.timecode = tc

    # Also accept an explicit frameNumber if present
    explicit_fn = _g(timing, "frameNumber") or _g(data, "frameNumber") or _g(data, "frame")
    if explicit_fn is not None:
        f.frame_number = int(explicit_fn)

    # Also check timing.sampleTimecode string  e.g. "00:00:04:17"
    if not f.timecode:
        stc = timing.get("sampleTimecode")
        if isinstance(stc, str) and stc:
            f.timecode = stc

    # ── Static metadata (real-world: 'static.camera') ─────────
    static  = data.get("static") or {}
    # 'camera' block may be top-level or nested under 'static'
    camera  = data.get("camera") or static.get("camera") or {}

    # ── Transform ──────────────────────────────────────────────
    # Check both top-level 'transforms' array and 'static.transforms'
    transforms = data.get("transforms") or static.get("transforms") or []
    transform = {}
    if transforms:
        transform = transforms[0] if isinstance(transforms, list) else transforms
    elif "globalStage" in data:
        transform = data["globalStage"]
    elif "transform" in data:
        transform = data["transform"]

    if transform:
        t = transform.get("translation") or {}
        f.tx = _flt(_g(t, "x"))
        f.ty = _flt(_g(t, "y"))
        f.tz = _flt(_g(t, "z"))
        r = transform.get("rotation") or {}
        f.pan  = _flt(_g(r, "pan")  or _g(r, "y"))
        f.tilt = _flt(_g(r, "tilt") or _g(r, "x"))
        f.roll = _flt(_g(r, "roll") or _g(r, "z"))

    # ── Lens ───────────────────────────────────────────────────
    lens = data.get("lens") or {}

    f.focal_length = _flt(
        _g(lens, "focalLength") or
        _g(lens, "pinholeFocalLength") or      # real-world field name
        _g(lens, "effectiveFocalLength") or
        _g(camera, "focalLength") or 35.0
    )
    f.focus_distance = _flt(
        _g(lens, "focusDistance") or
        _g(camera, "focusDistance") or 2.0
    )
    fs = (
        _g(lens, "fStop") or
        _g(lens, "aperture") or
        _g(camera, "apertureValue")
    )
    # fStop of 0 from raw encoders means "not available" — treat as None
    if fs is not None:
        fs_val = _flt(fs)
        f.f_stop = fs_val if fs_val > 0 else None
    else:
        f.f_stop = None

    nfl = _g(lens, "nominalFocalLength")
    f.nominal_focal_length = _flt(nfl) if nfl is not None else None

    epo = _g(lens, "entrancePupilOffset") or _g(lens, "entrancePupilDistance")
    f.entrance_pupil_offset = _flt(epo) if epo is not None else None

    # ── Camera body / sensor ───────────────────────────────────
    # Check both 'camera' and 'static.camera'
    sdims = (
        _g(camera, "activeSensorPhysicalDimensions") or
        _g(camera, "sensorPhysicalDimensions") or {}
    )
    if sdims:
        f.sensor_width  = _flt(_g(sdims, "width"),  36.0)
        f.sensor_height = _flt(_g(sdims, "height"), 24.0)

    pdims = (
        _g(camera, "activeSensorResolution") or
        _g(camera, "sensorPixelDimensions") or {}
    )
    if pdims:
        f.sensor_pixel_width  = _g(pdims, "width")
        f.sensor_pixel_height = _g(pdims, "height")

    f.camera_make   = _g(camera, "make")   or _g(camera, "manufacturer")
    f.camera_model  = _g(camera, "model")
    f.camera_serial = _g(camera, "serialNumber") or _g(camera, "serial")
    f.camera_label  = _g(camera, "label")  or _g(camera, "id")
    f.anamorphic_squeeze = _flt(_g(camera, "anamorphicSqueeze"), 1.0)

    return f


def _read_ndjson(path) -> list:
    """
    Read a file that may contain one OR multiple JSON objects.

    Handles three formats:
      • Standard JSON  – single object or array
      • NDJSON         – one JSON object per line  (most common in OpenTrackIO)
      • Concatenated   – multiple objects without newline separation
    Returns a list of dicts.
    """
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read().strip()

    if not text:
        return []

    # Fast path: valid standard JSON
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, list) else [obj]
    except json.JSONDecodeError:
        pass

    # NDJSON: try each non-empty line
    objects = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            objects.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # skip malformed lines

    if objects:
        return objects

    # Last resort: streaming decoder for concatenated objects
    objects, idx = [], 0
    decoder = json.JSONDecoder()
    while idx < len(text):
        while idx < len(text) and text[idx] in " \t\n\r":
            idx += 1
        if idx >= len(text):
            break
        try:
            obj, end = decoder.raw_decode(text, idx)
            objects.append(obj)
            idx = end
        except json.JSONDecodeError:
            idx += 1

    return objects


def _merge_samples(samples: list) -> OTFrame:
    """
    Merge multiple sub-frame samples from one file into a single OTFrame.
    Numerical values are averaged; string/metadata fields come from the
    first sample that has them.
    """
    if len(samples) == 1:
        return parse_frame(samples[0])

    frames = [parse_frame(s) for s in samples]
    merged = frames[0]

    # Average all numerical per-frame fields
    num_fields = ["tx", "ty", "tz", "pan", "tilt", "roll",
                  "focal_length", "focus_distance"]
    for field in num_fields:
        vals = [getattr(fr, field) for fr in frames]
        setattr(merged, field, sum(vals) / len(vals))

    fstops = [fr.f_stop for fr in frames if fr.f_stop is not None]
    merged.f_stop = sum(fstops) / len(fstops) if fstops else None

    return merged


def load_folder(folder: str) -> list:
    """Load, parse and sort all .json files from *folder*.

    Each file may contain one frame (standard JSON) or multiple
    sub-frame samples (NDJSON). Sub-frame samples are averaged into
    one OTFrame per file.
    """
    p = Path(folder)
    files = sorted(p.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No .json files found in: {folder}")
    print(f"  Loading {len(files)} JSON files from {p.name}/…")

    frames, errors = [], 0
    for file_idx, jf in enumerate(files):
        try:
            samples = _read_ndjson(jf)
            if not samples:
                raise ValueError("file is empty or contains no valid JSON")

            frm = _merge_samples(samples)

            # If frame number couldn't be derived from timecode, fall back
            # to the numeric stem of the filename (e.g. "00000417" → 417)
            if frm.frame_number == 0 and file_idx > 0:
                stem_digits = "".join(c for c in jf.stem if c.isdigit())
                frm.frame_number = int(stem_digits) if stem_digits else file_idx

            frames.append(frm)
        except Exception as e:
            print(f"    ⚠ Skipping {jf.name}: {e}")
            errors += 1

    frames.sort(key=lambda x: x.frame_number)
    print(f"  Parsed {len(frames)} frames ({errors} errors)")
    return frames


# ═══════════════════════════════════════════════════════════════
#  FBX ASCII 7.4 WRITER
# ═══════════════════════════════════════════════════════════════

KTIME_PER_SEC = 46186158000   # FBX internal time ticks per second


def _ktime(frame: int, fps: float) -> int:
    return int(frame * KTIME_PER_SEC / fps)


def write_fbx(frames: list, path: str, fps: float = 24.0) -> None:
    """
    Write an FBX ASCII 7.4 file containing a single animated camera.

    Coordinate system: Y-up, right-handed (matches Maya default).
    Units: centimetres (FBX default scene unit).

    The camera null carries translation (Tx/Ty/Tz) and rotation (Rx/Ry/Rz).
    A child Camera node carries lens parameters (FocalLength, FocusDistance).

    OpenTrackIO → FBX mapping
      translation (m)  →  Lcl Translation (cm,  ×100)
      pan  (°, Y-rot)  →  Lcl Rotation Y
      tilt (°, X-rot)  →  Lcl Rotation X
      roll (°, Z-rot)  →  Lcl Rotation Z
      focalLength (mm) →  FocalLength (mm, unchanged)
      focusDistance (m)→  FocusDistance (cm, ×100)
    """
    if not frames:
        raise ValueError("No frames to write")

    n   = len(frames)
    fps = float(fps)
    dur = _ktime(n, fps)

    # Per-frame arrays
    kt  = [_ktime(i, fps) for i in range(n)]
    tx  = [f.tx  * 100 for f in frames]
    ty  = [f.ty  * 100 for f in frames]
    tz  = [f.tz  * 100 for f in frames]
    rx  = [f.tilt        for f in frames]
    ry  = [f.pan         for f in frames]
    rz  = [f.roll        for f in frames]
    fl  = [f.focal_length                for f in frames]
    fd  = [f.focus_distance * 100        for f in frames]

    sw_in = frames[0].sensor_width  / 25.4
    sh_in = frames[0].sensor_height / 25.4
    ar    = frames[0].sensor_width  / frames[0].sensor_height
    sq    = frames[0].anamorphic_squeeze

    # Object IDs — camera is the single scene object (no null parent)
    ID_CAM       = 2001
    ID_STACK     = 3001
    ID_LAYER     = 3002
    ID_NODE_T    = 4001
    ID_NODE_R    = 4002
    ID_NODE_FL   = 4003
    ID_NODE_FD   = 4004
    ID_CRV_TX    = 5001
    ID_CRV_TY    = 5002
    ID_CRV_TZ    = 5003
    ID_CRV_RX    = 5004
    ID_CRV_RY    = 5005
    ID_CRV_RZ    = 5006
    ID_CRV_FL    = 5007
    ID_CRV_FD    = 5008

    # ── helpers ───────────────────────────────────────────────
    def key_str(times, vals):
        return ",".join(f"{t},{v:.6f},L" for t, v in zip(times, vals))

    def crv(fh, cid, name, vals):
        fh.write(
            f'\tAnimationCurve: {cid}, "AnimCurve::{name}", "" {{\n'
            f'\t\tDefault: {vals[0]:.6f}\n'
            f'\t\tKeyVer: 4008\n'
            f'\t\tKeyCount: {len(vals)}\n'
            f'\t\tKey: {key_str(kt, vals)}\n'
            f'\t\tKeyAttrFlags: 24836\n'
            f'\t\tKeyAttrDataFloat: 0,0,9.419963e+28,0\n'
            f'\t\tKeyAttrRefCount: {len(vals)}\n'
            f'\t}}\n'
        )

    # fps → FBX TimeMode enum
    _tm = {120: 12, 100: 11, 60: 9, 50: 8, 48: 7,
           30: 6, 25: 5, 24: 4}
    time_mode = _tm.get(int(round(fps)), 14)   # 14 = custom

    with open(path, "w", encoding="utf-8") as f:
        now = datetime.now()

        # ── Header ────────────────────────────────────────────
        f.write(
            "; FBX 7.4.0 project file\n"
            "; Generated by OpenTrackIO to FBX Converter\n"
            f"; Date: {now:%Y-%m-%d %H:%M:%S}\n"
            f"; Frames: {n}  FPS: {fps}\n\n"
            "FBXHeaderExtension:  {\n"
            "\tFBXHeaderVersion: 1003\n"
            "\tFBXVersion: 7400\n"
            "\tCreationTimeStamp:  {\n"
            "\t\tVersion: 1000\n"
            f"\t\tYear: {now.year}\n"
            f"\t\tMonth: {now.month}\n"
            f"\t\tDay: {now.day}\n"
            f"\t\tHour: {now.hour}\n"
            f"\t\tMinute: {now.minute}\n"
            f"\t\tSecond: {now.second}\n"
            "\t\tMillisecond: 0\n"
            "\t}\n"
            '\tCreator: "OpenTrackIO to FBX Converter"\n'
            "}\n\n"
        )

        # ── GlobalSettings ────────────────────────────────────
        f.write(
            "GlobalSettings:  {\n"
            "\tVersion: 1000\n"
            "\tProperties70:  {\n"
            '\t\tP: "UpAxis", "int", "Integer", "",1\n'
            '\t\tP: "UpAxisSign", "int", "Integer", "",1\n'
            '\t\tP: "FrontAxis", "int", "Integer", "",2\n'
            '\t\tP: "FrontAxisSign", "int", "Integer", "",1\n'
            '\t\tP: "CoordAxis", "int", "Integer", "",0\n'
            '\t\tP: "CoordAxisSign", "int", "Integer", "",1\n'
            '\t\tP: "UnitScaleFactor", "double", "Number", "",1\n'
            f'\t\tP: "TimeMode", "enum", "", "",{time_mode}\n'
            f'\t\tP: "CustomFrameRate", "double", "Number", "",{fps}\n'
            f'\t\tP: "TimeSpanStart", "KTime", "Time", "",0\n'
            f'\t\tP: "TimeSpanStop", "KTime", "Time", "",{dur}\n'
            "\t}\n"
            "}\n\n"
        )

        # ── Documents / References ─────────────────────────────
        f.write(
            "Documents:  {\n"
            "\tCount: 1\n"
            '\tDocument: 1000, "Scene", "Scene" {\n'
            '\t\tType: "Scene"\n'
            '\t\tRootNode: 0\n'
            "\t}\n"
            "}\n\n"
            "References:  {\n}\n\n"
        )

        # ── Definitions ────────────────────────────────────────
        f.write(
            "Definitions:  {\n"
            "\tVersion: 100\n"
            '\tObjectType: "NodeAttribute" {\n\t\tCount: 1\n\t}\n'
            '\tObjectType: "Model" {\n\t\tCount: 1\n\t}\n'
            '\tObjectType: "AnimationStack" {\n\t\tCount: 1\n\t}\n'
            '\tObjectType: "AnimationLayer" {\n\t\tCount: 1\n\t}\n'
            '\tObjectType: "AnimationCurveNode" {\n\t\tCount: 4\n\t}\n'
            '\tObjectType: "AnimationCurve" {\n\t\tCount: 8\n\t}\n'
            "}\n\n"
        )

        # ── Objects ────────────────────────────────────────────
        f.write("Objects:  {\n")

        # NodeAttribute — holds all camera/lens properties
        # (FocalLength animation connects HERE, not to the Model)
        ID_ATTR = ID_CAM - 1   # 2000
        import math as _math
        fov0 = 2.0 * _math.degrees(_math.atan(sw_in / (2.0 * fl[0] / 25.4)))
        fov0v = 2.0 * _math.degrees(_math.atan(sh_in / (2.0 * fl[0] / 25.4)))
        f.write(
            f'\tNodeAttribute: {ID_ATTR}, "NodeAttribute::Camera", "Camera" {{\n'
            '\t\tProperties70:  {\n'
            f'\t\t\tP: "FilmWidth","double","Number","",{sw_in:.6f}\n'
            f'\t\t\tP: "FilmHeight","double","Number","",{sh_in:.6f}\n'
            f'\t\t\tP: "FilmAspectRatio","double","Number","",{ar:.6f}\n'
            '\t\t\tP: "FilmOffsetX","double","Number","A",0\n'
            '\t\t\tP: "FilmOffsetY","double","Number","A",0\n'
            '\t\t\tP: "ApertureMode","enum","","",3\n'
            '\t\t\tP: "GateFit","enum","","",2\n'
            f'\t\t\tP: "FieldOfView","FieldOfView","","A",{fov0:.6f}\n'
            f'\t\t\tP: "FieldOfViewX","FieldOfViewX","","A",{fov0:.6f}\n'
            f'\t\t\tP: "FieldOfViewY","FieldOfViewY","","A",{fov0v:.6f}\n'
            f'\t\t\tP: "FocalLength","double","Number","A",{fl[0]:.4f}\n'
            f'\t\t\tP: "FocusDistance","double","Number","A",{fd[0]:.4f}\n'
            '\t\t\tP: "NearPlane","double","Number","",10\n'
            '\t\t\tP: "FarPlane","double","Number","",100000\n'
            '\t\t}\n'
            '\t\tTypeFlags: "Camera"\n'
            '\t\tGeometryVersion: 124\n'
            '\t\tPosition: 0,0,0\n'
            '\t\tUp: 0,1,0\n'
            '\t\tLookAt: 0,0,-1\n'
            '\t}\n'
        )

        # Model — holds transform; DefaultAttributeIndex links it to the NodeAttribute
        f.write(
            f'\tModel: {ID_CAM}, "Model::Camera", "Camera" {{\n'
            '\t\tVersion: 232\n'
            '\t\tProperties70:  {\n'
            f'\t\t\tP: "Lcl Translation","Lcl Translation","","A",{tx[0]:.6f},{ty[0]:.6f},{tz[0]:.6f}\n'
            f'\t\t\tP: "Lcl Rotation","Lcl Rotation","","A",{rx[0]:.6f},{ry[0]:.6f},{rz[0]:.6f}\n'
            '\t\t\tP: "Lcl Scaling","Lcl Scaling","","A",1,1,1\n'
            '\t\t\tP: "RotationOrder","enum","","",0\n'
            '\t\t\tP: "RotationActive","bool","","",1\n'
            '\t\t\tP: "DefaultAttributeIndex","int","Integer","",0\n'
            '\t\t}\n'
            '\t\tMultiLayer: 0\n'
            '\t\tMultiTake: 0\n'
            '\t\tShading: Y\n'
            '\t\tCulling: "CullingOff"\n'
            '\t}\n'
        )

        # AnimationStack
        f.write(
            f'\tAnimationStack: {ID_STACK}, "AnimStack::Take 001", "" {{\n'
            '\t\tProperties70:  {\n'
            '\t\t\tP: "LocalStart","KTime","Time","",0\n'
            f'\t\t\tP: "LocalStop","KTime","Time","",{dur}\n'
            '\t\t\tP: "ReferenceStart","KTime","Time","",0\n'
            f'\t\t\tP: "ReferenceStop","KTime","Time","",{dur}\n'
            '\t\t}\n'
            '\t}\n'
        )

        # AnimationLayer
        f.write(
            f'\tAnimationLayer: {ID_LAYER}, "AnimLayer::BaseLayer", "" {{\n'
            '\t}\n'
        )

        # CurveNodes
        for nid, name, px, py, pz in [
            (ID_NODE_T,  "T",           tx[0], ty[0], tz[0]),
            (ID_NODE_R,  "R",           rx[0], ry[0], rz[0]),
        ]:
            f.write(
                f'\tAnimationCurveNode: {nid}, "AnimCurveNode::{name}", "" {{\n'
                '\t\tProperties70:  {\n'
                f'\t\t\tP: "d|X","Number","","A",{px:.6f}\n'
                f'\t\t\tP: "d|Y","Number","","A",{py:.6f}\n'
                f'\t\t\tP: "d|Z","Number","","A",{pz:.6f}\n'
                '\t\t}\n'
                '\t}\n'
            )
        f.write(
            f'\tAnimationCurveNode: {ID_NODE_FL}, "AnimCurveNode::FocalLength", "" {{\n'
            '\t\tProperties70:  {\n'
            f'\t\t\tP: "d|FocalLength","Number","","A",{fl[0]:.6f}\n'
            '\t\t}\n'
            '\t}\n'
            f'\tAnimationCurveNode: {ID_NODE_FD}, "AnimCurveNode::FocusDistance", "" {{\n'
            '\t\tProperties70:  {\n'
            f'\t\t\tP: "d|FocusDistance","Number","","A",{fd[0]:.6f}\n'
            '\t\t}\n'
            '\t}\n'
        )

        # AnimationCurves
        crv(f, ID_CRV_TX, "Tx",           tx)
        crv(f, ID_CRV_TY, "Ty",           ty)
        crv(f, ID_CRV_TZ, "Tz",           tz)
        crv(f, ID_CRV_RX, "Rx",           rx)
        crv(f, ID_CRV_RY, "Ry",           ry)
        crv(f, ID_CRV_RZ, "Rz",           rz)
        crv(f, ID_CRV_FL, "FocalLength",  fl)
        crv(f, ID_CRV_FD, "FocusDistance",fd)

        f.write("}\n\n")   # end Objects

        # ── Connections ────────────────────────────────────────
        f.write(
            "Connections:  {\n"
            # Model → root scene
            f'\tC: "OO",{ID_CAM},0\n'
            # NodeAttribute → Model (links camera data to transform node)
            f'\tC: "OO",{ID_ATTR},{ID_CAM}\n'
            # Layer → Stack
            f'\tC: "OO",{ID_LAYER},{ID_STACK}\n'
            # CurveNodes → Layer
            f'\tC: "OO",{ID_NODE_T},{ID_LAYER}\n'
            f'\tC: "OO",{ID_NODE_R},{ID_LAYER}\n'
            f'\tC: "OO",{ID_NODE_FL},{ID_LAYER}\n'
            f'\tC: "OO",{ID_NODE_FD},{ID_LAYER}\n'
            # Transform anim → Model
            f'\tC: "OP",{ID_NODE_T},{ID_CAM},"Lcl Translation"\n'
            f'\tC: "OP",{ID_NODE_R},{ID_CAM},"Lcl Rotation"\n'
            # Lens anim → NodeAttribute (KEY: not the Model!)
            f'\tC: "OP",{ID_NODE_FL},{ID_ATTR},"FocalLength"\n'
            f'\tC: "OP",{ID_NODE_FD},{ID_ATTR},"FocusDistance"\n'
            # Curves → CurveNode channels
            f'\tC: "OP",{ID_CRV_TX},{ID_NODE_T},"d|X"\n'
            f'\tC: "OP",{ID_CRV_TY},{ID_NODE_T},"d|Y"\n'
            f'\tC: "OP",{ID_CRV_TZ},{ID_NODE_T},"d|Z"\n'
            f'\tC: "OP",{ID_CRV_RX},{ID_NODE_R},"d|X"\n'
            f'\tC: "OP",{ID_CRV_RY},{ID_NODE_R},"d|Y"\n'
            f'\tC: "OP",{ID_CRV_RZ},{ID_NODE_R},"d|Z"\n'
            f'\tC: "OP",{ID_CRV_FL},{ID_NODE_FL},"d|FocalLength"\n'
            f'\tC: "OP",{ID_CRV_FD},{ID_NODE_FD},"d|FocusDistance"\n'
            "}\n"
        )

    print(f"  ✓ FBX ASCII written → {Path(path).name}")


# ═══════════════════════════════════════════════════════════════
#  FBX BINARY 7.4 WRITER  (pure Python, Blender-compatible)
# ═══════════════════════════════════════════════════════════════

def write_fbx_binary(frames: list, path: str, fps: float = 24.0) -> None:
    """
    Write a binary FBX 7.4 file with an animated camera.

    Pure Python — no Autodesk FBX SDK required.
    Compatible with Blender 2.80+, Maya 2018+, and all major DCCs.

    Binary FBX is the only FBX variant Blender accepts.

    Format reference:
      https://code.blender.org/2013/08/fbx-binary-file-format-specification/
    """
    import struct
    import os
    from io import BytesIO

    fps  = float(fps)
    nf   = len(frames)
    KTIME = 46186158000

    def kt(i):  return int(i * KTIME / fps)

    # ── Per-frame arrays ──────────────────────────────────────
    ktimes = [kt(i) for i in range(nf)]
    tx = [f.tx  * 100 for f in frames]
    ty = [f.ty  * 100 for f in frames]
    tz = [f.tz  * 100 for f in frames]
    rx = [f.tilt       for f in frames]
    ry = [f.pan        for f in frames]
    rz = [f.roll       for f in frames]
    fl = [f.focal_length          for f in frames]
    fd = [f.focus_distance * 100  for f in frames]

    dur  = ktimes[-1] + kt(1) if ktimes else 0
    f0   = frames[0]
    sw, sh = f0.sensor_width, f0.sensor_height
    sq     = f0.anamorphic_squeeze

    _tm = {120: 12, 100: 11, 60: 9, 50: 8, 48: 7, 30: 6, 25: 5, 24: 4}
    time_mode = _tm.get(int(round(fps)), 14)

    # ── Property serialisers ──────────────────────────────────
    def pI(v):  return b'I' + struct.pack('<i', int(v))
    def pL(v):  return b'L' + struct.pack('<q', int(v))
    def pD(v):  return b'D' + struct.pack('<d', float(v))
    def pF(v):  return b'F' + struct.pack('<f', float(v))
    def pC(v):  return b'C' + struct.pack('<B', 1 if v else 0)
    def pS(v):
        b = v.encode('utf-8') if isinstance(v, str) else bytes(v)
        return b'S' + struct.pack('<I', len(b)) + b
    def pR(v):  return b'R' + struct.pack('<I', len(v)) + bytes(v)

    def pAf(vals):
        d = struct.pack(f'<{len(vals)}f', *[float(x) for x in vals])
        return b'f' + struct.pack('<III', len(vals), 0, len(d)) + d
    def pAd(vals):
        d = struct.pack(f'<{len(vals)}d', *[float(x) for x in vals])
        return b'd' + struct.pack('<III', len(vals), 0, len(d)) + d
    def pAi(vals):
        d = struct.pack(f'<{len(vals)}i', *[int(x) for x in vals])
        return b'i' + struct.pack('<III', len(vals), 0, len(d)) + d
    def pAl(vals):
        d = struct.pack(f'<{len(vals)}q', *[int(x) for x in vals])
        return b'l' + struct.pack('<III', len(vals), 0, len(d)) + d

    # ── Node class ────────────────────────────────────────────
    class Node:
        """One binary FBX node."""
        _SENTINEL = b'\x00' * 13

        def __init__(self, name, props=(), children=(), ch=None):
            self.nb = name.encode('ascii') if isinstance(name, str) else name
            self.pb = list(props)        # list[bytes] – one entry per property
            self.ch = list(ch if ch is not None else children)

        @property
        def pd(self):
            return b''.join(self.pb)

        def size(self):
            # header = EndOffset(4)+NumProps(4)+PropListLen(4)+NameLen(1)+Name
            h = 13 + len(self.nb)
            p = len(self.pd)
            c = sum(x.size() for x in self.ch) + (13 if self.ch else 0)
            return h + p + c

        def write(self, bio: BytesIO, base: int):
            pd = self.pd
            bio.write(struct.pack('<I', base + self.size()))  # EndOffset
            bio.write(struct.pack('<I', len(self.pb)))         # NumProperties
            bio.write(struct.pack('<I', len(pd)))              # PropertyListLen
            bio.write(struct.pack('<B', len(self.nb)))         # NameLen
            bio.write(self.nb)
            bio.write(pd)
            if self.ch:
                cbase = base + 13 + len(self.nb) + len(pd)
                for child in self.ch:
                    child.write(bio, cbase)
                    cbase += child.size()
                bio.write(self._SENTINEL)

    # ── Convenience constructors ──────────────────────────────
    def N(name, *props, ch=()):
        return Node(name, list(props), list(ch))

    # FBX binary uses shortened class aliases inside object names,
    # NOT the full node-type name.  Blender's importer enforces these
    # exact strings via elem_name_ensure_class().
    _CLASS_ALIAS = {
        'AnimationStack':     'AnimStack',
        'AnimationLayer':     'AnimLayer',
        'AnimationCurveNode': 'AnimCurveNode',
        'AnimationCurve':     'AnimCurve',
    }

    def oname(n, t):
        """FBX binary object name: "Name\0\x01ClassAlias"."""
        alias = _CLASS_ALIAS.get(t, t)
        return f"{n}\x00\x01{alias}"

    def P70(*entries):
        """Properties70 node. Each entry is a Node('P', ...)."""
        return Node('Properties70', children=list(entries))

    def P(name, type_, label, flags, *vals):
        """One P-node inside Properties70."""
        ps = [pS(name), pS(type_), pS(label), pS(flags)]
        for v in vals:
            if type_ in ('KTime',):
                ps.append(pL(int(v)))
            elif isinstance(v, bool):
                ps.append(pI(int(v)))
            elif isinstance(v, int):
                ps.append(pI(v))
            elif isinstance(v, float):
                ps.append(pD(v))
            elif isinstance(v, str):
                ps.append(pS(v))
        return Node('P', ps)

    # ── AnimationCurveNode ────────────────────────────────────
    def curve_node(nid, label, **channels):
        ch_nodes = [P(f'd|{k}', 'Number', '', 'A', float(v))
                    for k, v in channels.items()]
        return Node('AnimationCurveNode',
            [pL(nid), pS(oname(label, 'AnimationCurveNode')), pS('')],
            ch=[P70(*ch_nodes)] if ch_nodes else [])

    # ── AnimationCurve ────────────────────────────────────────
    def anim_curve(cid, label, times, values):
        nk = len(values)
        return Node('AnimationCurve',
            [pL(cid), pS(oname(label, 'AnimationCurve')), pS('')],
            ch=[
                N('Default',          pD(float(values[0]))),
                N('KeyVer',           pI(4008)),
                N('KeyCount',         pI(nk)),
                Node('KeyTime',          [pAl(times)]),
                Node('KeyValueFloat',    [pAf(values)]),
                Node('KeyAttrFlags',     [pAi([24836])]),          # linear
                Node('KeyAttrDataFloat', [pAf([0.0, 0.0, 0.0, 0.0])]),
                Node('KeyAttrRefCount',  [pAi([nk])]),
            ])

    # ── Connection helpers ────────────────────────────────────
    def C_OO(child, parent):
        return Node('C', [pS('OO'), pL(child), pL(parent)])
    def C_OP(child, parent, prop):
        return Node('C', [pS('OP'), pL(child), pL(parent), pS(prop)])

    # ── Object IDs ───────────────────────────────────────────
    # Blender's FBX camera uses TWO separate objects:
    #   NodeAttribute  – holds all camera/lens data (FocalLength etc.)
    #   Model          – holds transforms (Lcl Translation/Rotation)
    # Lens animation MUST connect to the NodeAttribute, NOT the Model.
    # Transform animation connects to the Model as usual.
    ID_ATTR = 2001   # NodeAttribute (camera lens data)
    ID_CAM  = 2002   # Model (camera transform)
    ID_STK  = 3001;  ID_LYR  = 3002
    ID_NT   = 4001;  ID_NR   = 4002;  ID_NFL = 4003;  ID_NFD = 4004
    ID_TX   = 5001;  ID_TY   = 5002;  ID_TZ  = 5003
    ID_RX   = 5004;  ID_RY   = 5005;  ID_RZ  = 5006
    ID_FL   = 5007;  ID_FD   = 5008

    # ─────────────────────────────────────────────────────────
    #  Build node tree
    # ─────────────────────────────────────────────────────────
    now = datetime.now()

    header_ext = Node('FBXHeaderExtension', ch=[
        N('FBXHeaderVersion', pI(1003)),
        N('FBXVersion',       pI(7400)),
        Node('CreationTimeStamp', ch=[
            N('Version',     pI(1000)),
            N('Year',        pI(now.year)),
            N('Month',       pI(now.month)),
            N('Day',         pI(now.day)),
            N('Hour',        pI(now.hour)),
            N('Minute',      pI(now.minute)),
            N('Second',      pI(now.second)),
            N('Millisecond', pI(0)),
        ]),
        N('Creator', pS('OpenTrackIO to FBX Converter')),
    ])

    global_settings = Node('GlobalSettings', ch=[
        N('Version', pI(1000)),
        P70(
            P('UpAxis',              'int',    'Integer', '', 1),
            P('UpAxisSign',          'int',    'Integer', '', 1),
            P('FrontAxis',           'int',    'Integer', '', 2),
            P('FrontAxisSign',       'int',    'Integer', '', 1),
            P('CoordAxis',           'int',    'Integer', '', 0),
            P('CoordAxisSign',       'int',    'Integer', '', 1),
            P('UnitScaleFactor',     'double', 'Number',  '', 1.0),
            P('TimeMode',            'enum',   '',        '', time_mode),
            P('CustomFrameRate',     'double', 'Number',  '', fps),
            P('TimeSpanStart',       'KTime',  'Time',    '', 0),
            P('TimeSpanStop',        'KTime',  'Time',    '', dur),
        )
    ])

    documents = Node('Documents', ch=[
        N('Count', pI(1)),
        Node('Document', [pL(1000), pS('Scene'), pS('Scene')], ch=[
            N('RootNode', pL(0)),
        ]),
    ])

    definitions = Node('Definitions', ch=[
        N('Version', pI(100)),
        Node('ObjectType', [pS('NodeAttribute')],      ch=[N('Count', pI(1))]),
        Node('ObjectType', [pS('Model')],              ch=[N('Count', pI(1))]),
        Node('ObjectType', [pS('AnimationStack')],     ch=[N('Count', pI(1))]),
        Node('ObjectType', [pS('AnimationLayer')],     ch=[N('Count', pI(1))]),
        Node('ObjectType', [pS('AnimationCurveNode')], ch=[N('Count', pI(4))]),
        Node('ObjectType', [pS('AnimationCurve')],     ch=[N('Count', pI(8))]),
    ])

    # Helper: compute horizontal FOV in degrees from focal length and sensor width
    def _fov(focal_mm):
        import math as _math
        return 2.0 * _math.degrees(_math.atan((sw / 25.4) / (2.0 * focal_mm / 25.4)))

    # ── NodeAttribute: Camera (holds ALL lens/optics data) ────────
    # Blender's importer reads FocalLength/FocusDistance from this
    # node, and expects animation curves to connect HERE (not the Model).
    cam_attr = Node('NodeAttribute',
        [pL(ID_ATTR), pS(oname('Camera', 'NodeAttribute')), pS('Camera')],
        ch=[
            P70(
                P('FilmWidth',       'double', 'Number', '',  sw / 25.4),
                P('FilmHeight',      'double', 'Number', '',  sh / 25.4),
                P('FilmAspectRatio', 'double', 'Number', '',  sw / sh),
                P('FilmOffsetX',     'double', 'Number', 'A', 0.0),
                P('FilmOffsetY',     'double', 'Number', 'A', 0.0),
                P('ApertureMode',    'enum',   '',       '',  3),
                P('GateFit',         'enum',   '',       '',  2),
                P('FieldOfView',     'FieldOfView',  '', 'A', _fov(fl[0])),
                P('FieldOfViewX',    'FieldOfViewX', '', 'A', _fov(fl[0])),
                P('FieldOfViewY',    'FieldOfViewY', '', 'A',
                  2.0 * __import__('math').degrees(
                      __import__('math').atan((sh / 25.4) / (2.0 * fl[0] / 25.4)))),
                P('FocalLength',     'double', 'Number', 'A', float(fl[0])),
                P('FocusDistance',   'double', 'Number', 'A', float(fd[0])),
                P('NearPlane',       'double', 'Number', '',  10.0),
                P('FarPlane',        'double', 'Number', '',  100000.0),
            ),
            N('TypeFlags',       pS('Camera')),
            N('GeometryVersion', pI(124)),
            Node('Position', [pD(0.0), pD(0.0), pD(0.0)]),
            Node('Up',       [pD(0.0), pD(1.0), pD(0.0)]),
            Node('LookAt',   [pD(0.0), pD(0.0), pD(-1.0)]),
        ])

    # ── Model: Camera (holds transform; links to NodeAttribute) ───
    # DefaultAttributeIndex = 0 tells Blender this Model uses the
    # first (and only) NodeAttribute connected to it.
    cam_model = Node('Model',
        [pL(ID_CAM), pS(oname('Camera', 'Model')), pS('Camera')],
        ch=[
            N('Version', pI(232)),
            P70(
                P('Lcl Translation', 'Lcl Translation', '', 'A',
                  float(tx[0]), float(ty[0]), float(tz[0])),
                P('Lcl Rotation',    'Lcl Rotation',    '', 'A',
                  float(rx[0]), float(ry[0]), float(rz[0])),
                P('Lcl Scaling',     'Lcl Scaling',     '', 'A',
                  1.0, 1.0, 1.0),
                P('RotationOrder',         'enum',    '', '', 0),
                P('RotationActive',        'bool',    '', '', 1),
                P('DefaultAttributeIndex', 'int', 'Integer', '', 0),
            ),
            N('MultiLayer', pI(0)),
            N('MultiTake',  pI(0)),
            N('Shading',    pC(True)),
            N('Culling',    pS('CullingOff')),
        ])

    # ── Animation ─────────────────────────────────────────────
    anim_stack = Node('AnimationStack',
        [pL(ID_STK), pS(oname('Take 001', 'AnimationStack')), pS('')],
        ch=[
            N('Version', pI(100)),
            P70(
                P('LocalStart',     'KTime', 'Time', '', 0),
                P('LocalStop',      'KTime', 'Time', '', dur),
                P('ReferenceStart', 'KTime', 'Time', '', 0),
                P('ReferenceStop',  'KTime', 'Time', '', dur),
            )
        ])

    anim_layer = Node('AnimationLayer',
        [pL(ID_LYR), pS(oname('BaseLayer', 'AnimationLayer')), pS('')],
        ch=[
            N('Version', pI(100)),
            P70(
                P('Weight', 'Number', '', 'A', 100.0),
                P('Mute',   'bool',   '', '', 0),
                P('Solo',   'bool',   '', '', 0),
                P('Lock',   'bool',   '', '', 0),
            )
        ])

    cn_T  = curve_node(ID_NT,  'T',             X=tx[0], Y=ty[0], Z=tz[0])
    cn_R  = curve_node(ID_NR,  'R',             X=rx[0], Y=ry[0], Z=rz[0])
    cn_FL = curve_node(ID_NFL, 'FocalLength',   FocalLength=fl[0])
    cn_FD = curve_node(ID_NFD, 'FocusDistance', FocusDistance=fd[0])

    crv_tx = anim_curve(ID_TX, 'Tx',           ktimes, tx)
    crv_ty = anim_curve(ID_TY, 'Ty',           ktimes, ty)
    crv_tz = anim_curve(ID_TZ, 'Tz',           ktimes, tz)
    crv_rx = anim_curve(ID_RX, 'Rx',           ktimes, rx)
    crv_ry = anim_curve(ID_RY, 'Ry',           ktimes, ry)
    crv_rz = anim_curve(ID_RZ, 'Rz',           ktimes, rz)
    crv_fl = anim_curve(ID_FL, 'FocalLength',  ktimes, fl)
    crv_fd = anim_curve(ID_FD, 'FocusDistance',ktimes, fd)

    objects = Node('Objects', ch=[
        cam_attr, cam_model,
        anim_stack, anim_layer,
        cn_T, cn_R, cn_FL, cn_FD,
        crv_tx, crv_ty, crv_tz,
        crv_rx, crv_ry, crv_rz,
        crv_fl, crv_fd,
    ])

    connections = Node('Connections', ch=[
        C_OO(ID_CAM,  0),           # Model → scene root
        C_OO(ID_ATTR, ID_CAM),      # NodeAttribute → Model (links camera data)
        C_OO(ID_LYR,  ID_STK),      # Layer → Stack
        C_OO(ID_NT,   ID_LYR),      # CurveNodes → Layer
        C_OO(ID_NR,   ID_LYR),
        C_OO(ID_NFL,  ID_LYR),
        C_OO(ID_NFD,  ID_LYR),
        C_OP(ID_NT,   ID_CAM,  'Lcl Translation'),   # T → Model
        C_OP(ID_NR,   ID_CAM,  'Lcl Rotation'),      # R → Model
        C_OP(ID_NFL,  ID_ATTR, 'FocalLength'),        # FL → NodeAttribute ← KEY
        C_OP(ID_NFD,  ID_ATTR, 'FocusDistance'),      # FD → NodeAttribute ← KEY
        C_OP(ID_TX,   ID_NT,   'd|X'),
        C_OP(ID_TY,   ID_NT,   'd|Y'),
        C_OP(ID_TZ,   ID_NT,   'd|Z'),
        C_OP(ID_RX,   ID_NR,   'd|X'),
        C_OP(ID_RY,   ID_NR,   'd|Y'),
        C_OP(ID_RZ,   ID_NR,   'd|Z'),
        C_OP(ID_FL,   ID_NFL,  'd|FocalLength'),
        C_OP(ID_FD,   ID_NFD,  'd|FocusDistance'),
    ])

    takes = Node('Takes', ch=[N('Current', pS('Take 001'))])

    top_nodes = [
        header_ext,
        N('FileId',       pR(os.urandom(16))),
        N('CreationTime', pS(now.strftime('%Y-%m-%d %H:%M:%S:%f'))),
        N('Creator',      pS('OpenTrackIO to FBX Converter')),
        global_settings,
        documents,
        Node('References'),
        definitions,
        objects,
        connections,
        takes,
    ]

    # ─────────────────────────────────────────────────────────
    #  Serialise
    # ─────────────────────────────────────────────────────────
    MAGIC = b"Kaydara FBX Binary  \0\x1a\x00"   # 23 bytes
    bio   = BytesIO()
    bio.write(MAGIC)
    bio.write(struct.pack('<I', 7400))  # version

    pos = 27   # 23 magic + 4 version
    for tn in top_nodes:
        tn.write(bio, pos)
        pos += tn.size()

    bio.write(b'\x00' * 13)   # final null record

    with open(path, 'wb') as fh:
        fh.write(bio.getvalue())

    print(f"  ✓ FBX binary written → {Path(path).name}")


# ═══════════════════════════════════════════════════════════════
#  STATISTICS HELPERS
# ═══════════════════════════════════════════════════════════════

def stat_row(label, values, unit="", fmt=".3f"):
    """Return a 5-cell table row: label, min, max, mean, unit."""
    arr = [v for v in values if v is not None]
    if not arr:
        return [label, "–", "–", "–", unit]
    lo, hi, mu = min(arr), max(arr), sum(arr) / len(arr)
    f = f"{{:{fmt}}}"
    return [label, f.format(lo), f.format(hi), f.format(mu), unit]


def _collect_stats(frames):
    """Build statistics dict from frame list."""
    return {
        "tx": [f.tx for f in frames],
        "ty": [f.ty for f in frames],
        "tz": [f.tz for f in frames],
        "pan":  [f.pan  for f in frames],
        "tilt": [f.tilt for f in frames],
        "roll": [f.roll for f in frames],
        "focal_length":   [f.focal_length   for f in frames],
        "focus_distance": [f.focus_distance for f in frames],
        "f_stop": [f.f_stop for f in frames if f.f_stop is not None],
    }


# ═══════════════════════════════════════════════════════════════
#  MATPLOTLIB CHARTS  (saved to temp PNGs, embedded in PDF)
# ═══════════════════════════════════════════════════════════════

_STYLE = {
    "figure.facecolor":  "#0f1117",
    "axes.facecolor":    "#1a1d27",
    "axes.edgecolor":    "#3a3d4e",
    "axes.labelcolor":   "#c8ccd8",
    "xtick.color":       "#8890a4",
    "ytick.color":       "#8890a4",
    "text.color":        "#c8ccd8",
    "grid.color":        "#2a2d3e",
    "grid.linestyle":    "--",
    "grid.alpha":        0.6,
    "legend.facecolor":  "#1a1d27",
    "legend.edgecolor":  "#3a3d4e",
    "lines.linewidth":   1.6,
    "font.size":         9,
}

_PAL = ["#5b9dff", "#ff7c5b", "#5bffb8", "#ffda5b", "#c45bff"]


def _apply_style(ax):
    ax.grid(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def chart_position(frames, tmp_dir) -> str:
    frames_idx = list(range(len(frames)))
    with plt.style.context(_STYLE):
        fig, axes = plt.subplots(3, 1, figsize=(9, 6), sharex=True)
        fig.suptitle("Camera Position over Time", fontsize=11, y=1.0)
        for ax, attr, label, color in zip(
            axes,
            ["tx", "ty", "tz"],
            ["X (m)", "Y (m)", "Z (m)"],
            _PAL,
        ):
            ax.plot(frames_idx, [getattr(f, attr) for f in frames], color=color)
            ax.set_ylabel(label)
            _apply_style(ax)
        axes[-1].set_xlabel("Frame")
        fig.tight_layout()
        out = os.path.join(tmp_dir, "chart_pos.png")
        fig.savefig(out, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
    return out


def chart_rotation(frames, tmp_dir) -> str:
    frames_idx = list(range(len(frames)))
    with plt.style.context(_STYLE):
        fig, axes = plt.subplots(3, 1, figsize=(9, 6), sharex=True)
        fig.suptitle("Camera Rotation over Time", fontsize=11, y=1.0)
        for ax, attr, label, color in zip(
            axes,
            ["pan", "tilt", "roll"],
            ["Pan (°)", "Tilt (°)", "Roll (°)"],
            _PAL[1:],
        ):
            ax.plot(frames_idx, [getattr(f, attr) for f in frames], color=color)
            ax.set_ylabel(label)
            _apply_style(ax)
        axes[-1].set_xlabel("Frame")
        fig.tight_layout()
        out = os.path.join(tmp_dir, "chart_rot.png")
        fig.savefig(out, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
    return out


def chart_lens(frames, tmp_dir) -> str:
    frames_idx = list(range(len(frames)))
    fl = [f.focal_length   for f in frames]
    fd = [f.focus_distance for f in frames]
    fs = [f.f_stop for f in frames]
    has_fstop = any(v is not None for v in fs)
    rows = 3 if has_fstop else 2

    with plt.style.context(_STYLE):
        fig, axes = plt.subplots(rows, 1, figsize=(9, rows * 2.1), sharex=True)
        if rows == 2:
            axes = list(axes)
        fig.suptitle("Lens Parameters over Time", fontsize=11, y=1.0)

        axes[0].plot(frames_idx, fl, color=_PAL[0])
        axes[0].set_ylabel("Focal Length (mm)")
        _apply_style(axes[0])

        axes[1].plot(frames_idx, fd, color=_PAL[2])
        axes[1].set_ylabel("Focus Distance (m)")
        _apply_style(axes[1])

        if has_fstop and rows == 3:
            fs_filled = [v if v is not None else float("nan") for v in fs]
            axes[2].plot(frames_idx, fs_filled, color=_PAL[3])
            axes[2].set_ylabel("F-stop")
            _apply_style(axes[2])
            axes[2].set_xlabel("Frame")
        else:
            axes[-1].set_xlabel("Frame")

        fig.tight_layout()
        out = os.path.join(tmp_dir, "chart_lens.png")
        fig.savefig(out, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
    return out


def chart_path_topdown(frames, tmp_dir) -> str:
    xs = [f.tx for f in frames]
    zs = [f.tz for f in frames]
    n  = len(frames)

    with plt.style.context(_STYLE):
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.set_title("Camera Path – Top-Down View (X/Z plane)", fontsize=11)

        # Colour-map path by frame progression
        for i in range(n - 1):
            t = i / max(n - 1, 1)
            c = plt.cm.cool(t)
            ax.plot([xs[i], xs[i+1]], [zs[i], zs[i+1]], color=c, lw=1.5)

        # Start / end markers
        ax.scatter([xs[0]],  [zs[0]],  color="#5bffb8", s=80, zorder=5, label="Start")
        ax.scatter([xs[-1]], [zs[-1]], color="#ff7c5b", s=80, zorder=5, label="End")

        # Arrow indicating look direction at first frame
        pan_rad = math.radians(frames[0].pan)
        ax.annotate(
            "", xy=(xs[0] + 0.5 * math.sin(pan_rad),
                    zs[0] - 0.5 * math.cos(pan_rad)),
            xytext=(xs[0], zs[0]),
            arrowprops=dict(arrowstyle="->", color="#ffda5b", lw=1.5),
        )

        sm = plt.cm.ScalarMappable(cmap="cool",
                                   norm=plt.Normalize(0, max(n-1, 1)))
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
        cb.set_label("Frame", color="#c8ccd8")
        cb.ax.yaxis.set_tick_params(color="#8890a4")
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="#8890a4")

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")
        ax.legend(facecolor="#1a1d27", edgecolor="#3a3d4e",
                  labelcolor="#c8ccd8", fontsize=8)
        ax.set_aspect("equal", adjustable="datalim")
        _apply_style(ax)
        fig.tight_layout()
        out = os.path.join(tmp_dir, "chart_path.png")
        fig.savefig(out, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
    return out


# ═══════════════════════════════════════════════════════════════
#  PDF REPORT BUILDER
# ═══════════════════════════════════════════════════════════════

# Colours
C_DARK   = colors.HexColor("#0f1117")
C_PANEL  = colors.HexColor("#1a1d27")
C_BORDER = colors.HexColor("#2a3050")
C_ACCENT = colors.HexColor("#5b9dff")
C_TEXT   = colors.HexColor("#c8ccd8")
C_MUTED  = colors.HexColor("#6b7280")
C_GREEN  = colors.HexColor("#5bffb8")
C_HEAD_ROW = colors.HexColor("#1e2235")
C_ALT_ROW  = colors.HexColor("#151825")
C_WHITE    = colors.white


def _styles():
    ss = getSampleStyleSheet()
    base = dict(fontName="Helvetica", textColor=C_TEXT, backColor=C_DARK)
    styles = {
        "title": ParagraphStyle("OT_Title", parent=ss["Title"],
            fontSize=26, leading=30, textColor=C_WHITE,
            fontName="Helvetica-Bold", alignment=TA_LEFT),
        "sub":   ParagraphStyle("OT_Sub", parent=ss["Normal"],
            fontSize=12, textColor=C_ACCENT, fontName="Helvetica",
            spaceAfter=4, alignment=TA_LEFT),
        "h2":    ParagraphStyle("OT_H2", parent=ss["Heading2"],
            fontSize=13, textColor=C_WHITE, fontName="Helvetica-Bold",
            spaceBefore=14, spaceAfter=4),
        "body":  ParagraphStyle("OT_Body", parent=ss["Normal"],
            fontSize=9, textColor=C_TEXT, fontName="Helvetica",
            leading=14, spaceAfter=4),
        "small": ParagraphStyle("OT_Small", parent=ss["Normal"],
            fontSize=8, textColor=C_MUTED, fontName="Helvetica"),
        "th":    ParagraphStyle("OT_TH", parent=ss["Normal"],
            fontSize=8, textColor=C_WHITE, fontName="Helvetica-Bold",
            alignment=TA_CENTER),
        "td":    ParagraphStyle("OT_TD", parent=ss["Normal"],
            fontSize=8, textColor=C_TEXT, fontName="Helvetica",
            alignment=TA_CENTER),
        "td_l":  ParagraphStyle("OT_TDL", parent=ss["Normal"],
            fontSize=8, textColor=C_TEXT, fontName="Helvetica",
            alignment=TA_LEFT),
    }
    return styles


def _table_style(n_rows, header=True):
    cmds = [
        ("BACKGROUND",  (0, 0), (-1, 0 if not header else 0), C_HEAD_ROW),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_PANEL, C_ALT_ROW]),
        ("TEXTCOLOR",   (0, 0), (-1, -1),   C_TEXT),
        ("FONTNAME",    (0, 0), (-1, 0),     "Helvetica-Bold"),
        ("FONTNAME",    (0, 1), (-1, -1),    "Helvetica"),
        ("FONTSIZE",    (0, 0), (-1, -1),    8),
        ("INNERGRID",   (0, 0), (-1, -1),    0.4, C_BORDER),
        ("BOX",         (0, 0), (-1, -1),    0.8, C_ACCENT),
        ("TOPPADDING",  (0, 0), (-1, -1),    5),
        ("BOTTOMPADDING",(0, 0),(-1, -1),    5),
        ("LEFTPADDING", (0, 0), (-1, -1),    6),
        ("RIGHTPADDING",(0, 0), (-1, -1),    6),
        ("VALIGN",      (0, 0), (-1, -1),    "MIDDLE"),
    ]
    return TableStyle(cmds)


def build_pdf(frames: list, output_path: str, shot_name: str = "Shot") -> None:
    """Generate the PDF camera report."""
    styles = _styles()
    stats  = _collect_stats(frames)
    f0     = frames[0]
    n      = len(frames)
    fps    = f0.fps
    dur_s  = n / fps if fps else 0

    tmp_dir = tempfile.mkdtemp()

    # ── Generate charts ───────────────────────────────────────
    print("  Generating charts…")
    p_pos  = chart_position(frames,  tmp_dir)
    p_rot  = chart_rotation(frames,  tmp_dir)
    p_lens = chart_lens(frames,      tmp_dir)
    p_path = chart_path_topdown(frames, tmp_dir)

    # ── ReportLab document ────────────────────────────────────
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=1.8*cm, rightMargin=1.8*cm,
        topMargin=2*cm,    bottomMargin=2*cm,
    )
    PAGE_W = A4[0] - 3.6*cm
    story  = []
    S      = styles

    def img(path, width=None, height=None):
        w = width or PAGE_W
        im = Image(path, width=w, height=height or w * 0.55)
        im.hAlign = "LEFT"
        return im

    def hr():
        return HRFlowable(width="100%", thickness=0.5,
                          color=C_BORDER, spaceAfter=8, spaceBefore=8)

    # ─────────────────────────────────────────────────────────
    #  PAGE 1 – Cover / Shot Overview
    # ─────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph("OpenTrackIO", S["sub"]))
    story.append(Paragraph("Camera Report", S["title"]))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(shot_name, S["sub"]))
    story.append(hr())
    story.append(Spacer(1, 0.3*cm))

    # Shot overview table
    tc_start = frames[0].timecode  or "–"
    tc_end   = frames[-1].timecode or "–"
    cam_label  = f0.camera_label  or "–"
    cam_make   = f0.camera_make   or "–"
    cam_model  = f0.camera_model  or "–"
    cam_serial = f0.camera_serial or "–"
    nfl = f"{f0.nominal_focal_length:.0f} mm" if f0.nominal_focal_length else "–"

    overview_data = [
        [Paragraph("Parameter", S["th"]), Paragraph("Value", S["th"])],
        [Paragraph("Total frames",     S["td_l"]), Paragraph(str(n),        S["td"])],
        [Paragraph("Frame rate",       S["td_l"]), Paragraph(f"{fps:.3f} fps", S["td"])],
        [Paragraph("Duration",         S["td_l"]), Paragraph(f"{dur_s:.2f} s", S["td"])],
        [Paragraph("Start timecode",   S["td_l"]), Paragraph(tc_start,      S["td"])],
        [Paragraph("End timecode",     S["td_l"]), Paragraph(tc_end,        S["td"])],
        [Paragraph("Camera label",     S["td_l"]), Paragraph(cam_label,     S["td"])],
        [Paragraph("Camera make",      S["td_l"]), Paragraph(cam_make,      S["td"])],
        [Paragraph("Camera model",     S["td_l"]), Paragraph(cam_model,     S["td"])],
        [Paragraph("Serial number",    S["td_l"]), Paragraph(cam_serial,    S["td"])],
        [Paragraph("Nominal focal length", S["td_l"]), Paragraph(nfl,       S["td"])],
    ]
    t = Table(overview_data, colWidths=[PAGE_W*0.45, PAGE_W*0.55])
    t.setStyle(_table_style(len(overview_data)))
    story.append(Paragraph("Shot Overview", S["h2"]))
    story.append(t)

    # ─────────────────────────────────────────────────────────
    #  PAGE 1 continued – Lens & Sensor Metadata
    # ─────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.6*cm))
    story.append(Paragraph("Lens &amp; Sensor Metadata", S["h2"]))

    sw   = f0.sensor_width
    sh   = f0.sensor_height
    spw  = f0.sensor_pixel_width
    sph  = f0.sensor_pixel_height
    sq   = f0.anamorphic_squeeze
    epo  = f0.entrance_pupil_offset

    sensor_str = f"{sw:.2f} × {sh:.2f} mm"
    if spw and sph:
        sensor_str += f"  ({int(spw)} × {int(sph)} px)"
    epo_str  = f"{epo:.2f} mm" if epo else "–"

    meta_data = [
        [Paragraph("Parameter", S["th"]), Paragraph("Value", S["th"])],
        [Paragraph("Sensor dimensions",        S["td_l"]), Paragraph(sensor_str,          S["td"])],
        [Paragraph("Anamorphic squeeze",        S["td_l"]), Paragraph(f"{sq:.3f}×",       S["td"])],
        [Paragraph("Entrance pupil offset",     S["td_l"]), Paragraph(epo_str,            S["td"])],
    ]
    tm = Table(meta_data, colWidths=[PAGE_W*0.45, PAGE_W*0.55])
    tm.setStyle(_table_style(len(meta_data)))
    story.append(tm)

    # ─────────────────────────────────────────────────────────
    #  PAGE 2 – Summary Statistics
    # ─────────────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("Summary Statistics", S["h2"]))
    story.append(Paragraph(
        "Min, max and mean values across all recorded frames.",
        S["body"],
    ))
    story.append(Spacer(1, 0.3*cm))

    hdr = [Paragraph(h, S["th"]) for h in
           ["Parameter", "Min", "Max", "Mean", "Unit"]]

    def prow(row):
        return [Paragraph(str(c), S["td_l"] if i == 0 else S["td"])
                for i, c in enumerate(row)]

    stat_rows = [
        stat_row("Translation X",     stats["tx"],            "m"),
        stat_row("Translation Y",     stats["ty"],            "m"),
        stat_row("Translation Z",     stats["tz"],            "m"),
        stat_row("Pan",               stats["pan"],           "°", ".2f"),
        stat_row("Tilt",              stats["tilt"],          "°", ".2f"),
        stat_row("Roll",              stats["roll"],          "°", ".2f"),
        stat_row("Focal Length",      stats["focal_length"],  "mm", ".2f"),
        stat_row("Focus Distance",    stats["focus_distance"],"m",  ".3f"),
        stat_row("F-stop",            stats["f_stop"],        "f/", ".2f"),
    ]

    table_data = [hdr] + [prow(r) for r in stat_rows]
    cw = [PAGE_W*0.32, PAGE_W*0.17, PAGE_W*0.17, PAGE_W*0.17, PAGE_W*0.17]
    ts = Table(table_data, colWidths=cw)
    ts.setStyle(_table_style(len(table_data)))
    story.append(ts)

    # ─────────────────────────────────────────────────────────
    #  PAGE 3 – Camera Position & Rotation Charts
    # ─────────────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("Camera Movement", S["h2"]))
    story.append(img(p_pos,  height=PAGE_W * 0.62))
    story.append(Spacer(1, 0.4*cm))
    story.append(img(p_rot,  height=PAGE_W * 0.62))

    # ─────────────────────────────────────────────────────────
    #  PAGE 4 – Lens Charts + Top-Down Path
    # ─────────────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("Lens Parameters", S["h2"]))
    story.append(img(p_lens, height=PAGE_W * 0.65))
    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph("Camera Path – Top-Down View", S["h2"]))
    story.append(img(p_path, width=PAGE_W * 0.75,
                     height=PAGE_W * 0.65))

    # ─────────────────────────────────────────────────────────
    #  Build with dark background on every page
    # ─────────────────────────────────────────────────────────
    def on_page(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(C_DARK)
        canvas.rect(0, 0, A4[0], A4[1], fill=1, stroke=0)
        # Footer
        canvas.setFillColor(C_MUTED)
        canvas.setFont("Helvetica", 7)
        canvas.drawString(
            1.8*cm, 1.2*cm,
            f"OpenTrackIO Camera Report  ·  {shot_name}  ·  "
            f"Generated {datetime.now():%Y-%m-%d %H:%M}",
        )
        canvas.drawRightString(
            A4[0] - 1.8*cm, 1.2*cm,
            f"Page {doc.page}",
        )
        canvas.restoreState()

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    print(f"  ✓ PDF written  → {Path(output_path).name}")

    # Tidy temp files
    for fp in [p_pos, p_rot, p_lens, p_path]:
        try:
            os.remove(fp)
        except OSError:
            pass


# ═══════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════

def _folder_has_json(folder: Path) -> bool:
    """Return True if *folder* contains at least one .json file directly."""
    return any(folder.glob("*.json"))


def process_sequence(seq_folder: Path, output_dir: Path, args) -> bool:
    """
    Process a single sequence folder.

    Loads frames, writes FBX and/or PDF into *output_dir*.
    Returns True on success, False if no frames could be parsed.
    """
    shot_name = seq_folder.name

    print(f"\n{'─'*55}")
    print(f"  Sequence : {seq_folder.name}")
    print(f"  Output   : {output_dir}")
    print(f"{'─'*55}")

    try:
        frames = load_folder(str(seq_folder))
    except FileNotFoundError as exc:
        print(f"  ⚠ Skipping – {exc}")
        return False

    if not frames:
        print("  ⚠ Skipping – no frames could be parsed.")
        return False

    fps = args.fps or frames[0].fps or 24.0
    print(f"  Frame rate: {fps:.3f} fps")
    print(f"  Duration  : {len(frames)/fps:.2f} s  ({len(frames)} frames)")

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── FBX output ────────────────────────────────────────────
    if not args.no_fbx:
        fbx_path = output_dir / args.fbx_name
        print(f"\n  Writing FBX (binary)…")
        write_fbx_binary(frames, str(fbx_path), fps=fps)

    # ── PDF ───────────────────────────────────────────────────
    if not args.no_pdf:
        pdf_path = output_dir / args.pdf_name
        print(f"\n  Writing PDF…")
        build_pdf(frames, str(pdf_path), shot_name=shot_name)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert OpenTrackIO JSON frames to a 3D camera file + PDF report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input_folder",
        help="Sequence folder (single mode) or top-level folder of sequences (batch mode)")
    parser.add_argument("--fps", type=float, default=None,
        help="Override frame rate (default: auto-detect from data)")
    parser.add_argument("--output-dir", default=None,
        help="Output directory (default: same as input_folder / sequence folder)")
    parser.add_argument("--fbx-name", default="camera_animation.fbx",
        help="FBX output filename")
    parser.add_argument("--pdf-name", default="camera_report.pdf",
        help="PDF output filename")
    parser.add_argument("--shot-name", default=None,
        help="Shot label used in the PDF – single mode only (default: folder name)")
    parser.add_argument("--no-fbx", action="store_true",
        help="Skip FBX generation")
    parser.add_argument("--no-pdf", action="store_true",
        help="Skip PDF report generation")
    parser.add_argument("--batch", action="store_true",
        help="Process each sub-folder of input_folder as an independent sequence")

    args = parser.parse_args()

    input_folder = Path(args.input_folder).expanduser().resolve()
    if not input_folder.is_dir():
        print(f"ERROR: Not a directory: {input_folder}", file=sys.stderr)
        sys.exit(1)

    # ── Auto-detect batch mode ────────────────────────────────
    # Treat as batch if the folder has no .json files itself but its
    # immediate sub-folders do, OR if --batch was passed explicitly.
    seq_dirs = sorted(
        d for d in input_folder.iterdir()
        if d.is_dir() and _folder_has_json(d)
    )
    batch_mode = args.batch or (not _folder_has_json(input_folder) and bool(seq_dirs))

    print(f"\n{'═'*55}")
    print(f"  OpenTrackIO Converter")
    print(f"  Input  : {input_folder}")
    print(f"  Mode   : {'batch (%d sequences)' % len(seq_dirs) if batch_mode else 'single'}")
    if args.output_dir:
        print(f"  Output : {args.output_dir}")
    print(f"{'═'*55}")

    # ── Batch mode ────────────────────────────────────────────
    if batch_mode:
        if not seq_dirs:
            print("ERROR: No sub-folders containing .json files found.", file=sys.stderr)
            sys.exit(1)

        base_output = Path(args.output_dir).resolve() if args.output_dir else None
        ok = err = 0
        for seq_dir in seq_dirs:
            # Each sequence gets its own sub-folder inside the output root,
            # mirroring the input structure.
            out_dir = (base_output / seq_dir.name) if base_output else seq_dir
            success = process_sequence(seq_dir, out_dir, args)
            if success:
                ok += 1
            else:
                err += 1

        print(f"\n{'═'*55}")
        print(f"  Batch complete: {ok} succeeded, {err} failed.")
        print(f"{'═'*55}\n")
        if err and ok == 0:
            sys.exit(1)

    # ── Single mode ───────────────────────────────────────────
    else:
        out_dir = Path(args.output_dir).resolve() if args.output_dir else input_folder
        success = process_sequence(input_folder, out_dir, args)
        if not success:
            sys.exit(1)
        print(f"\n  Done.\n")


if __name__ == "__main__":
    main()
