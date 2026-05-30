import io
import html
import os
import base64
import re
import sqlite3
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import streamlit as st


DATABASE_TABLE = "labeled_interval_stats"
DATABASE_SEGMENTS_TABLE = "labeled_interval_segments"
APP_VERSION = "v0.8.8"
APP_VERSION_DATE = "2026-05-30"


def trapezoid_area(y, x):
    """
    NumPy 2.x removed the np.trapz alias; older environments may not have np.trapezoid.
    Keep AUC calculations compatible with both.
    """
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    return np.trapz(y, x)


# -----------------------------
# Philips Xper PW6 parsing utils
# -----------------------------

def ascii_strings(data: bytes, min_len: int = 2):
    return [
        (m.start(), m.group().decode("latin1", errors="ignore"))
        for m in re.finditer(rb"[ -~]{%d,}" % min_len, data)
    ]


def internal_strip_label(data: bytes):
    """
    Philips Xper PW6 often stores the displayed pressure channel label
    near byte ~6648. This is more reliable than WAV_002/004/etc because
    numbering differs by case/procedure.
    """
    labels = {"RA", "RV", "PA", "PW", "AO", "LV", "BP"}
    candidates = []
    for off, s in ascii_strings(data, min_len=2):
        if 6500 <= off <= 6900:
            ss = s.strip().upper()
            if ss in labels:
                candidates.append((off, ss))
    if candidates:
        return candidates[0][1]
    return None


def infer_label_from_name(filename: str):
    name = filename.upper()
    if "WAV_000" in name:
        return "ECG"
    fallback = {
        "WAV_002": "RA",
        "WAV_004": "RV",
        "WAV_006": "PCWP",
        "WAV_008": "PA",
        "WAV_010": "AO",
    }
    for key, value in fallback.items():
        if key in name:
            return value
    return Path(filename).stem


def resolve_strip_label(filename: str, data: bytes):
    if is_dedicated_ecg_file(filename):
        return "ECG"
    return internal_strip_label(data) or infer_label_from_name(filename)


def is_dedicated_ecg_file(filename: str) -> bool:
    return "WAV_000" in filename.upper()


def find_float_runs(data: bytes, align: int = 2, min_len: int = 100):
    """
    Xper PW6 waveform arrays in the uploaded examples were little-endian float32,
    aligned at data[2:].
    """
    usable = len(data) - ((len(data) - align) % 4)
    arr = np.frombuffer(data[align:usable], dtype="<f4")

    mask = (
        np.isfinite(arr)
        & (arr > -5000)
        & (arr < 5000)
        & (np.abs(arr) > 0.001)
    )

    runs = []
    start = None
    for i, ok in enumerate(mask):
        if ok and start is None:
            start = i
        if (not ok or i == len(mask) - 1) and start is not None:
            end = i if not ok else i + 1
            if end - start >= min_len:
                vals = arr[start:end].copy()
                runs.append(
                    {
                        "byte_start": align + 4 * start,
                        "byte_end": align + 4 * end,
                        "n": end - start,
                        "min": float(np.nanmin(vals)),
                        "max": float(np.nanmax(vals)),
                        "mean": float(np.nanmean(vals)),
                        "std": float(np.nanstd(vals)),
                        "p01": float(np.nanpercentile(vals, 1)),
                        "p05": float(np.nanpercentile(vals, 5)),
                        "p50": float(np.nanpercentile(vals, 50)),
                        "p95": float(np.nanpercentile(vals, 95)),
                        "p99": float(np.nanpercentile(vals, 99)),
                        "values": vals,
                    }
                )
            start = None

    return runs


def clean_ecg_mv(y):
    y = y.astype(float) / 1000.0
    y[np.abs(y) > 4] = np.nan
    return y


def split_sequential_ecg_leads(values: np.ndarray, lead_names: list[str], duration_s: float):
    lead_count = len(lead_names)
    n = len(values) // lead_count
    if n < 1000:
        return None

    usable = n * lead_count
    arr = values[:usable].reshape(lead_count, n)
    t = np.linspace(0, duration_s, n, endpoint=False)
    out = pd.DataFrame({"time_s": t})
    for i, lead_name in enumerate(lead_names):
        out[f"EKG_{lead_name}_mV"] = clean_ecg_mv(arr[i])
    return out


def extract_ecg_from_run0(data: bytes, duration_s: float = 7.0, layout: str = "3_lead"):
    """
    Extracts an ECG-like trace from run 0.
    For dedicated WAV_000 files, this may contain sequential ECG leads.
    For pressure strip files, run 0 is often an ECG strip trace useful as Lead II.
    """
    runs = find_float_runs(data, min_len=100)
    if not runs:
        return None

    run0 = runs[0]["values"].astype(float)

    if layout == "auto":
        # Some pressure strips store D I, D II, D III sequentially; others
        # store one ECG trace. Choose by run length so the time axis stays sane.
        if len(run0) >= 18000:
            layout = "6_lead"
        elif len(run0) >= 9000:
            layout = "3_lead"
        else:
            layout = "single"

    # Dedicated 3-lead ECG pattern: D I, D II, D III sequentially with equal length.
    if layout == "3_lead":
        split = split_sequential_ecg_leads(run0, ["DI", "DII", "DIII"], duration_s)
        if split is not None:
            return split

    # Dedicated 6-lead ECG pattern: 6 sequential leads with equal length.
    if layout == "6_lead":
        split = split_sequential_ecg_leads(run0, ["I", "II", "III", "aVR", "aVL", "aVF"], duration_s)
        if split is not None:
            return split

    # Otherwise return run0 as a single ECG-like trace.
    t = np.linspace(0, duration_s, len(run0), endpoint=False)
    y = clean_ecg_mv(run0)

    return pd.DataFrame({"time_s": t, "EKG_II_mV": y})


def pressure_candidate_from_run(run_values: np.ndarray, duration_s: float, broad_search: bool = False):
    """
    A run can sometimes contain 3 sequential traces. In those Xper files,
    segment 0 is the pressure waveform and the later segments are display/
    cursor/auxiliary traces. Return the best segment as mmHg-like pressure.
    """
    v = run_values.astype(float)

    primary_candidates = [v]

    # Candidate B: first of 3 sequential segments.
    if len(v) >= 3000:
        n3 = (len(v) // 3) * 3
        if n3 >= 3000:
            primary_candidates.append(v[:n3].reshape(3, -1)[0])

    # Candidate C: first of 2 sequential segments.
    if len(v) >= 3000:
        n2 = (len(v) // 2) * 2
        if n2 >= 3000:
            primary_candidates.append(v[:n2].reshape(2, -1)[0])

    scored = []
    expected_pressure_samples = max(1000.0, float(duration_s) * 500.0)

    def score_candidate(c, scale=1.0):
        c = c.astype(float) * scale
        if len(c) < 1000:
            return

        # Score on the real pressure portion and drop leading blank samples from
        # the returned trace. The report display starts at the visible waveform,
        # so keeping those blanks creates an artificial rightward pressure shift.
        idx = np.where(np.abs(c) > 0.5)[0]
        leading_blank = int(idx[0]) if len(idx) else 0
        c2 = c[leading_blank:] if len(idx) else c
        if len(c2) < 1000:
            return

        p01, p99 = np.nanpercentile(c2, [1, 99])
        p05, p95 = np.nanpercentile(c2, [5, 95])
        cmin, cmax, cstd = np.nanmin(c2), np.nanmax(c2), np.nanstd(c2)

        mmhg_like = (
            cmax <= 300
            and cmin >= -30
            and p99 <= 260
            and p01 >= -20
            and cstd <= 90
            and cmax > 8
        )

        if not mmhg_like:
            return

        dynamic = p95 - p05
        sample_count_penalty = abs(len(c2) - expected_pressure_samples) * 0.005
        score = dynamic + 0.002 * len(c2) - abs(np.nanmedian(c2)) * 0.01 - sample_count_penalty
        scored.append((score, c2.astype(float).copy()))

    for c in primary_candidates:
        score_candidate(c, scale=1.0)

    if broad_search and not scored:
        # Some Xper exports pack pressure into later equal-sized segments, and
        # some store pressure as centi-mmHg. Only use this broader search as a
        # fallback so already-working cases keep their original extraction path.
        for segment_count in range(2, 7):
            n = len(v) // segment_count
            if n < 1000:
                continue
            arr = v[: n * segment_count].reshape(segment_count, n)
            for segment in arr:
                score_candidate(segment, scale=1.0)
                score_candidate(segment, scale=0.01)

    if not scored:
        return None

    scored = sorted(scored, key=lambda x: x[0], reverse=True)
    return scored[0][1]


def extract_pressure_waveform(data: bytes, duration_s: float = 7.0):
    runs = find_float_runs(data, min_len=100)

    candidates = []
    for r in runs:
        c = pressure_candidate_from_run(r["values"], duration_s, broad_search=False)
        if c is not None:
            score = (np.nanpercentile(c, 95) - np.nanpercentile(c, 5)) + 0.001 * len(c)
            candidates.append((score, r["byte_start"], c))

    if not candidates:
        for r in runs:
            c = pressure_candidate_from_run(r["values"], duration_s, broad_search=True)
            if c is not None:
                score = (np.nanpercentile(c, 95) - np.nanpercentile(c, 5)) + 0.001 * len(c)
                candidates.append((score, r["byte_start"], c))

    if not candidates:
        return None, pd.DataFrame()

    candidates = sorted(candidates, key=lambda x: (x[0], x[1]), reverse=True)
    values = candidates[0][2].astype(float)

    t = np.linspace(0, duration_s, len(values), endpoint=False)
    return values, pd.DataFrame({"time_s": t, "pressure_mmHg": values})


def interpolate_to_grid(df: pd.DataFrame, ycol: str, grid: np.ndarray):
    x = df["time_s"].to_numpy(float)
    y = df[ycol].to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    keep = np.concatenate(([True], np.diff(x) > 0)) if len(x) > 1 else np.array([True] * len(x))
    x = x[keep]
    y = y[keep]
    if len(x) < 2:
        return np.full_like(grid, np.nan, dtype=float)
    out = np.interp(grid, x, y)
    out[(grid < x[0]) | (grid > x[-1])] = np.nan
    return out


def first_valid_time(df: pd.DataFrame, ycol: str) -> float:
    if df is None or df.empty or ycol not in df.columns:
        return np.nan
    valid_idx = np.where(np.isfinite(df[ycol].to_numpy(float)))[0]
    if not len(valid_idx):
        return np.nan
    return float(df["time_s"].iloc[int(valid_idx[0])])


def shift_aligned_values(grid: np.ndarray, values: np.ndarray, shift_s: float) -> np.ndarray:
    """
    Shift a signal on an existing time grid without extrapolating.
    Positive shift moves the signal later/right; negative shift moves it earlier/left.
    """
    y = np.asarray(values, dtype=float)
    if abs(float(shift_s)) < 1e-12:
        return y
    ok = np.isfinite(grid) & np.isfinite(y)
    if ok.sum() < 2:
        return np.full_like(grid, np.nan, dtype=float)
    source_x = grid[ok] + float(shift_s)
    source_y = y[ok]
    keep = np.concatenate(([True], np.diff(source_x) > 0)) if len(source_x) > 1 else np.array([True] * len(source_x))
    source_x = source_x[keep]
    source_y = source_y[keep]
    out = np.interp(grid, source_x, source_y)
    out[(grid < source_x[0]) | (grid > source_x[-1])] = np.nan
    return out


def calculate_stats(segment: pd.DataFrame, signal_cols):
    """
    General waveform stats + morphology-focused features.

    AUC definitions:
    - raw_auc_to_zero: area under the signal relative to y=0.
    - auc_above_start_baseline: area after subtracting y at interval start.
    - auc_above_horizontal_baseline: area after subtracting mean(start, end).
    - excess_auc_linear_baseline_net: area after subtracting the line connecting start and end.
    - excess_auc_linear_baseline_positive: only the positive part above the start-end baseline line.

    Composite morphology indices:
    - vwave_sharpness_index = peak_above_linear_baseline / fwhm_excess_s
    - area_density_index = excess_auc_linear_baseline_positive / fwhm_excess_s
    - relative_vwave_amplitude = peak_above_linear_baseline / mean(signal)
    - vwave_burden_ratio = excess_auc_linear_baseline_positive / raw_auc_to_zero
    - slope_area_ratio = excess_rise_slope_units_per_s / normalized_positive_excess_auc
    """
    rows = []
    x = segment["time_s"].to_numpy(float)

    for col in signal_cols:
        y = segment[col].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        x_ok = x[ok]
        y_ok = y[ok]

        if len(y_ok) == 0:
            continue

        n = len(y_ok)
        start_t = float(np.nanmin(x_ok))
        end_t = float(np.nanmax(x_ok))
        duration = float(end_t - start_t)

        raw_auc_to_zero = float(trapezoid_area(y_ok, x_ok)) if n > 1 else np.nan

        baseline_start = float(y_ok[0])
        baseline_end = float(y_ok[-1])
        horizontal_baseline = float((baseline_start + baseline_end) / 2.0)

        auc_above_start_baseline = float(trapezoid_area(y_ok - baseline_start, x_ok)) if n > 1 else np.nan
        auc_above_horizontal_baseline = float(trapezoid_area(y_ok - horizontal_baseline, x_ok)) if n > 1 else np.nan

        mean_signal = float(np.nanmean(y_ok))
        median_signal = float(np.nanmedian(y_ok))

        if n > 1:
            linear_baseline = np.linspace(baseline_start, baseline_end, n)
            excess = y_ok - linear_baseline

            excess_auc_linear_baseline_net = float(trapezoid_area(excess, x_ok))
            excess_auc_linear_baseline_positive = float(trapezoid_area(np.maximum(excess, 0), x_ok))
            excess_auc_linear_baseline_negative = float(trapezoid_area(np.minimum(excess, 0), x_ok))

            peak_idx = int(np.nanargmax(y_ok))
            trough_idx = int(np.nanargmin(y_ok))
            excess_peak_idx = int(np.nanargmax(excess))

            peak_value = float(y_ok[peak_idx])
            trough_value = float(y_ok[trough_idx])
            peak_time_s = float(x_ok[peak_idx])
            trough_time_s = float(x_ok[trough_idx])
            excess_peak_value = float(excess[excess_peak_idx])
            excess_peak_time_s = float(x_ok[excess_peak_idx])
            linear_baseline_at_peak = float(linear_baseline[excess_peak_idx])

            time_to_peak_s = float(peak_time_s - start_t)
            time_to_excess_peak_s = float(excess_peak_time_s - start_t)

            rise_time = peak_time_s - start_t
            fall_time = end_t - peak_time_s

            rise_slope = float((peak_value - baseline_start) / rise_time) if rise_time > 0 else np.nan
            fall_slope = float((baseline_end - peak_value) / fall_time) if fall_time > 0 else np.nan

            excess_rise_time = excess_peak_time_s - start_t
            excess_fall_time = end_t - excess_peak_time_s
            excess_rise_slope = float(excess_peak_value / excess_rise_time) if excess_rise_time > 0 else np.nan
            excess_fall_slope = float((0 - excess_peak_value) / excess_fall_time) if excess_fall_time > 0 else np.nan

            normalized_positive_excess_auc = (
                float(excess_auc_linear_baseline_positive / duration) if duration > 0 else np.nan
            )

            # Full width at half maximum of the excess waveform above linear baseline.
            half_max = excess_peak_value / 2.0
            above_half = excess >= half_max if excess_peak_value > 0 else np.zeros_like(excess, dtype=bool)
            if np.any(above_half):
                idxs = np.where(above_half)[0]
                fwhm_s = float(x_ok[idxs[-1]] - x_ok[idxs[0]])
            else:
                fwhm_s = np.nan

            # Symmetry index: 0.5 is symmetric; closer to 0 means early peak, closer to 1 means late peak.
            symmetry_index = float(time_to_excess_peak_s / duration) if duration > 0 else np.nan

            peak_to_mean_ratio = float(peak_value / mean_signal) if mean_signal != 0 else np.nan
            excess_peak_to_mean_excess_ratio = (
                float(excess_peak_value / np.nanmean(np.maximum(excess, 0)))
                if np.nanmean(np.maximum(excess, 0)) != 0
                else np.nan
            )

            # Composite indices for amyloid/restrictive CM vs HFrEF morphology comparison.
            vwave_sharpness_index = (
                float(excess_peak_value / fwhm_s)
                if fwhm_s is not None and np.isfinite(fwhm_s) and fwhm_s > 0
                else np.nan
            )
            area_density_index = (
                float(excess_auc_linear_baseline_positive / fwhm_s)
                if fwhm_s is not None and np.isfinite(fwhm_s) and fwhm_s > 0
                else np.nan
            )
            relative_vwave_amplitude = (
                float(excess_peak_value / mean_signal) if mean_signal != 0 else np.nan
            )
            relative_vwave_amplitude_to_median = (
                float(excess_peak_value / median_signal) if median_signal != 0 else np.nan
            )
            vwave_burden_ratio = (
                float(excess_auc_linear_baseline_positive / raw_auc_to_zero)
                if raw_auc_to_zero not in [0, np.nan] and np.isfinite(raw_auc_to_zero) and raw_auc_to_zero != 0
                else np.nan
            )
            slope_area_ratio = (
                float(excess_rise_slope / normalized_positive_excess_auc)
                if normalized_positive_excess_auc not in [0, np.nan]
                and np.isfinite(normalized_positive_excess_auc)
                and normalized_positive_excess_auc != 0
                else np.nan
            )
            rise_to_fall_slope_ratio = (
                float(abs(excess_rise_slope) / abs(excess_fall_slope))
                if np.isfinite(excess_rise_slope)
                and np.isfinite(excess_fall_slope)
                and abs(excess_fall_slope) > 0
                else np.nan
            )
        else:
            linear_baseline = np.array([baseline_start])
            excess = y_ok - linear_baseline
            excess_auc_linear_baseline_net = np.nan
            excess_auc_linear_baseline_positive = np.nan
            excess_auc_linear_baseline_negative = np.nan
            peak_value = float(y_ok[0])
            trough_value = float(y_ok[0])
            peak_time_s = start_t
            trough_time_s = start_t
            excess_peak_value = float(excess[0])
            excess_peak_time_s = start_t
            linear_baseline_at_peak = baseline_start
            time_to_peak_s = 0.0
            time_to_excess_peak_s = 0.0
            rise_slope = np.nan
            fall_slope = np.nan
            excess_rise_slope = np.nan
            excess_fall_slope = np.nan
            normalized_positive_excess_auc = np.nan
            fwhm_s = np.nan
            symmetry_index = np.nan
            peak_to_mean_ratio = np.nan
            excess_peak_to_mean_excess_ratio = np.nan
            vwave_sharpness_index = np.nan
            area_density_index = np.nan
            relative_vwave_amplitude = np.nan
            relative_vwave_amplitude_to_median = np.nan
            vwave_burden_ratio = np.nan
            slope_area_ratio = np.nan
            rise_to_fall_slope_ratio = np.nan

        rows.append(
            {
                "signal": col,
                "n_points": int(n),
                "start_s": start_t,
                "end_s": end_t,
                "duration_s": duration,

                # Conventional stats
                "min": float(np.nanmin(y_ok)),
                "p05": float(np.nanpercentile(y_ok, 5)),
                "mean": mean_signal,
                "median": median_signal,
                "p95": float(np.nanpercentile(y_ok, 95)),
                "max": float(np.nanmax(y_ok)),
                "std": float(np.nanstd(y_ok, ddof=1)) if n > 1 else 0.0,

                # Baseline definitions
                "baseline_start": baseline_start,
                "baseline_end": baseline_end,
                "horizontal_baseline_start_end_mean": horizontal_baseline,

                # AUC definitions
                "raw_auc_to_zero": raw_auc_to_zero,
                "auc_above_start_baseline": auc_above_start_baseline,
                "auc_above_horizontal_baseline": auc_above_horizontal_baseline,
                "excess_auc_linear_baseline_net": excess_auc_linear_baseline_net,
                "excess_auc_linear_baseline_positive": excess_auc_linear_baseline_positive,
                "excess_auc_linear_baseline_negative": excess_auc_linear_baseline_negative,
                "normalized_positive_excess_auc": normalized_positive_excess_auc,

                # Peak/morphology
                "peak_value": peak_value,
                "peak_time_s": peak_time_s,
                "trough_value": trough_value,
                "trough_time_s": trough_time_s,
                "linear_baseline_at_excess_peak": linear_baseline_at_peak,
                "peak_above_linear_baseline": excess_peak_value,
                "excess_peak_time_s": excess_peak_time_s,
                "time_to_peak_s": time_to_peak_s,
                "time_to_excess_peak_s": time_to_excess_peak_s,
                "rise_slope_raw_units_per_s": rise_slope,
                "fall_slope_raw_units_per_s": fall_slope,
                "excess_rise_slope_units_per_s": excess_rise_slope,
                "excess_fall_slope_units_per_s": excess_fall_slope,
                "fwhm_excess_s": fwhm_s,
                "symmetry_index_excess_peak": symmetry_index,
                "peak_to_mean_ratio": peak_to_mean_ratio,
                "excess_peak_to_mean_positive_excess_ratio": excess_peak_to_mean_excess_ratio,

                # Composite indices for group comparison
                "vwave_sharpness_index": vwave_sharpness_index,
                "area_density_index": area_density_index,
                "relative_vwave_amplitude": relative_vwave_amplitude,
                "relative_vwave_amplitude_to_median": relative_vwave_amplitude_to_median,
                "vwave_burden_ratio": vwave_burden_ratio,
                "slope_area_ratio": slope_area_ratio,
                "rise_to_fall_slope_ratio": rise_to_fall_slope_ratio,
            }
        )

    return pd.DataFrame(rows)

def get_segment(aligned: pd.DataFrame, start_s: float, end_s: float):
    lo, hi = sorted([float(start_s), float(end_s)])
    return aligned[(aligned["time_s"] >= lo) & (aligned["time_s"] <= hi)].copy()


def sanitize_label_token(text: str):
    """
    Make patient/case ID safe for filenames and interval labels.
    """
    text = str(text).strip()
    if not text:
        return "case"
    text = re.sub(r"[^A-Za-z0-9_-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "case"


def next_vwave_label(patient_id: str, intervals: list):
    """
    Default labels:
      PatientID_vwave_1
      PatientID_vwave_2
      PatientID_vwave_3
    """
    pid = sanitize_label_token(patient_id)
    vwave_count = sum(1 for item in intervals if "_vwave_" in str(item.get("label", "")).lower())
    return f"{pid}_vwave_{vwave_count + 1}"


def build_labeled_exports(aligned: pd.DataFrame, intervals: list, signal_cols: list):
    """
    Build long-form selected segments and per-interval/per-signal stats.
    """
    segment_parts = []
    stats_parts = []

    for i, item in enumerate(intervals):
        label = item["label"]
        start_s = float(item["start_s"])
        end_s = float(item["end_s"])
        segment = get_segment(aligned, start_s, end_s)
        if segment.empty:
            continue

        long = segment.melt(id_vars="time_s", value_vars=signal_cols, var_name="signal", value_name="value")
        long.insert(0, "interval_id", i + 1)
        long.insert(1, "interval_label", label)
        long.insert(2, "interval_start_s", min(start_s, end_s))
        long.insert(3, "interval_end_s", max(start_s, end_s))
        segment_parts.append(long)

        stats = calculate_stats(segment, signal_cols)
        stats.insert(0, "interval_id", i + 1)
        stats.insert(1, "interval_label", label)
        stats.insert(2, "interval_start_s", min(start_s, end_s))
        stats.insert(3, "interval_end_s", max(start_s, end_s))
        stats_parts.append(stats)

    labeled_segments = pd.concat(segment_parts, ignore_index=True) if segment_parts else pd.DataFrame()
    labeled_stats = pd.concat(stats_parts, ignore_index=True) if stats_parts else pd.DataFrame()

    return labeled_segments, labeled_stats


def build_labeled_raw_segments(aligned: pd.DataFrame, intervals: list, signal_cols: list):
    """
    Build wide-form raw waveform samples for each labeled interval.
    This is intended for downstream ML/AI workflows and visual replay.
    """
    parts = []
    for i, item in enumerate(intervals):
        label = item["label"]
        start_s = float(item["start_s"])
        end_s = float(item["end_s"])
        lo, hi = sorted([start_s, end_s])
        segment = get_segment(aligned, lo, hi)
        if segment.empty:
            continue

        available_cols = ["time_s"] + [c for c in signal_cols if c in segment.columns]
        wide = segment[available_cols].copy()
        wide.insert(0, "interval_id", i + 1)
        wide.insert(1, "interval_label", label)
        wide.insert(2, "interval_start_s", lo)
        wide.insert(3, "interval_end_s", hi)
        wide.insert(4, "relative_time_s", wide["time_s"] - lo)
        parts.append(wide)

    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def add_case_metadata(df: pd.DataFrame, patient_id: str = "", procedure_date: str = "", notes: str = ""):
    """
    Add case-level metadata columns to exported tables.
    """
    out = df.copy()
    out.insert(0, "patient_id", patient_id)
    out.insert(1, "procedure_date", procedure_date)
    out.insert(2, "case_notes", notes)
    return out


def package_outputs(
    aligned: pd.DataFrame,
    segment: pd.DataFrame,
    stats: pd.DataFrame,
    labeled_segments: pd.DataFrame,
    labeled_stats: pd.DataFrame,
    raw_labeled_segments: pd.DataFrame,
    patient_id: str = "",
    procedure_date: str = "",
    notes: str = "",
):
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("aligned_full_waveforms.csv", add_case_metadata(aligned, patient_id, procedure_date, notes).to_csv(index=False))
        z.writestr("current_selected_segment.csv", add_case_metadata(segment, patient_id, procedure_date, notes).to_csv(index=False))
        z.writestr("current_selected_segment_stats.csv", add_case_metadata(stats, patient_id, procedure_date, notes).to_csv(index=False))
        if not labeled_segments.empty:
            z.writestr("labeled_intervals_segments_long.csv", add_case_metadata(labeled_segments, patient_id, procedure_date, notes).to_csv(index=False))
        if not raw_labeled_segments.empty:
            z.writestr("raw_labeled_waveform_segments_wide.csv", add_case_metadata(raw_labeled_segments, patient_id, procedure_date, notes).to_csv(index=False))
        if not labeled_stats.empty:
            z.writestr("labeled_intervals_stats.csv", add_case_metadata(labeled_stats, patient_id, procedure_date, notes).to_csv(index=False))

        metadata = pd.DataFrame([{
            "patient_id": patient_id,
            "procedure_date": procedure_date,
            "case_notes": notes,
        }])
        z.writestr("case_metadata.csv", metadata.to_csv(index=False))
    mem.seek(0)
    return mem


def segment_signal_columns(df: pd.DataFrame):
    metadata_cols = {
        "saved_at",
        "source_files",
        "database_case_key",
        "patient_id",
        "procedure_date",
        "case_notes",
        "interval_id",
        "interval_label",
        "interval_start_s",
        "interval_end_s",
        "relative_time_s",
        "time_s",
    }
    return [c for c in df.columns if c not in metadata_cols and pd.api.types.is_numeric_dtype(df[c])]


def labeled_interval_figure(segment_df: pd.DataFrame, title: str = "Labeled waveform segment"):
    signal_cols = segment_signal_columns(segment_df)
    pressure_cols = sorted([c for c in signal_cols if "EKG" not in c.upper()], key=pressure_sort_key)
    ecg_cols = sorted([c for c in signal_cols if "EKG" in c.upper()])
    plot_cols = ecg_cols + pressure_cols

    rows = 2 if ecg_cols and pressure_cols else 1
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(["EKG", "Pressure"] if rows == 2 else ["Waveform segment"]),
    )

    x = segment_df["relative_time_s"] if "relative_time_s" in segment_df.columns else segment_df["time_s"]
    for c in plot_cols:
        is_ecg = "EKG" in c.upper()
        row = 1 if rows == 1 or is_ecg else 2
        fig.add_trace(
            go.Scatter(
                x=x,
                y=segment_df[c],
                mode="lines",
                name=c,
                hovertemplate=f"Time: %{{x:.3f}} s<br>{c}: %{{y:.3f}}<extra></extra>",
            ),
            row=row,
            col=1,
        )

    fig.update_xaxes(title_text="Relative time (s)", row=rows, col=1)
    fig.update_yaxes(title_text="mV", row=1, col=1)
    if rows == 2:
        fig.update_yaxes(title_text="mmHg", row=2, col=1)
    else:
        fig.update_yaxes(title_text="Value", row=1, col=1)

    fig.update_layout(
        title=title,
        height=460 if rows == 2 else 330,
        hovermode="x",
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
        margin=dict(t=70, b=95),
    )
    return fig


def build_labeled_visualizations_html(
    raw_labeled_segments: pd.DataFrame,
    patient_id: str = "",
    procedure_date: str = "",
):
    if raw_labeled_segments.empty:
        return ""

    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Labeled waveform segments</title>",
        "<style>body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:28px;color:#111827;}"
        ".segment{margin:0 0 42px;} h1{margin-bottom:0.2rem;} .meta{color:#6b7280;margin-bottom:2rem;}</style>",
        "</head><body>",
        "<h1>Labeled waveform segments</h1>",
        f"<div class='meta'>Patient/case: {html.escape(patient_id or 'Unknown')}"
        f"{' | Procedure date: ' + html.escape(procedure_date) if procedure_date else ''}</div>",
    ]

    include_plotlyjs = True
    grouping_cols = ["interval_id", "interval_label", "interval_start_s", "interval_end_s"]
    for keys, segment_df in raw_labeled_segments.groupby(grouping_cols, dropna=False, sort=True):
        interval_id, label, start_s, end_s = keys
        title = f"{interval_id}. {label} ({float(start_s):.3f}-{float(end_s):.3f} s)"
        fig = labeled_interval_figure(segment_df, title=title)
        parts.append("<div class='segment'>")
        parts.append(pio.to_html(fig, full_html=False, include_plotlyjs=True if include_plotlyjs else False))
        parts.append("</div>")
        include_plotlyjs = False

    parts.append("</body></html>")
    return "\n".join(parts)


def default_database_path() -> Path:
    configured_path = app_setting("XPER_DATABASE_PATH", "")
    if configured_path:
        return Path(configured_path).expanduser()
    return Path.home() / "Documents" / "Xper Hemodynamic Viewer" / "xper_hemo_cases.sqlite"


def app_setting(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name, "")
    except Exception:
        value = ""
    return str(value or os.environ.get(name, default) or "")


def asset_data_uri(relative_path: str, mime_type: str = "image/png") -> str:
    path = Path(__file__).resolve().parent / relative_path
    if not path.exists():
        return ""
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{data}"


def render_institutional_header():
    bwh_logo_uri = asset_data_uri("assets/bwh_logo.png", "image/png")
    institution_logo_uri = asset_data_uri("assets/institution_logo.svg", "image/svg+xml")
    logos = [
        (bwh_logo_uri, "Brigham and Women's Hospital", "height: 48px; max-width: min(450px, 92vw);"),
        (institution_logo_uri, "Institutional logo", "height: 54px; max-width: min(300px, 70vw);"),
    ]
    visible_logos = [(src, alt, style) for src, alt, style in logos if src]
    if not visible_logos:
        return

    logo_html = "\n".join(
        f'<img src="{src}" alt="{alt}" style="{style} width: auto; object-fit: contain;">'
        for src, alt, style in visible_logos
    )
    st.markdown(
        f"""
        <div style="
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1.25rem;
            flex-wrap: wrap;
            padding: 0.35rem 0 0.9rem;
            margin-bottom: 0.25rem;
            border-bottom: 1px solid rgba(49, 51, 63, 0.12);
        ">
            {logo_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def require_password_if_configured():
    expected_password = app_setting("APP_PASSWORD", "")
    if not expected_password:
        return

    if st.session_state.get("authenticated", False):
        with st.sidebar:
            if st.button("Lock app"):
                st.session_state.authenticated = False
                st.rerun()
        return

    render_institutional_header()
    st.title("Hemodynamic RHC Viewer")
    st.caption("Enter the app password to continue.")
    entered_password = st.text_input("Password", type="password", key="app_password_entry")
    if st.button("Open app", type="primary"):
        if entered_password == expected_password:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()


def quote_sql_identifier(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def sqlite_type_for_series(series: pd.Series) -> str:
    if pd.api.types.is_integer_dtype(series):
        return "INTEGER"
    if pd.api.types.is_float_dtype(series) or pd.api.types.is_bool_dtype(series):
        return "REAL"
    return "TEXT"


def normalize_for_sqlite(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_object_dtype(out[col]) or pd.api.types.is_string_dtype(out[col]):
            out[col] = out[col].where(out[col].isna(), out[col].astype(str))
    return out


def ensure_sqlite_table(conn: sqlite3.Connection, table_name: str, df: pd.DataFrame) -> None:
    quoted_table = quote_sql_identifier(table_name)
    existing = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()

    if existing is None:
        df.head(0).to_sql(table_name, conn, index=False)
        return

    existing_cols = {
        row[1]
        for row in conn.execute(f"PRAGMA table_info({quoted_table})").fetchall()
    }
    for col in df.columns:
        if col not in existing_cols:
            col_type = sqlite_type_for_series(df[col])
            conn.execute(
                f"ALTER TABLE {quoted_table} ADD COLUMN {quote_sql_identifier(col)} {col_type}"
            )


def save_labeled_stats_to_database(
    labeled_stats: pd.DataFrame,
    raw_labeled_segments: pd.DataFrame,
    patient_id: str,
    procedure_date: str,
    notes: str,
    source_files: list[str],
    db_path: Path,
) -> dict:
    if labeled_stats.empty and raw_labeled_segments.empty:
        return {"stats_rows": 0, "segment_rows": 0}

    saved_at = datetime.now().isoformat(timespec="seconds")
    source_files_text = "; ".join(source_files)
    database_case_key = sanitize_label_token(patient_id)

    stats_rows = pd.DataFrame()
    if not labeled_stats.empty:
        stats_rows = add_case_metadata(labeled_stats, patient_id, procedure_date, notes)
        stats_rows.insert(0, "saved_at", saved_at)
        stats_rows.insert(1, "source_files", source_files_text)
        stats_rows.insert(2, "database_case_key", database_case_key)
        stats_rows = normalize_for_sqlite(stats_rows)

    segment_rows = pd.DataFrame()
    if not raw_labeled_segments.empty:
        segment_rows = add_case_metadata(raw_labeled_segments, patient_id, procedure_date, notes)
        segment_rows.insert(0, "saved_at", saved_at)
        segment_rows.insert(1, "source_files", source_files_text)
        segment_rows.insert(2, "database_case_key", database_case_key)
        segment_rows = normalize_for_sqlite(segment_rows)

    db_path = Path(db_path).expanduser()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        if not stats_rows.empty:
            ensure_sqlite_table(conn, DATABASE_TABLE, stats_rows)
            stats_rows.to_sql(DATABASE_TABLE, conn, if_exists="append", index=False)
        if not segment_rows.empty:
            ensure_sqlite_table(conn, DATABASE_SEGMENTS_TABLE, segment_rows)
            segment_rows.to_sql(DATABASE_SEGMENTS_TABLE, conn, if_exists="append", index=False)
    return {"stats_rows": len(stats_rows), "segment_rows": len(segment_rows)}


def load_database_table(db_path: Path, table_name: str) -> pd.DataFrame:
    db_path = Path(db_path).expanduser()
    if not db_path.exists():
        return pd.DataFrame()
    with sqlite3.connect(db_path) as conn:
        table_exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        ).fetchone()
        if table_exists is None:
            return pd.DataFrame()
        order_clause = " ORDER BY saved_at DESC" if table_name == DATABASE_TABLE else " ORDER BY saved_at DESC, interval_id, time_s"
        return pd.read_sql_query(f"SELECT * FROM {quote_sql_identifier(table_name)}{order_clause}", conn)


def load_database_rows(db_path: Path) -> pd.DataFrame:
    return load_database_table(db_path, DATABASE_TABLE)


def load_database_segments(db_path: Path) -> pd.DataFrame:
    return load_database_table(db_path, DATABASE_SEGMENTS_TABLE)


def database_summary(db_path: Path) -> dict:
    rows = load_database_rows(db_path)
    if rows.empty:
        return {"rows": 0, "cases": 0, "intervals": 0}
    cases = rows["patient_id"].nunique() if "patient_id" in rows.columns else 0
    if {"patient_id", "interval_label"}.issubset(rows.columns):
        intervals = rows[["patient_id", "interval_label"]].drop_duplicates().shape[0]
    elif "interval_label" in rows.columns:
        intervals = rows["interval_label"].nunique()
    else:
        intervals = 0
    return {"rows": len(rows), "cases": cases, "intervals": intervals}


def ecg_source_sort_key(candidate):
    filename = candidate[0].upper()
    label = str(candidate[1]).upper()
    if label in {"PW", "PCWP"}:
        return (0, filename)
    if label == "PA":
        return (1, filename)
    if label in {"RA", "RV"}:
        return (2, filename)
    if "WAV_000" in filename:
        return (4, filename)
    return (3, filename)


def pressure_sort_key(label_or_col: str, filename: str = ""):
    text = f"{label_or_col} {filename}".upper()
    if "PCWP" in text or re.search(r"(^|[^A-Z])PW([^A-Z]|$)", text) or "WAV_006" in text:
        return (0, text)
    if "PA" in text or "WAV_008" in text:
        return (1, text)
    if "RA" in text or "WAV_002" in text:
        return (2, text)
    if "RV" in text or "WAV_004" in text:
        return (3, text)
    if "AO" in text or "LV" in text:
        return (4, text)
    return (5, text)


def is_pcwp_signal(label_or_col: str, filename: str = "") -> bool:
    return pressure_sort_key(label_or_col, filename)[0] == 0


def is_rv_signal(label_or_col: str, filename: str = "") -> bool:
    text = f"{label_or_col} {filename}".upper()
    return bool(re.search(r"(^|[^A-Z])RV([^A-Z]|$)", text) or "WAV_004" in text)


def is_reference_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in {".pdf", ".jpg", ".jpeg", ".png"}


def reference_sort_key(reference: dict, pressure_col: str = "", source_filename: str = ""):
    name = reference["name"].upper()
    pressure_text = f"{pressure_col} {source_filename}".upper()
    score = 0
    if is_pcwp_signal(pressure_text) and ("PCWP" in name or re.search(r"(^|[^A-Z])PW([^A-Z]|$)", name)):
        score -= 40
    if "WAV_006" in name:
        score -= 30
    source_stem = Path(source_filename).stem.upper()
    ref_stem = Path(reference["name"]).stem.upper()
    if source_stem and (source_stem in ref_stem or ref_stem in source_stem):
        score -= 25
    for token in re.split(r"[^A-Z0-9]+", pressure_text):
        if len(token) >= 2 and token in name:
            score -= 5
    if reference["ext"] == ".pdf":
        score -= 1
    return (score, name)


def render_reference_file(reference: dict):
    caption = f"Reference waveform output: {reference['name']}"
    if reference["ext"] == ".pdf":
        st.pdf(reference["data"], height=620, key=f"reference_pdf_{reference['name']}")
        st.caption(caption)
    else:
        st.image(reference["data"], caption=caption, width="stretch")


def resolve_ecg_layout(filename: str, requested_layout: str) -> str:
    if requested_layout != "automatic_by_file_type":
        return requested_layout
    if is_dedicated_ecg_file(filename):
        return "6_lead"
    return "auto"


def describe_ecg_candidate(candidate):
    name, label, ecg_df = candidate
    ecg_cols = [c for c in ecg_df.columns if c.startswith("EKG_") and c.endswith("_mV")]
    duration = float(ecg_df["time_s"].max()) if "time_s" in ecg_df.columns and len(ecg_df) else 0.0
    lead_text = ", ".join(c.replace("EKG_", "").replace("_mV", "") for c in ecg_cols)
    source_type = "dedicated ECG" if "WAV_000" in name.upper() else "pressure-strip ECG"
    return f"{name} ({label}, {source_type}) - {len(ecg_cols)} lead(s): {lead_text}; {len(ecg_df)} samples/lead over {duration:.2f} s"


def preferred_ecg_column(ecg_cols):
    for col in ("EKG_DII_mV", "EKG_II_mV", "EKG_DI_mV", "EKG_I_mV"):
        if col in ecg_cols:
            return col
    return ecg_cols[0] if ecg_cols else None


def preferred_overlay_ecg_column(ecg_cols):
    for col in ("EKG_DII_mV", "EKG_II_mV", "EKG_DI_mV", "EKG_I_mV", "EKG_DIII_mV", "EKG_III_mV"):
        if col in ecg_cols:
            return col
    return ecg_cols[0] if ecg_cols else None


def detect_r_waves(ecg_time: np.ndarray, ecg_signal: np.ndarray, min_distance_s: float = 0.35, source_signal: str = ""):
    """
    Simple R-wave detector for an ECG signal.
    Uses whichever polarity has the larger absolute deflection.
    Returns a DataFrame with r_time_s and r_value.
    """
    try:
        from scipy.signal import find_peaks
    except Exception:
        return pd.DataFrame(columns=["r_time_s", "r_value", "r_polarity", "r_wave_source_signal"])

    x = np.asarray(ecg_time, dtype=float)
    y = np.asarray(ecg_signal, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if len(y) < 10:
        return pd.DataFrame(columns=["r_time_s", "r_value", "r_polarity", "r_wave_source_signal"])

    # Detrend using median and choose polarity.
    y0 = y - np.nanmedian(y)
    polarity = 1 if np.nanmax(y0) >= abs(np.nanmin(y0)) else -1
    yp = y0 * polarity

    fs_est = 1.0 / np.nanmedian(np.diff(x)) if len(x) > 2 and np.nanmedian(np.diff(x)) > 0 else 500
    distance = max(1, int(min_distance_s * fs_est))

    # Adaptive threshold: favor clear QRS-like peaks, but remain permissive.
    prominence = max(np.nanstd(yp) * 1.0, (np.nanpercentile(yp, 95) - np.nanpercentile(yp, 50)) * 0.5)
    height = np.nanpercentile(yp, 80)

    peaks, _ = find_peaks(yp, distance=distance, prominence=prominence, height=height)
    return pd.DataFrame({"r_time_s": x[peaks], "r_value": y[peaks], "r_polarity": polarity, "r_wave_source_signal": source_signal})


def detect_r_waves_from_ecg_source(time_s: np.ndarray, ecg_source: dict, min_required: int = 2):
    if not ecg_source:
        return pd.DataFrame(columns=["r_time_s", "r_value", "r_polarity", "r_wave_source_signal"]), ""

    candidates = []
    primary_col = ecg_source.get("ecg_col", "")
    values_by_col = ecg_source.get("values_by_col") or {}
    if primary_col and primary_col in values_by_col:
        candidates.append((primary_col, values_by_col[primary_col]))
    for col, values in values_by_col.items():
        if col != primary_col:
            candidates.append((col, values))

    best = pd.DataFrame(columns=["r_time_s", "r_value", "r_polarity", "r_wave_source_signal"])
    best_col = ""
    for col, values in candidates:
        detected = detect_r_waves(time_s, np.asarray(values, dtype=float), source_signal=col)
        if len(detected) > len(best):
            best = detected
            best_col = col
        if len(detected) >= min_required:
            return detected, col
    return best, best_col


def pcwp_r_wave_source(analysis_signal_cols: list, same_file_ecg_by_pressure_col: dict):
    """
    Use the ECG trace embedded in the PCWP/PW pressure PW6 file for all ECG timing metrics.
    Dedicated WAV_000 ECG files and other pressure-strip ECGs can still be displayed, but they
    are not used for qrs_to_excess_peak_ms.
    """
    for pressure_col in analysis_signal_cols:
        if not is_pcwp_signal(pressure_col):
            continue
        source = same_file_ecg_by_pressure_col.get(pressure_col)
        if source is not None and source.get("values") is not None:
            return pressure_col, source
    return None, None


def zscore_signal(y: np.ndarray):
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(y)
    if finite.sum() < 3:
        return None

    filled = y.copy()
    if not finite.all():
        idx = np.arange(len(y))
        filled[~finite] = np.interp(idx[~finite], idx[finite], y[finite])

    center = np.nanmedian(filled)
    spread = np.nanstd(filled - center)
    if not np.isfinite(spread) or spread <= 0:
        return None
    return (filled - center) / spread


def interpolate_rv_piso_sine(t_fit: np.ndarray, p_fit: np.ndarray, t_start: float, t_end: float, min_piso: float | None = None):
    t_fit = np.asarray(t_fit, dtype=float)
    p_fit = np.asarray(p_fit, dtype=float)
    ok = np.isfinite(t_fit) & np.isfinite(p_fit)
    t_fit = t_fit[ok]
    p_fit = p_fit[ok]
    if len(t_fit) < 8 or not np.isfinite(t_start) or not np.isfinite(t_end) or t_end <= t_start:
        return None

    try:
        from scipy.optimize import least_squares
    except Exception:
        return None

    x_fit = t_fit - t_start
    fit_duration = max(float(t_end - t_start), 1e-3)
    amp0 = max((float(np.nanmax(p_fit)) - float(np.nanmin(p_fit))) / 2.0, 1.0)
    offset0 = float(np.nanmean(p_fit))
    omega0 = 2.0 * np.pi / fit_duration
    min_piso = float(min_piso) if min_piso is not None and np.isfinite(min_piso) else None

    def sine_model(x, amp, omega, phase, offset):
        return offset + amp * np.sin(omega * x + phase)

    lower_bounds = np.array([0.0, np.pi / (2.0 * fit_duration), -4.0 * np.pi, -200.0])
    upper_bounds = np.array([300.0, 8.0 * np.pi / fit_duration, 4.0 * np.pi, 300.0])
    t_curve = np.linspace(float(t_start), float(t_end), 220)
    x_curve = t_curve - t_start

    def objective(params):
        residuals = sine_model(x_fit, *params) - p_fit
        if min_piso is None:
            return residuals
        piso_gap = min_piso - float(np.nanmax(sine_model(x_curve, *params)))
        if piso_gap <= 0:
            return residuals
        return np.concatenate([residuals, np.repeat(piso_gap * 8.0, 8)])

    candidate_params = []
    for omega_factor in (0.75, 1.0, 1.5, 2.0):
        for phase0 in (-np.pi, -np.pi / 2.0, 0.0, np.pi / 2.0, np.pi):
            candidate_params.append([amp0, omega0 * omega_factor, phase0, offset0])
    if min_piso is not None:
        candidate_params.append([
            max(min_piso - float(np.nanmin(p_fit)), amp0, 1.0),
            omega0,
            -np.pi / 2.0,
            float(np.nanmin(p_fit)),
        ])

    best = None
    best_score = np.inf
    for initial in candidate_params:
        initial = np.clip(np.asarray(initial, dtype=float), lower_bounds, upper_bounds)
        try:
            result = least_squares(
                objective,
                initial,
                bounds=(lower_bounds, upper_bounds),
                max_nfev=8000,
            )
        except Exception:
            continue
        if not result.success:
            continue
        y_candidate = sine_model(x_curve, *result.x)
        if not np.isfinite(y_candidate).any():
            continue
        candidate_piso = float(np.nanmax(y_candidate))
        if min_piso is not None and candidate_piso < min_piso:
            continue
        score = float(np.nanmean((sine_model(x_fit, *result.x) - p_fit) ** 2))
        if score < best_score:
            best = result.x
            best_score = score

    if best is None:
        return None

    y_curve = sine_model(x_curve, *best)
    if not np.isfinite(y_curve).any():
        return None
    peak_idx = int(np.nanargmax(y_curve))
    residuals = p_fit - sine_model(x_fit, *best)
    rmse = float(np.sqrt(np.nanmean(residuals ** 2))) if len(residuals) else np.nan
    return {
        "time_s": t_curve,
        "pressure_mmHg": y_curve,
        "piso_mmHg": float(y_curve[peak_idx]),
        "piso_time_s": float(t_curve[peak_idx]),
        "fit_rmse_mmHg": rmse,
        "fit_n": int(len(t_fit)),
    }


def rv_single_beat_derivative_analysis(time_s: np.ndarray, pressure: np.ndarray, r_wave_times: np.ndarray | None = None):
    """
    Beat-level visual feature detection for the Bellofiore RV single-beat methods.
    Consecutive R waves define each beat window. Within each QRS-to-QRS window,
    identify dP/dt extrema, estimate a first-derivative Piso sine interpolation, and find
    candidate second-derivative minima. This does not yet calculate final Ees.
    """
    t = np.asarray(time_s, dtype=float)
    p = np.asarray(pressure, dtype=float)
    ok = np.isfinite(t) & np.isfinite(p)
    if ok.sum() < 12:
        return None

    t_ok = t[ok]
    p_ok = p[ok]
    order = np.argsort(t_ok)
    t_ok = t_ok[order]
    p_ok = p_ok[order]
    unique_time, unique_idx = np.unique(t_ok, return_index=True)
    t_ok = unique_time
    p_ok = p_ok[unique_idx]
    if len(t_ok) < 12:
        return None

    dt = np.nanmedian(np.diff(t_ok))
    if not np.isfinite(dt) or dt <= 0:
        return None

    p_smooth = p_ok.copy()
    try:
        from scipy.signal import savgol_filter, find_peaks

        target_window_s = 0.055
        window = int(round(target_window_s / dt))
        window = max(7, window)
        if window % 2 == 0:
            window += 1
        if window >= len(p_smooth):
            window = len(p_smooth) - 1 if len(p_smooth) % 2 == 0 else len(p_smooth)
        if window >= 7:
            p_smooth = savgol_filter(p_smooth, window_length=window, polyorder=3, mode="interp")
    except Exception:
        find_peaks = None

    dpdt = np.gradient(p_smooth, t_ok)
    d2pdt2 = np.gradient(dpdt, t_ok)

    derivative_df = pd.DataFrame(
        {
            "time_s": t_ok,
            "rv_pressure_smooth_mmHg": p_smooth,
            "rv_dpdt_mmHg_per_s": dpdt,
            "rv_d2pdt2_mmHg_per_s2": d2pdt2,
        }
    )

    r_times = np.asarray([] if r_wave_times is None else r_wave_times, dtype=float)
    r_times = np.sort(r_times[np.isfinite(r_times)])
    r_times = r_times[(r_times >= t_ok[0]) & (r_times <= t_ok[-1])]
    if len(r_times) < 2:
        return derivative_df, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    events = []
    fit_parts = []
    sample_parts = []
    beat_id = 0
    for beat_start, beat_end in zip(r_times[:-1], r_times[1:]):
        rr_s = float(beat_end - beat_start)
        if rr_s < 0.40 or rr_s > 1.80:
            continue

        beat_mask = (t_ok >= beat_start) & (t_ok < beat_end)
        beat_indices = np.flatnonzero(beat_mask)
        if len(beat_indices) < 12:
            continue

        beat_id += 1
        pressure_peak_idx = int(beat_indices[np.nanargmax(p_smooth[beat_indices])])
        early_indices = beat_indices[t_ok[beat_indices] <= beat_start + 0.65 * rr_s]
        if len(early_indices) < 4:
            early_indices = beat_indices
        late_indices = beat_indices[t_ok[beat_indices] >= t_ok[pressure_peak_idx]]
        if len(late_indices) < 4:
            late_indices = beat_indices

        dpdt_max_idx = int(early_indices[np.nanargmax(dpdt[early_indices])])
        dpdt_min_idx = int(late_indices[np.nanargmin(dpdt[late_indices])])

        base_event = {
            "beat_id": beat_id,
            "rr_start_s": float(beat_start),
            "rr_end_s": float(beat_end),
            "rr_duration_s": rr_s,
        }
        events.extend(
            [
                {
                    **base_event,
                    "event": "RV pressure peak",
                    "method": "Peak detection",
                    "time_s": float(t_ok[pressure_peak_idx]),
                    "value": float(p_smooth[pressure_peak_idx]),
                    "row": "rv_peak_fit",
                    "description": "Measured RV pressure peak within QRS-to-QRS beat",
                },
                {
                    **base_event,
                    "event": "dP/dt max",
                    "method": "First derivative",
                    "time_s": float(t_ok[dpdt_max_idx]),
                    "value": float(dpdt[dpdt_max_idx]),
                    "row": "dpdt",
                    "description": "Beat-level isovolumic contraction reference",
                },
                {
                    **base_event,
                    "event": "dP/dt min",
                    "method": "First derivative",
                    "time_s": float(t_ok[dpdt_min_idx]),
                    "value": float(dpdt[dpdt_min_idx]),
                    "row": "dpdt",
                    "description": "Beat-level isovolumic relaxation reference",
                },
            ]
        )

        second_derivative_minima = []
        if find_peaks is not None:
            beat_d2 = d2pdt2[beat_indices]
            distance = max(1, int(round(0.08 / dt)))
            iqr = np.nanpercentile(beat_d2, 75) - np.nanpercentile(beat_d2, 25)
            prominence = max(np.nanstd(beat_d2) * 0.25, iqr * 0.25)
            peaks, _ = find_peaks(-beat_d2, distance=distance, prominence=prominence)
            second_derivative_minima = [int(beat_indices[i]) for i in peaks]

        if not second_derivative_minima:
            second_derivative_minima = list(beat_indices)

        opening_candidates = [
            i for i in second_derivative_minima
            if t_ok[dpdt_max_idx] <= t_ok[i] <= t_ok[pressure_peak_idx]
        ]
        if not opening_candidates:
            opening_candidates = [i for i in second_derivative_minima if i < pressure_peak_idx]

        closing_candidates = [
            i for i in second_derivative_minima
            if t_ok[pressure_peak_idx] <= t_ok[i] <= t_ok[dpdt_min_idx]
        ]
        if not closing_candidates:
            closing_candidates = [i for i in second_derivative_minima if i > pressure_peak_idx]

        selected_minima = []
        if opening_candidates:
            selected_minima.append(("PV opening candidate", min(opening_candidates, key=lambda i: d2pdt2[i])))
        if closing_candidates:
            selected_minima.append(("PV closing candidate", min(closing_candidates, key=lambda i: d2pdt2[i])))

        ic_threshold = 0.20 * float(dpdt[dpdt_max_idx])
        ic_candidates = beat_indices[(beat_indices <= dpdt_max_idx) & (dpdt[beat_indices] >= ic_threshold)]
        ic_onset_idx = int(ic_candidates[0]) if len(ic_candidates) else int(beat_indices[0])

        ir_threshold = 0.20 * float(dpdt[dpdt_min_idx])
        ir_candidates = beat_indices[(beat_indices >= dpdt_min_idx) & (dpdt[beat_indices] >= ir_threshold)]
        ir_end_idx = int(ir_candidates[0]) if len(ir_candidates) else int(beat_indices[-1])

        fit_mask = (
            ((t_ok >= t_ok[ic_onset_idx]) & (t_ok <= t_ok[dpdt_max_idx]))
            | ((t_ok >= t_ok[dpdt_min_idx]) & (t_ok <= t_ok[ir_end_idx]))
        )
        fit_indices = np.flatnonzero(fit_mask)
        measured_peak = float(p_smooth[pressure_peak_idx])
        min_piso = measured_peak + max(1.0, 0.02 * abs(measured_peak))
        fit_result = interpolate_rv_piso_sine(
            t_ok[fit_indices],
            p_smooth[fit_indices],
            float(t_ok[ic_onset_idx]),
            float(t_ok[ir_end_idx]),
            min_piso=min_piso,
        )
        if fit_result is not None:
            ic_fit_indices = np.flatnonzero((t_ok >= t_ok[ic_onset_idx]) & (t_ok <= t_ok[dpdt_max_idx]))
            ir_fit_indices = np.flatnonzero((t_ok >= t_ok[dpdt_min_idx]) & (t_ok <= t_ok[ir_end_idx]))
            sample_parts.append(
                pd.DataFrame(
                    {
                        "beat_id": beat_id,
                        "time_s": np.concatenate([t_ok[ic_fit_indices], t_ok[ir_fit_indices]]),
                        "pressure_mmHg": np.concatenate([p_smooth[ic_fit_indices], p_smooth[ir_fit_indices]]),
                        "range": ["IC interpolation samples"] * len(ic_fit_indices)
                        + ["IR interpolation samples"] * len(ir_fit_indices),
                    }
                )
            )
            fit_df = pd.DataFrame(
                {
                    "beat_id": beat_id,
                    "time_s": fit_result["time_s"],
                    "piso_fit_mmHg": fit_result["pressure_mmHg"],
                    "measured_peak_mmHg": measured_peak,
                    "piso_mmHg": fit_result["piso_mmHg"],
                    "piso_margin_mmHg": fit_result["piso_mmHg"] - measured_peak,
                    "piso_time_s": fit_result["piso_time_s"],
                    "fit_rmse_mmHg": fit_result["fit_rmse_mmHg"],
                    "fit_n": fit_result["fit_n"],
                }
            )
            fit_parts.append(fit_df)
            events.extend(
                [
                    {
                        **base_event,
                        "event": "IC onset 20% dP/dt max",
                        "method": "First derivative sine interpolation",
                        "time_s": float(t_ok[ic_onset_idx]),
                        "value": float(p_smooth[ic_onset_idx]),
                        "row": "rv_peak_fit",
                        "description": "Start of IC interpolation range",
                    },
                    {
                        **base_event,
                        "event": "IR end 20% dP/dt min",
                        "method": "First derivative sine interpolation",
                        "time_s": float(t_ok[ir_end_idx]),
                        "value": float(p_smooth[ir_end_idx]),
                        "row": "rv_peak_fit",
                        "description": "End of IR interpolation range",
                    },
                    {
                        **base_event,
                        "event": "Piso estimate",
                        "method": "First derivative sine interpolation",
                        "time_s": float(fit_result["piso_time_s"]),
                        "value": float(fit_result["piso_mmHg"]),
                        "row": "rv_peak_fit",
                        "description": (
                            f"Sine interpolation peak pressure; {fit_result['piso_mmHg'] - measured_peak:.2f} mmHg "
                            f"above measured RV peak; RMSE {fit_result['fit_rmse_mmHg']:.2f} mmHg"
                        ),
                    },
                ]
            )

        for label, idx in selected_minima:
            events.append(
                {
                    **base_event,
                    "event": label,
                    "method": "Second derivative",
                    "time_s": float(t_ok[idx]),
                    "value": float(d2pdt2[idx]),
                    "row": "d2pdt2",
                    "description": "Beat-level candidate sharp slope-change point",
                }
            )

    fits_df = pd.concat(fit_parts, ignore_index=True) if fit_parts else pd.DataFrame()
    samples_df = pd.concat(sample_parts, ignore_index=True) if sample_parts else pd.DataFrame()
    return derivative_df, pd.DataFrame(events), fits_df, samples_df


def pressure_ecg_lag_window(
    time_s: np.ndarray,
    pressure: np.ndarray,
    ecg: np.ndarray,
    start_s: float,
    end_s: float,
    max_lag_s: float = 1.0,
    feature: str = "positive_upslope",
):
    """
    Estimate the lag where pressure features best follow ECG R-wave impulses.
    Positive lag means the pressure feature occurs later/right of the ECG R wave.
    This is a timing stability QC, not a claim that the physiologic lag should be zero.
    """
    t = np.asarray(time_s, dtype=float)
    p = np.asarray(pressure, dtype=float)
    e = np.asarray(ecg, dtype=float)
    if len(t) < 10 or len(p) != len(t) or len(e) != len(t):
        return None

    dt = np.nanmedian(np.diff(t))
    if not np.isfinite(dt) or dt <= 0:
        return None

    in_window = (t >= start_s) & (t <= end_s) & np.isfinite(e)
    if in_window.sum() < 10:
        return None

    r_waves = detect_r_waves(t[in_window], e[in_window])
    if len(r_waves) < 2:
        return None

    p_z = zscore_signal(p)
    if p_z is None:
        return None

    if feature == "positive_upslope":
        pressure_feature = np.maximum(np.gradient(p_z), 0)
        pressure_feature = zscore_signal(pressure_feature)
        if pressure_feature is None:
            pressure_feature = p_z
    else:
        pressure_feature = p_z

    r_idx = np.searchsorted(t, r_waves["r_time_s"].to_numpy(float))
    max_lag_n = max(1, int(round(float(max_lag_s) / dt)))

    best_lag = None
    best_score = -np.inf
    best_count = 0
    for lag_n in range(-max_lag_n, max_lag_n + 1):
        idx = r_idx + lag_n
        valid = (
            (idx >= 0)
            & (idx < len(pressure_feature))
            & (t[np.clip(idx, 0, len(t) - 1)] >= start_s)
            & (t[np.clip(idx, 0, len(t) - 1)] <= end_s)
        )
        if valid.sum() < 2:
            continue
        vals = pressure_feature[idx[valid]]
        vals = vals[np.isfinite(vals)]
        if len(vals) < 2:
            continue
        score = float(np.nanmean(vals))
        if score > best_score:
            best_score = score
            best_lag = lag_n
            best_count = len(vals)

    if best_lag is None:
        return None

    return {
        "start_s": float(start_s),
        "end_s": float(end_s),
        "lag_samples": int(best_lag),
        "lag_ms": float(best_lag * dt * 1000.0),
        "score": float(best_score),
        "r_peaks": int(len(r_waves)),
        "peaks_used": int(best_count),
    }


def pressure_ecg_lag_qc(
    time_s: np.ndarray,
    pressure: np.ndarray,
    ecg: np.ndarray,
    pressure_col: str,
    ecg_col: str,
    window_s: float = 2.5,
    step_s: float = 1.0,
    max_lag_s: float = 1.0,
    feature: str = "positive_upslope",
):
    t = np.asarray(time_s, dtype=float)
    if len(t) < 10:
        return None, []

    duration = float(np.nanmax(t) - np.nanmin(t))
    whole = pressure_ecg_lag_window(t, pressure, ecg, float(np.nanmin(t)), float(np.nanmax(t)), max_lag_s, feature)

    windows = []
    if duration >= window_s:
        starts = np.arange(float(np.nanmin(t)), float(np.nanmax(t)) - window_s + step_s * 0.25, step_s)
        for start in starts:
            end = min(float(start + window_s), float(np.nanmax(t)))
            result = pressure_ecg_lag_window(t, pressure, ecg, float(start), end, max_lag_s, feature)
            if result is not None:
                windows.append(result)

    valid_lags = np.array([w["lag_ms"] for w in windows if np.isfinite(w["lag_ms"])], dtype=float)
    summary = {
        "pressure": pressure_col,
        "same_file_ekg": ecg_col,
        "whole_strip_lag_ms": whole["lag_ms"] if whole else np.nan,
        "valid_windows": int(len(valid_lags)),
        "median_window_lag_ms": float(np.nanmedian(valid_lags)) if len(valid_lags) else np.nan,
        "min_window_lag_ms": float(np.nanmin(valid_lags)) if len(valid_lags) else np.nan,
        "max_window_lag_ms": float(np.nanmax(valid_lags)) if len(valid_lags) else np.nan,
        "lag_range_ms": float(np.nanmax(valid_lags) - np.nanmin(valid_lags)) if len(valid_lags) else np.nan,
        "status": "stable" if len(valid_lags) >= 2 and (np.nanmax(valid_lags) - np.nanmin(valid_lags)) <= 80 else "review",
    }
    return summary, windows


def add_ecg_timing_metrics(
    stats: pd.DataFrame,
    r_waves: pd.DataFrame,
    r_wave_source_file: str = "",
    r_wave_source_signal: str = "",
    timing_signal_cols: list | None = None,
):
    """
    Add ECG timing metrics to stats table using detected R waves.
    For matching interval/signal rows, relate the morphology peak to nearby R waves.
    """
    if stats.empty:
        return stats

    out = stats.copy()
    timing_signal_set = set(timing_signal_cols or [])
    if timing_signal_set and "signal" in out.columns:
        timed_rows = out["signal"].isin(timing_signal_set)
    else:
        timed_rows = pd.Series(True, index=out.index)

    if r_waves.empty or "excess_peak_time_s" not in stats.columns:
        out["previous_r_time_s"] = np.nan
        out["next_r_time_s"] = np.nan
        out["qrs_to_excess_peak_ms"] = np.nan
        out["rr_cycle_length_ms"] = np.nan
        out["cycle_normalized_excess_peak_phase"] = np.nan
        out["r_wave_source_file"] = np.where(timed_rows, r_wave_source_file, "")
        out["r_wave_source_signal"] = np.where(timed_rows, r_wave_source_signal, "")
        return out

    r_times = r_waves["r_time_s"].to_numpy(float)

    prev_r = []
    next_r = []
    qrs_to_peak_ms = []
    cycle_length_ms = []
    cycle_normalized_phase = []

    for _, row in out.iterrows():
        if timing_signal_set and row.get("signal") not in timing_signal_set:
            prev_r.append(np.nan)
            next_r.append(np.nan)
            qrs_to_peak_ms.append(np.nan)
            cycle_length_ms.append(np.nan)
            cycle_normalized_phase.append(np.nan)
            continue

        peak_t = float(row["excess_peak_time_s"])
        before = r_times[r_times <= peak_t]
        after = r_times[r_times > peak_t]

        pr = before[-1] if len(before) else np.nan
        nr = after[0] if len(after) else np.nan

        prev_r.append(pr)
        next_r.append(nr)

        if np.isfinite(pr):
            qrs_to_peak_ms.append((peak_t - pr) * 1000.0)
        else:
            qrs_to_peak_ms.append(np.nan)

        if np.isfinite(pr) and np.isfinite(nr) and nr > pr:
            cycle_ms = (nr - pr) * 1000.0
            cycle_length_ms.append(cycle_ms)
            cycle_normalized_phase.append((peak_t - pr) / (nr - pr))
        else:
            cycle_length_ms.append(np.nan)
            cycle_normalized_phase.append(np.nan)

    out["previous_r_time_s"] = prev_r
    out["next_r_time_s"] = next_r
    out["qrs_to_excess_peak_ms"] = qrs_to_peak_ms
    out["rr_cycle_length_ms"] = cycle_length_ms
    out["cycle_normalized_excess_peak_phase"] = cycle_normalized_phase
    out["r_wave_source_file"] = np.where(timed_rows, r_wave_source_file, "")
    out["r_wave_source_signal"] = np.where(timed_rows, r_wave_source_signal, "")
    return out



def get_data_dictionary():
    """
    Data dictionary for exported waveform metrics.
    """
    rows = [
        {
            "category": "Identifier / metadata",
            "variable": "patient_id",
            "definition": "User-entered patient/case identifier.",
            "calculation": "Entered in the Case metadata sidebar.",
            "units": "text",
            "interpretation": "Used to link waveform intervals and stats to a specific patient/case.",
            "recommended_for_vwave_analysis": "Essential",
        },
        {
            "category": "Identifier / metadata",
            "variable": "procedure_date",
            "definition": "User-entered date of the procedure.",
            "calculation": "Entered in the Case metadata sidebar.",
            "units": "date/text",
            "interpretation": "Useful for longitudinal studies or chart validation.",
            "recommended_for_vwave_analysis": "Essential",
        },
        {
            "category": "Identifier / metadata",
            "variable": "case_notes",
            "definition": "Free-text notes about the case.",
            "calculation": "Entered in the Case metadata sidebar.",
            "units": "text",
            "interpretation": "Can include rhythm, MR severity, diagnosis, wedge quality, or artifact notes.",
            "recommended_for_vwave_analysis": "Helpful",
        },
        {
            "category": "Interval label",
            "variable": "interval_id",
            "definition": "Numeric identifier for a labeled interval.",
            "calculation": "Sequential number assigned when an interval is added.",
            "units": "integer",
            "interpretation": "Links rows from the same selected waveform interval.",
            "recommended_for_vwave_analysis": "Essential",
        },
        {
            "category": "Interval label",
            "variable": "interval_label",
            "definition": "Name assigned to the selected interval.",
            "calculation": "Auto-generated as PatientID_vwave_N or manually entered.",
            "units": "text",
            "interpretation": "Example: HH10994_vwave_1, HH10994_vwave_2, End_expiratory_PCWP.",
            "recommended_for_vwave_analysis": "Essential",
        },
        {
            "category": "Interval timing",
            "variable": "start_s",
            "definition": "Start time of the analyzed interval.",
            "calculation": "Minimum time included in the selected interval.",
            "units": "seconds",
            "interpretation": "Start of the user-selected waveform frame.",
            "recommended_for_vwave_analysis": "Essential",
        },
        {
            "category": "Interval timing",
            "variable": "end_s",
            "definition": "End time of the analyzed interval.",
            "calculation": "Maximum time included in the selected interval.",
            "units": "seconds",
            "interpretation": "End of the user-selected waveform frame.",
            "recommended_for_vwave_analysis": "Essential",
        },
        {
            "category": "Interval timing",
            "variable": "duration_s",
            "definition": "Duration of the selected interval.",
            "calculation": "end_s - start_s.",
            "units": "seconds",
            "interpretation": "Used to normalize AUC and compare intervals of different length.",
            "recommended_for_vwave_analysis": "Essential",
        },
        {
            "category": "Signal",
            "variable": "signal",
            "definition": "Waveform channel analyzed.",
            "calculation": "Column name from aligned pressure data, for example RA_mmHg, RV_mmHg, PA_mmHg, or PCWP_mmHg.",
            "units": "text",
            "interpretation": "Identifies the pressure signal used for interval morphology statistics.",
            "recommended_for_vwave_analysis": "Essential",
        },
        {
            "category": "Conventional statistics",
            "variable": "n_points",
            "definition": "Number of sampled data points in the selected interval.",
            "calculation": "Count of finite waveform values within interval.",
            "units": "samples",
            "interpretation": "Small values may indicate very narrow interval or missing data.",
            "recommended_for_vwave_analysis": "QC",
        },
        {
            "category": "Conventional statistics",
            "variable": "min",
            "definition": "Minimum waveform value within the interval.",
            "calculation": "min(signal).",
            "units": "mmHg",
            "interpretation": "Lowest pressure in selected frame.",
            "recommended_for_vwave_analysis": "Helpful",
        },
        {
            "category": "Conventional statistics",
            "variable": "p05",
            "definition": "5th percentile waveform value.",
            "calculation": "5th percentile of signal values.",
            "units": "mmHg",
            "interpretation": "Robust low-end value less sensitive to single-sample noise.",
            "recommended_for_vwave_analysis": "Helpful",
        },
        {
            "category": "Conventional statistics",
            "variable": "mean",
            "definition": "Mean waveform value within selected interval.",
            "calculation": "mean(signal).",
            "units": "mmHg",
            "interpretation": "Average pressure over the selected frame.",
            "recommended_for_vwave_analysis": "Essential covariate",
        },
        {
            "category": "Conventional statistics",
            "variable": "median",
            "definition": "Median waveform value within selected interval.",
            "calculation": "median(signal).",
            "units": "mmHg",
            "interpretation": "Robust central pressure in the selected frame.",
            "recommended_for_vwave_analysis": "Helpful",
        },
        {
            "category": "Conventional statistics",
            "variable": "p95",
            "definition": "95th percentile waveform value.",
            "calculation": "95th percentile of signal values.",
            "units": "mmHg",
            "interpretation": "Robust high-end value less sensitive to single-sample spikes.",
            "recommended_for_vwave_analysis": "Helpful",
        },
        {
            "category": "Conventional statistics",
            "variable": "max",
            "definition": "Maximum waveform value within the interval.",
            "calculation": "max(signal).",
            "units": "mmHg",
            "interpretation": "Absolute peak pressure.",
            "recommended_for_vwave_analysis": "Helpful",
        },
        {
            "category": "Conventional statistics",
            "variable": "std",
            "definition": "Standard deviation of waveform values.",
            "calculation": "Sample standard deviation of signal.",
            "units": "mmHg",
            "interpretation": "Overall variability within selected frame.",
            "recommended_for_vwave_analysis": "QC / secondary",
        },
        {
            "category": "Baseline",
            "variable": "baseline_start",
            "definition": "Waveform value at the beginning of the selected interval.",
            "calculation": "First signal value in the interval.",
            "units": "mmHg",
            "interpretation": "Ideally the foot of the V wave if manually selected well.",
            "recommended_for_vwave_analysis": "Essential",
        },
        {
            "category": "Baseline",
            "variable": "baseline_end",
            "definition": "Waveform value at the end of the selected interval.",
            "calculation": "Last signal value in the interval.",
            "units": "mmHg",
            "interpretation": "End local baseline after the V wave returns.",
            "recommended_for_vwave_analysis": "Essential",
        },
        {
            "category": "Baseline",
            "variable": "horizontal_baseline_start_end_mean",
            "definition": "Horizontal baseline defined by start/end average.",
            "calculation": "(baseline_start + baseline_end) / 2.",
            "units": "mmHg",
            "interpretation": "Flat reference baseline for simple baseline-corrected AUC.",
            "recommended_for_vwave_analysis": "Secondary",
        },
        {
            "category": "AUC",
            "variable": "raw_auc_to_zero",
            "definition": "Total area under waveform relative to y = 0.",
            "calculation": "Integral of raw signal over time using trapezoidal rule.",
            "units": "mmHg·s",
            "interpretation": "Pressure burden. Strongly affected by baseline pressure and interval duration; not a pure morphology metric.",
            "recommended_for_vwave_analysis": "Secondary / covariate",
        },
        {
            "category": "AUC",
            "variable": "auc_above_start_baseline",
            "definition": "Area above the interval starting value.",
            "calculation": "Integral of signal - baseline_start over time.",
            "units": "mmHg·s",
            "interpretation": "Useful if interval starts exactly at the V-wave foot; sensitive to start-point choice.",
            "recommended_for_vwave_analysis": "Secondary",
        },
        {
            "category": "AUC",
            "variable": "auc_above_horizontal_baseline",
            "definition": "Area above a horizontal baseline defined by the start/end average.",
            "calculation": "Integral of signal - horizontal_baseline_start_end_mean over time.",
            "units": "mmHg·s",
            "interpretation": "Baseline-corrected area assuming no local drift.",
            "recommended_for_vwave_analysis": "Secondary",
        },
        {
            "category": "AUC",
            "variable": "excess_auc_linear_baseline_net",
            "definition": "Net area relative to a straight line connecting interval start and end.",
            "calculation": "Integral of signal - linear_start_to_end_baseline over time.",
            "units": "mmHg·s",
            "interpretation": "Net deviation from local trend. Positive and negative regions can cancel.",
            "recommended_for_vwave_analysis": "Secondary",
        },
        {
            "category": "AUC",
            "variable": "excess_auc_linear_baseline_positive",
            "definition": "Positive area above the straight line connecting interval start and end.",
            "calculation": "Integral of max(signal - linear_start_to_end_baseline, 0) over time.",
            "units": "mmHg·s",
            "interpretation": "Primary V-wave area morphology metric; captures the V-wave bulge above local baseline.",
            "recommended_for_vwave_analysis": "Primary",
        },
        {
            "category": "AUC",
            "variable": "excess_auc_linear_baseline_negative",
            "definition": "Negative area below the straight line connecting interval start and end.",
            "calculation": "Integral of min(signal - linear_start_to_end_baseline, 0) over time.",
            "units": "mmHg·s",
            "interpretation": "Amount of waveform falling below local baseline; usually a secondary/QC metric.",
            "recommended_for_vwave_analysis": "Secondary / QC",
        },
        {
            "category": "AUC",
            "variable": "normalized_positive_excess_auc",
            "definition": "Positive excess AUC normalized by duration.",
            "calculation": "excess_auc_linear_baseline_positive / duration_s.",
            "units": "mmHg",
            "interpretation": "Average excess pressure above local baseline; useful when intervals have different durations.",
            "recommended_for_vwave_analysis": "Primary",
        },
        {
            "category": "Peak morphology",
            "variable": "peak_value",
            "definition": "Maximum raw waveform value in the selected interval.",
            "calculation": "max(signal).",
            "units": "mmHg",
            "interpretation": "Absolute V-wave peak pressure, but confounded by baseline pressure.",
            "recommended_for_vwave_analysis": "Secondary",
        },
        {
            "category": "Peak morphology",
            "variable": "peak_time_s",
            "definition": "Time of maximum raw waveform value.",
            "calculation": "Time corresponding to max(signal).",
            "units": "seconds",
            "interpretation": "Timing of raw peak within the strip.",
            "recommended_for_vwave_analysis": "Secondary",
        },
        {
            "category": "Peak morphology",
            "variable": "trough_value",
            "definition": "Minimum raw waveform value in the selected interval.",
            "calculation": "min(signal).",
            "units": "mmHg",
            "interpretation": "Lowest value in interval; useful for QC and amplitude context.",
            "recommended_for_vwave_analysis": "QC / secondary",
        },
        {
            "category": "Peak morphology",
            "variable": "trough_time_s",
            "definition": "Time of minimum raw waveform value.",
            "calculation": "Time corresponding to min(signal).",
            "units": "seconds",
            "interpretation": "Timing of trough within the strip.",
            "recommended_for_vwave_analysis": "QC / secondary",
        },
        {
            "category": "Peak morphology",
            "variable": "linear_baseline_at_excess_peak",
            "definition": "Value of local linear baseline at the excess peak time.",
            "calculation": "Straight start-end baseline evaluated at excess_peak_time_s.",
            "units": "mmHg",
            "interpretation": "Expected local baseline at the V-wave peak time.",
            "recommended_for_vwave_analysis": "Helpful",
        },
        {
            "category": "Peak morphology",
            "variable": "peak_above_linear_baseline",
            "definition": "Maximum height above the linear start-end baseline.",
            "calculation": "max(signal - linear_start_to_end_baseline).",
            "units": "mmHg",
            "interpretation": "Primary V-wave amplitude metric; less dependent on absolute wedge/RA pressure.",
            "recommended_for_vwave_analysis": "Primary",
        },
        {
            "category": "Peak morphology",
            "variable": "excess_peak_time_s",
            "definition": "Time when the waveform reaches maximal height above the linear baseline.",
            "calculation": "Time corresponding to max(signal - linear_start_to_end_baseline).",
            "units": "seconds",
            "interpretation": "Timing of the morphology-corrected V-wave peak.",
            "recommended_for_vwave_analysis": "Primary for timing",
        },
        {
            "category": "Peak morphology",
            "variable": "time_to_peak_s",
            "definition": "Time from interval start to raw peak.",
            "calculation": "peak_time_s - start_s.",
            "units": "seconds",
            "interpretation": "How quickly the raw waveform reaches its maximum.",
            "recommended_for_vwave_analysis": "Secondary",
        },
        {
            "category": "Peak morphology",
            "variable": "time_to_excess_peak_s",
            "definition": "Time from interval start to excess peak.",
            "calculation": "excess_peak_time_s - start_s.",
            "units": "seconds",
            "interpretation": "How quickly the V wave rises above local baseline.",
            "recommended_for_vwave_analysis": "Primary",
        },
        {
            "category": "Slope / shape",
            "variable": "rise_slope_raw_units_per_s",
            "definition": "Raw rise slope from interval start to raw peak.",
            "calculation": "(peak_value - baseline_start) / (peak_time_s - start_s).",
            "units": "mmHg/s",
            "interpretation": "Raw upstroke steepness; confounded by baseline drift.",
            "recommended_for_vwave_analysis": "Secondary",
        },
        {
            "category": "Slope / shape",
            "variable": "fall_slope_raw_units_per_s",
            "definition": "Raw fall slope from raw peak to interval end.",
            "calculation": "(baseline_end - peak_value) / (end_s - peak_time_s).",
            "units": "mmHg/s",
            "interpretation": "Raw downstroke steepness; usually negative if returning downward.",
            "recommended_for_vwave_analysis": "Secondary",
        },
        {
            "category": "Slope / shape",
            "variable": "excess_rise_slope_units_per_s",
            "definition": "Rise slope of the excess waveform above local baseline.",
            "calculation": "peak_above_linear_baseline / time_to_excess_peak_s.",
            "units": "mmHg/s",
            "interpretation": "Steepness of V-wave upstroke. High values may suggest abrupt compliance-limited pressure rise.",
            "recommended_for_vwave_analysis": "Primary",
        },
        {
            "category": "Slope / shape",
            "variable": "excess_fall_slope_units_per_s",
            "definition": "Fall slope of the excess waveform back toward baseline.",
            "calculation": "(0 - peak_above_linear_baseline) / (end_s - excess_peak_time_s).",
            "units": "mmHg/s",
            "interpretation": "Steepness of V-wave descent. More negative values indicate faster decline.",
            "recommended_for_vwave_analysis": "Primary",
        },
        {
            "category": "Slope / shape",
            "variable": "fwhm_excess_s",
            "definition": "Full width at half maximum of the excess waveform.",
            "calculation": "Duration where signal - linear_baseline is at least 50% of peak_above_linear_baseline.",
            "units": "seconds",
            "interpretation": "V-wave width. Narrower waves have smaller FWHM; broader waves have larger FWHM.",
            "recommended_for_vwave_analysis": "Primary",
        },
        {
            "category": "Slope / shape",
            "variable": "symmetry_index_excess_peak",
            "definition": "Relative timing of excess peak within selected interval.",
            "calculation": "time_to_excess_peak_s / duration_s.",
            "units": "unitless",
            "interpretation": "0.5 is symmetric; <0.5 early peak; >0.5 late peak.",
            "recommended_for_vwave_analysis": "Primary",
        },
        {
            "category": "Shape ratio",
            "variable": "peak_to_mean_ratio",
            "definition": "Raw peak relative to mean signal.",
            "calculation": "peak_value / mean.",
            "units": "unitless",
            "interpretation": "Shape/spikiness metric, but affected by baseline pressure.",
            "recommended_for_vwave_analysis": "Secondary",
        },
        {
            "category": "Shape ratio",
            "variable": "excess_peak_to_mean_positive_excess_ratio",
            "definition": "Excess peak relative to average positive excess.",
            "calculation": "peak_above_linear_baseline / mean(max(signal - linear_baseline, 0)).",
            "units": "unitless",
            "interpretation": "How spiky the V wave is relative to its positive excess area.",
            "recommended_for_vwave_analysis": "Secondary",
        },
        {
            "category": "Composite index",
            "variable": "vwave_sharpness_index",
            "definition": "Height-to-width index for the V wave.",
            "calculation": "peak_above_linear_baseline / fwhm_excess_s.",
            "units": "mmHg/s",
            "interpretation": "Higher values indicate taller/narrower sharper V waves; candidate amyloid/restrictive physiology metric.",
            "recommended_for_vwave_analysis": "Primary",
        },
        {
            "category": "Composite index",
            "variable": "area_density_index",
            "definition": "Positive V-wave area concentrated over wave width.",
            "calculation": "excess_auc_linear_baseline_positive / fwhm_excess_s.",
            "units": "mmHg",
            "interpretation": "Higher values indicate more concentrated excess V-wave pressure area.",
            "recommended_for_vwave_analysis": "Primary",
        },
        {
            "category": "Composite index",
            "variable": "relative_vwave_amplitude",
            "definition": "V-wave amplitude relative to mean pressure.",
            "calculation": "peak_above_linear_baseline / mean.",
            "units": "unitless",
            "interpretation": "Amplitude normalized to local pressure level; useful across different baseline wedge pressures.",
            "recommended_for_vwave_analysis": "Primary",
        },
        {
            "category": "Composite index",
            "variable": "relative_vwave_amplitude_to_median",
            "definition": "V-wave amplitude relative to median pressure.",
            "calculation": "peak_above_linear_baseline / median.",
            "units": "unitless",
            "interpretation": "Robust alternative to relative_vwave_amplitude.",
            "recommended_for_vwave_analysis": "Secondary",
        },
        {
            "category": "Composite index",
            "variable": "vwave_burden_ratio",
            "definition": "Fraction of raw interval AUC attributable to positive V-wave excess.",
            "calculation": "excess_auc_linear_baseline_positive / raw_auc_to_zero.",
            "units": "unitless",
            "interpretation": "How much of pressure burden is V-wave bulge rather than baseline pressure.",
            "recommended_for_vwave_analysis": "Primary",
        },
        {
            "category": "Composite index",
            "variable": "slope_area_ratio",
            "definition": "Rise steepness relative to average positive excess area.",
            "calculation": "excess_rise_slope_units_per_s / normalized_positive_excess_auc.",
            "units": "1/s",
            "interpretation": "Higher values suggest abrupt rise for a given average excess pressure.",
            "recommended_for_vwave_analysis": "Exploratory",
        },
        {
            "category": "Composite index",
            "variable": "rise_to_fall_slope_ratio",
            "definition": "Relative steepness of upstroke vs downstroke.",
            "calculation": "abs(excess_rise_slope_units_per_s) / abs(excess_fall_slope_units_per_s).",
            "units": "unitless",
            "interpretation": "Values >1 mean rise is steeper than fall; values <1 mean fall is steeper.",
            "recommended_for_vwave_analysis": "Exploratory",
        },
        {
            "category": "ECG timing",
            "variable": "r_wave_source_file",
            "definition": "PW6 file whose embedded ECG was used for R-wave detection.",
            "calculation": "The ECG trace embedded in the same PW6 file as the analyzed PCWP/PW pressure channel.",
            "units": "text",
            "interpretation": "Confirms that ECG timing metrics use the PCWP/PW same-file ECG rather than WAV_000 or another strip.",
            "recommended_for_vwave_analysis": "QC",
        },
        {
            "category": "ECG timing",
            "variable": "r_wave_source_signal",
            "definition": "ECG lead used for R-wave detection.",
            "calculation": "Preferred lead from the PCWP/PW file ECG, usually DII/II when available.",
            "units": "text",
            "interpretation": "Documents the exact ECG channel used for QRS-to-V-wave timing.",
            "recommended_for_vwave_analysis": "QC",
        },
        {
            "category": "ECG timing",
            "variable": "previous_r_time_s",
            "definition": "Time of preceding detected R wave before the excess peak.",
            "calculation": "Nearest detected R wave at or before excess_peak_time_s in the ECG embedded in the PCWP/PW pressure file.",
            "units": "seconds",
            "interpretation": "Reference QRS timing for V-wave peak.",
            "recommended_for_vwave_analysis": "Primary if ECG quality adequate",
        },
        {
            "category": "ECG timing",
            "variable": "next_r_time_s",
            "definition": "Time of next detected R wave after the excess peak.",
            "calculation": "Nearest detected R wave after excess_peak_time_s in the ECG embedded in the PCWP/PW pressure file.",
            "units": "seconds",
            "interpretation": "Used to estimate cardiac cycle length.",
            "recommended_for_vwave_analysis": "Primary if ECG quality adequate",
        },
        {
            "category": "ECG timing",
            "variable": "qrs_to_excess_peak_ms",
            "definition": "Delay from preceding QRS/R wave to excess V-wave peak.",
            "calculation": "(excess_peak_time_s - previous_r_time_s) × 1000.",
            "units": "milliseconds",
            "interpretation": "Timing of V-wave peak relative to cardiac cycle.",
            "recommended_for_vwave_analysis": "Primary if ECG quality adequate",
        },
        {
            "category": "ECG timing",
            "variable": "rr_cycle_length_ms",
            "definition": "R-R interval around the excess peak.",
            "calculation": "(next_r_time_s - previous_r_time_s) × 1000.",
            "units": "milliseconds",
            "interpretation": "Local cardiac cycle length.",
            "recommended_for_vwave_analysis": "Covariate / QC",
        },
        {
            "category": "ECG timing",
            "variable": "cycle_normalized_excess_peak_phase",
            "definition": "V-wave peak timing normalized to local cardiac cycle.",
            "calculation": "(excess_peak_time_s - previous_r_time_s) / (next_r_time_s - previous_r_time_s).",
            "units": "unitless, 0–1",
            "interpretation": "0 = at R wave; 1 = next R wave. Helps compare timing across different heart rates.",
            "recommended_for_vwave_analysis": "Primary if ECG quality adequate",
        },
    ]

    return pd.DataFrame(rows)


def schematic_waveform():
    t = np.linspace(0, 6.0, 240)
    baseline = 18.0 + 0.55 * t
    v_wave = 21.0 * np.exp(-0.5 * ((t - 3.25) / 0.55) ** 2)
    y = baseline + v_wave + 1.1 * np.sin(2 * np.pi * t / 2.2)
    linear_baseline = np.interp(t, [t[0], t[-1]], [y[0], y[-1]])
    return t, y, baseline, linear_baseline


def apply_dictionary_figure_style(fig, title: str):
    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor="left", font=dict(size=15)),
        height=300,
        margin=dict(l=38, r=18, t=48, b=38),
        showlegend=True,
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
        hovermode=False,
    )
    fig.update_xaxes(
        title_text="Time",
        showgrid=True,
        gridcolor="rgba(65, 65, 65, 0.18)",
        zeroline=False,
    )
    fig.update_yaxes(
        title_text="Pressure",
        showgrid=True,
        gridcolor="rgba(65, 65, 65, 0.18)",
        zeroline=False,
    )
    return fig


def auc_to_zero_cartoon():
    t, y, _, _ = schematic_waveform()
    zero = np.zeros_like(t)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=zero, mode="lines", line=dict(color="rgba(80,80,80,0.65)", width=1), name="Zero line"))
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([t, t[::-1]]),
            y=np.concatenate([y, zero[::-1]]),
            fill="toself",
            fillcolor="rgba(58, 134, 255, 0.22)",
            line=dict(color="rgba(58, 134, 255, 0)"),
            name="AUC to zero",
        )
    )
    fig.add_trace(go.Scatter(x=t, y=y, mode="lines", line=dict(color="rgb(20,20,20)", width=2.4), name="Pressure waveform"))
    fig.add_annotation(x=3.0, y=8.0, text="Total pressure-time burden", showarrow=False, font=dict(size=13))
    return apply_dictionary_figure_style(fig, "raw_auc_to_zero")


def auc_to_baseline_cartoon():
    t, y, baseline, linear_baseline = schematic_waveform()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=baseline, mode="lines", line=dict(color="rgb(16, 130, 92)", width=2, dash="dash"), name="Start/baseline reference"))
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([t, t[::-1]]),
            y=np.concatenate([y, baseline[::-1]]),
            fill="toself",
            fillcolor="rgba(16, 185, 129, 0.24)",
            line=dict(color="rgba(16, 185, 129, 0)"),
            name="Area above baseline",
        )
    )
    fig.add_trace(go.Scatter(x=t, y=linear_baseline, mode="lines", line=dict(color="rgb(245, 125, 40)", width=2, dash="dot"), name="Linear local baseline"))
    fig.add_trace(go.Scatter(x=t, y=y, mode="lines", line=dict(color="rgb(20,20,20)", width=2.4), name="Pressure waveform"))
    fig.add_annotation(x=3.3, y=34.0, text="V-wave excess area", showarrow=True, ax=40, ay=-40, font=dict(size=13))
    return apply_dictionary_figure_style(fig, "auc_above_baseline / excess_auc_linear_baseline_positive")


def slope_peak_cartoon():
    t, y, _, linear_baseline = schematic_waveform()
    excess = y - linear_baseline
    peak_idx = int(np.nanargmax(excess))
    peak_t = float(t[peak_idx])
    peak_y = float(y[peak_idx])
    start_t = 2.1
    end_t = 4.65
    start_y = float(np.interp(start_t, t, y))
    end_y = float(np.interp(end_t, t, y))
    half_height = float(np.interp(peak_t, t, linear_baseline) + excess[peak_idx] / 2.0)
    above_half = np.where(y >= half_height)[0]
    left_t = float(t[above_half[0]])
    right_t = float(t[above_half[-1]])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=linear_baseline, mode="lines", line=dict(color="rgb(130,130,130)", width=2, dash="dot"), name="Linear baseline"))
    fig.add_trace(go.Scatter(x=t, y=y, mode="lines", line=dict(color="rgb(20,20,20)", width=2.4), name="Pressure waveform"))
    fig.add_trace(go.Scatter(x=[start_t, peak_t], y=[start_y, peak_y], mode="lines+markers", line=dict(color="rgb(220, 38, 38)", width=3), name="Rise slope"))
    fig.add_trace(go.Scatter(x=[peak_t, end_t], y=[peak_y, end_y], mode="lines+markers", line=dict(color="rgb(37, 99, 235)", width=3), name="Fall slope"))
    fig.add_trace(go.Scatter(x=[left_t, right_t], y=[half_height, half_height], mode="lines", line=dict(color="rgb(124, 58, 237)", width=4), name="FWHM"))
    fig.add_annotation(x=peak_t, y=peak_y, text="Peak above baseline", showarrow=True, ax=30, ay=-35, font=dict(size=13))
    fig.add_annotation(x=(left_t + right_t) / 2, y=half_height, text="Width at half max", showarrow=True, ax=0, ay=35, font=dict(size=13))
    return apply_dictionary_figure_style(fig, "peak, slope, and width metrics")


def ecg_timing_cartoon():
    t = np.linspace(0, 6.0, 600)
    ecg = np.zeros_like(t)
    r_times = np.array([1.0, 2.25, 3.5, 4.75])
    for rt in r_times:
        ecg += 1.0 * np.exp(-0.5 * ((t - rt) / 0.025) ** 2)
    pressure = 0.25 + 0.35 * np.sin(2 * np.pi * t / 2.4) + 1.1 * np.exp(-0.5 * ((t - 3.88) / 0.34) ** 2)
    pressure = pressure + 1.8
    peak_t = float(t[np.argmax(pressure)])
    prev_r = float(r_times[r_times <= peak_t][-1])
    next_r = float(r_times[r_times > peak_t][0])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=ecg + 4.0, mode="lines", line=dict(color="rgb(37, 99, 235)", width=1.8), name="Same-file ECG"))
    fig.add_trace(go.Scatter(x=t, y=pressure, mode="lines", line=dict(color="rgb(20,20,20)", width=2.4), name="PCWP pressure"))
    for rt in r_times:
        fig.add_vline(x=rt, line_color="rgba(37, 99, 235, 0.45)", line_dash="dot", line_width=1)
    fig.add_vrect(x0=prev_r, x1=peak_t, fillcolor="rgba(245, 158, 11, 0.22)", line_width=0)
    fig.add_vrect(x0=prev_r, x1=next_r, fillcolor="rgba(37, 99, 235, 0.08)", line_width=0)
    fig.add_annotation(x=(prev_r + peak_t) / 2, y=3.05, text="QRS to excess peak", showarrow=False, font=dict(size=13))
    fig.add_annotation(x=(prev_r + next_r) / 2, y=4.9, text="RR cycle for normalized phase", showarrow=False, font=dict(size=13))
    return apply_dictionary_figure_style(fig, "qrs_to_excess_peak_ms / cycle_normalized_excess_peak_phase")


def render_visual_data_dictionary():
    st.subheader("Graphic explanations")
    st.caption("Small schematic cartoons for the main waveform-shape concepts. These are illustrations, not patient data.")

    with st.expander("AUC and baseline concepts", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(auc_to_zero_cartoon(), width="stretch", key="visual_dictionary_auc_zero")
            st.caption("AUC to zero is the total area between the pressure trace and y = 0. It reflects overall pressure burden, so baseline pressure strongly affects it.")
        with c2:
            st.plotly_chart(auc_to_baseline_cartoon(), width="stretch", key="visual_dictionary_auc_baseline")
            st.caption("Baseline-corrected AUC asks how much extra area the wave adds above a chosen local baseline. This is closer to V-wave morphology.")

    with st.expander("Peak, slope, width, and ECG timing", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(slope_peak_cartoon(), width="stretch", key="visual_dictionary_slope_peak")
            st.caption("Peak height, rise/fall slope, and FWHM describe how tall, steep, and broad the selected wave is.")
        with c2:
            st.plotly_chart(ecg_timing_cartoon(), width="stretch", key="visual_dictionary_ecg_timing")
            st.caption("ECG timing metrics relate the PCWP morphology peak to same-file R waves, then optionally normalize by the RR cycle.")


def get_recommended_feature_set():
    """
    Suggested feature set for amyloid/restrictive CM vs HFrEF V-wave morphology comparison.
    """
    return pd.DataFrame(
        [
            {
                "priority": 1,
                "feature": "vwave_sharpness_index",
                "reason": "Captures tall/narrow abrupt V waves expected in low-compliance physiology.",
            },
            {
                "priority": 2,
                "feature": "peak_above_linear_baseline",
                "reason": "Best baseline-corrected V-wave amplitude metric.",
            },
            {
                "priority": 3,
                "feature": "excess_auc_linear_baseline_positive",
                "reason": "Best baseline-corrected V-wave area metric.",
            },
            {
                "priority": 4,
                "feature": "normalized_positive_excess_auc",
                "reason": "Area metric normalized for interval duration.",
            },
            {
                "priority": 5,
                "feature": "area_density_index",
                "reason": "Captures how concentrated the V-wave excess area is.",
            },
            {
                "priority": 6,
                "feature": "relative_vwave_amplitude",
                "reason": "V-wave amplitude normalized to mean local pressure.",
            },
            {
                "priority": 7,
                "feature": "fwhm_excess_s",
                "reason": "Captures narrow vs broad V-wave morphology.",
            },
            {
                "priority": 8,
                "feature": "excess_rise_slope_units_per_s",
                "reason": "Steepness of V-wave upstroke.",
            },
            {
                "priority": 9,
                "feature": "excess_fall_slope_units_per_s",
                "reason": "Steepness of V-wave descent.",
            },
            {
                "priority": 10,
                "feature": "qrs_to_excess_peak_ms",
                "reason": "V-wave peak timing relative to QRS.",
            },
            {
                "priority": 11,
                "feature": "cycle_normalized_excess_peak_phase",
                "reason": "Heart-rate-normalized V-wave timing within the cardiac cycle.",
            },
            {
                "priority": 12,
                "feature": "mean",
                "reason": "Important covariate to distinguish morphology from baseline pressure burden.",
            },
        ]
    )


# -----------------------------
# Streamlit app
# -----------------------------

st.set_page_config(page_title="Hemodynamic RHC Viewer", layout="wide")

require_password_if_configured()

render_institutional_header()
st.title("Hemodynamic RHC Viewer")
st.caption(
    "Upload Philips Xper .PW6 files, align EKG/pressure strips by time, select intervals with synchronized cursors, "
    "label intervals, choose visible channels, customize axes, calculate stats, and export labeled segments."
)
st.caption(f"{APP_VERSION} · {APP_VERSION_DATE}")

if "database_path" not in st.session_state:
    st.session_state.database_path = str(default_database_path())
if "case_reset_nonce" not in st.session_state:
    st.session_state.case_reset_nonce = 0
if "last_upload_signature" not in st.session_state:
    st.session_state.last_upload_signature = None
if "pending_restored_interval_set" not in st.session_state:
    st.session_state.pending_restored_interval_set = None
if "applied_restored_interval_key" not in st.session_state:
    st.session_state.applied_restored_interval_key = None


def reset_case_state():
    st.session_state.case_reset_nonce += 1
    st.session_state.intervals = []
    st.session_state.last_upload_signature = None
    st.session_state.pending_restored_interval_set = None
    st.session_state.applied_restored_interval_key = None


def cursor_keys(case_key):
    return (
        f"cursor_slider_{case_key}",
        f"cursor_a_{case_key}",
        f"cursor_b_{case_key}",
    )


def set_cursor_window(case_key, start_s, end_s, duration_s):
    slider_key, cursor_a_key, cursor_b_key = cursor_keys(case_key)
    start_s = min(max(float(start_s), 0.0), float(duration_s))
    end_s = min(max(float(end_s), 0.0), float(duration_s))
    lo, hi = sorted([start_s, end_s])
    st.session_state[slider_key] = (lo, hi)
    st.session_state[cursor_a_key] = lo
    st.session_state[cursor_b_key] = hi


def sync_cursor_from_slider(case_key, duration_s):
    slider_key, _, _ = cursor_keys(case_key)
    start_s, end_s = st.session_state.get(slider_key, (0.50, min(1.00, float(duration_s))))
    set_cursor_window(case_key, start_s, end_s, duration_s)


def sync_cursor_from_inputs(case_key, duration_s):
    _, cursor_a_key, cursor_b_key = cursor_keys(case_key)
    set_cursor_window(
        case_key,
        st.session_state.get(cursor_a_key, 0.50),
        st.session_state.get(cursor_b_key, min(1.00, float(duration_s))),
        duration_s,
    )


def normalize_interval_records(intervals: list, duration_s: float | None = None):
    normalized = []
    for item in intervals:
        label = str(item.get("label", "")).strip() or "Interval"
        start_s = float(item.get("start_s", 0.0))
        end_s = float(item.get("end_s", start_s))
        if duration_s is not None:
            start_s = min(max(start_s, 0.0), float(duration_s))
            end_s = min(max(end_s, 0.0), float(duration_s))
        lo, hi = sorted([start_s, end_s])
        normalized.append({"label": label, "start_s": lo, "end_s": hi})
    return normalized


def saved_interval_sets(db_rows: pd.DataFrame):
    required = {"saved_at", "patient_id", "source_files", "interval_id", "interval_label", "interval_start_s", "interval_end_s"}
    if db_rows.empty or not required.issubset(db_rows.columns):
        return pd.DataFrame()
    interval_sets = db_rows.copy()
    if "procedure_date" not in interval_sets.columns:
        interval_sets["procedure_date"] = ""
    return (
        interval_sets[
            ["saved_at", "patient_id", "procedure_date", "source_files", "interval_id", "interval_label", "interval_start_s", "interval_end_s"]
        ]
        .drop_duplicates()
        .sort_values(["saved_at", "patient_id", "interval_id"], ascending=[False, True, True])
        .reset_index(drop=True)
    )


def interval_set_options(interval_sets: pd.DataFrame):
    if interval_sets.empty:
        return []
    grouped = (
        interval_sets.groupby(["saved_at", "patient_id", "procedure_date", "source_files"], dropna=False)
        .agg(interval_count=("interval_id", "nunique"))
        .reset_index()
        .sort_values("saved_at", ascending=False)
    )
    return grouped.to_dict("records")


viewer_tab, database_tab, dictionary_tab = st.tabs(["Waveform viewer", "Database", "Data dictionary"])

with database_tab:
    st.subheader("Accumulated interval database")
    st.caption("Local SQLite database with interval statistics plus raw waveform samples for visual replay.")

    db_path_input = st.text_input(
        "Database file",
        value=st.session_state.database_path,
        key="database_path_input",
    )
    st.session_state.database_path = db_path_input
    db_path = Path(db_path_input).expanduser()

    db_segments = load_database_segments(db_path)
    summary = database_summary(db_path)
    m1, m2, m3 = st.columns(3)
    m1.metric("Rows", summary["rows"])
    m2.metric("Cases", summary["cases"])
    m3.metric("Intervals", summary["intervals"])
    st.metric("Saved waveform samples", len(db_segments))

    db_rows = load_database_rows(db_path)
    if db_rows.empty:
        st.info("No saved intervals yet. Label intervals in the Waveform viewer, then use Save labeled interval stats to database.")
    else:
        search_db = st.text_input("Search database", value="", key="database_search").strip().lower()
        db_view = db_rows.copy()
        if search_db:
            mask = db_view.apply(lambda row: search_db in " ".join(row.astype(str)).lower(), axis=1)
            db_view = db_view[mask]
        st.dataframe(db_view, width="stretch", hide_index=True)
        st.download_button(
            "Download database table CSV",
            db_rows.to_csv(index=False).encode("utf-8"),
            "xper_hemo_accumulated_interval_stats.csv",
            "text/csv",
        )

        interval_sets = saved_interval_sets(db_rows)
        options = interval_set_options(interval_sets)
        if options:
            st.subheader("Restore saved interval set")
            st.caption(
                "Loads a saved set of interval labels/times back into the Waveform viewer. "
                "Upload the same PW6 files there, then the shaded selections can be edited and re-saved."
            )
            selected_restore_idx = st.selectbox(
                "Saved case/session",
                options=list(range(len(options))),
                format_func=lambda i: (
                    f"{options[i]['saved_at']} | {options[i]['patient_id']} | "
                    f"{options[i]['interval_count']} interval(s)"
                ),
                key="restore_interval_set_select",
            )
            selected_restore = options[selected_restore_idx]
            restore_rows = interval_sets[
                (interval_sets["saved_at"] == selected_restore["saved_at"])
                & (interval_sets["patient_id"] == selected_restore["patient_id"])
                & (interval_sets["source_files"] == selected_restore["source_files"])
            ].copy()
            st.dataframe(
                restore_rows[["interval_id", "interval_label", "interval_start_s", "interval_end_s"]],
                width="stretch",
                hide_index=True,
            )
            if st.button("Load these intervals into Waveform viewer", type="primary", width="stretch"):
                restored_intervals = [
                    {
                        "label": row["interval_label"],
                        "start_s": float(row["interval_start_s"]),
                        "end_s": float(row["interval_end_s"]),
                    }
                    for _, row in restore_rows.sort_values("interval_id").iterrows()
                ]
                st.session_state.pending_restored_interval_set = {
                    "key": f"{selected_restore['saved_at']}|{selected_restore['patient_id']}|{selected_restore['source_files']}",
                    "intervals": normalize_interval_records(restored_intervals),
                    "patient_id": selected_restore["patient_id"],
                    "procedure_date": "" if pd.isna(selected_restore["procedure_date"]) else str(selected_restore["procedure_date"]),
                    "source_files": selected_restore["source_files"],
                }
                st.success("Saved intervals are ready. Open the Waveform viewer and upload the matching PW6 files.")

    if not db_segments.empty:
        st.subheader("Saved waveform segment preview")
        interval_keys = (
            db_segments[
                ["saved_at", "patient_id", "interval_id", "interval_label", "interval_start_s", "interval_end_s"]
            ]
            .drop_duplicates()
            .sort_values(["saved_at", "patient_id", "interval_id"], ascending=[False, True, True])
            .reset_index(drop=True)
        )
        selected_idx = st.selectbox(
            "Saved segment",
            options=list(range(len(interval_keys))),
            format_func=lambda i: (
                f"{interval_keys.loc[i, 'saved_at']} | {interval_keys.loc[i, 'patient_id']} | "
                f"{interval_keys.loc[i, 'interval_label']} "
                f"({float(interval_keys.loc[i, 'interval_start_s']):.3f}-{float(interval_keys.loc[i, 'interval_end_s']):.3f} s)"
            ),
            key="saved_segment_preview_select",
        )
        selected_key = interval_keys.loc[selected_idx]
        preview = db_segments[
            (db_segments["saved_at"] == selected_key["saved_at"])
            & (db_segments["patient_id"] == selected_key["patient_id"])
            & (db_segments["interval_id"] == selected_key["interval_id"])
            & (db_segments["interval_label"] == selected_key["interval_label"])
        ].copy()
        st.plotly_chart(
            labeled_interval_figure(preview, title=f"Saved segment: {selected_key['interval_label']}"),
            width="stretch",
            key="saved_segment_preview_plot",
        )
        with st.expander("Saved raw samples for this interval"):
            st.dataframe(preview, width="stretch", hide_index=True)
        st.download_button(
            "Download saved segment raw CSV",
            preview.to_csv(index=False).encode("utf-8"),
            "saved_labeled_waveform_segment.csv",
            "text/csv",
        )

with dictionary_tab:
    st.subheader("Data dictionary")
    st.caption("Definitions for exported waveform morphology, AUC, composite, and ECG-timing variables.")

    render_visual_data_dictionary()

    dd = get_data_dictionary()
    search_term = st.text_input("Search variables", value="", key="dictionary_search").strip().lower()
    category_options = ["All"] + sorted(dd["category"].unique().tolist())
    selected_category = st.selectbox("Filter by category", category_options, key="dictionary_category")

    dd_view = dd.copy()
    if selected_category != "All":
        dd_view = dd_view[dd_view["category"] == selected_category]
    if search_term:
        mask = dd_view.apply(lambda row: search_term in " ".join(row.astype(str)).lower(), axis=1)
        dd_view = dd_view[mask]

    st.dataframe(dd_view, width="stretch", hide_index=True)

    st.download_button(
        "Download data dictionary CSV",
        dd.to_csv(index=False).encode("utf-8"),
        "xper_hemo_data_dictionary.csv",
        "text/csv",
    )

    st.subheader("Recommended feature set for amyloid/restrictive CM vs HFrEF")
    recommended = get_recommended_feature_set()
    st.dataframe(recommended, width="stretch", hide_index=True)

    st.download_button(
        "Download recommended feature set CSV",
        recommended.to_csv(index=False).encode("utf-8"),
        "xper_hemo_recommended_vwave_features.csv",
        "text/csv",
    )

with viewer_tab:
    st.subheader("Waveform viewer")

    if "intervals" not in st.session_state:
        st.session_state.intervals = []

    with st.sidebar:
        case_key = st.session_state.case_reset_nonce

        if st.button("Start new subject / clear uploads", key=f"reset_case_{case_key}"):
            reset_case_state()
            st.rerun()

        st.header("Upload")
        uploaded_files = st.file_uploader(
            "Upload PW6 files plus optional PDF/JPEG/PNG reference outputs",
            type=["PW6", "pw6", "pdf", "PDF", "jpg", "jpeg", "png", "JPG", "JPEG", "PNG"],
            accept_multiple_files=True,
            key=f"pw6_upload_{case_key}",
            help="Select the PW6 files and, if available, the exported Xper waveform PDF/JPEG/PNG files for the same patient/case.",
        )
        show_reference_outputs = st.checkbox(
            "Open PCWP reference PDF/image panel when uploaded",
            value=True,
            key=f"show_reference_outputs_{case_key}",
        )

        duration_s = st.number_input("Strip duration (seconds)", min_value=1.0, max_value=30.0, value=7.0, step=0.5, key=f"duration_s_{case_key}")
        fs = st.number_input("Aligned display sampling rate (Hz)", min_value=50, max_value=1000, value=500, step=50, key=f"fs_{case_key}")

        ecg_layout_label = st.selectbox(
            "ECG layout",
            options=[
                "Automatic: WAV_000 is 6-lead; pressure strips are D I / D II / D III",
                "3 sequential limb leads: D I, D II, D III",
                "6 sequential limb leads: I, II, III, aVR, aVL, aVF",
                "Single ECG trace",
            ],
            index=0,
            help="Automatic mode keeps pressure-strip ECG aligned with pressure channels and treats WAV_000 as a separate 6-lead ECG file.",
            key=f"ecg_layout_{case_key}",
        )
        ecg_layout = {
            "Automatic: WAV_000 is 6-lead; pressure strips are D I / D II / D III": "automatic_by_file_type",
            "3 sequential limb leads: D I, D II, D III": "3_lead",
            "6 sequential limb leads: I, II, III, aVR, aVL, aVF": "6_lead",
            "Single ECG trace": "single",
        }[ecg_layout_label]

        pending_restore = st.session_state.get("pending_restored_interval_set")
        if pending_restore:
            if f"patient_id_{case_key}" not in st.session_state and pending_restore.get("patient_id"):
                st.session_state[f"patient_id_{case_key}"] = str(pending_restore["patient_id"])
            if f"procedure_date_{case_key}" not in st.session_state and pending_restore.get("procedure_date"):
                st.session_state[f"procedure_date_{case_key}"] = str(pending_restore["procedure_date"])

        st.header("Case metadata")
        patient_id = st.text_input("Patient / Case ID", value="", key=f"patient_id_{case_key}").strip()
        procedure_date = st.text_input("Procedure date", value="", key=f"procedure_date_{case_key}").strip()
        notes = st.text_area("Notes", value="", height=80, key=f"notes_{case_key}")

        st.header("Database")
        st.session_state.database_path = st.text_input(
            "SQLite database file",
            value=st.session_state.database_path,
            key="sidebar_database_path",
        )

        st.header("Dual synchronized cursors")
        st.caption("These two cursors are displayed across all waveforms simultaneously.")
        cursor_slider_key, cursor_a_key, cursor_b_key = cursor_keys(case_key)
        if cursor_slider_key not in st.session_state:
            set_cursor_window(case_key, 0.50, min(1.00, float(duration_s)), duration_s)
        else:
            current_start, current_end = st.session_state.get(cursor_slider_key, (0.50, min(1.00, float(duration_s))))
            set_cursor_window(case_key, current_start, current_end, duration_s)

        st.slider(
            "Cursor A / Cursor B time window (s)",
            min_value=0.0,
            max_value=float(duration_s),
            step=0.01,
            key=cursor_slider_key,
            on_change=sync_cursor_from_slider,
            args=(case_key, float(duration_s)),
        )

        cA, cB = st.columns(2)
        with cA:
            st.number_input(
                "Cursor A (s)",
                min_value=0.0,
                max_value=float(duration_s),
                step=0.01,
                format="%.3f",
                key=cursor_a_key,
                on_change=sync_cursor_from_inputs,
                args=(case_key, float(duration_s)),
            )
        with cB:
            st.number_input(
                "Cursor B (s)",
                min_value=0.0,
                max_value=float(duration_s),
                step=0.01,
                format="%.3f",
                key=cursor_b_key,
                on_change=sync_cursor_from_inputs,
                args=(case_key, float(duration_s)),
            )
        cursor_start = float(st.session_state[cursor_a_key])
        cursor_end = float(st.session_state[cursor_b_key])

        st.header("Label selected interval")
        auto_vwave_labels = st.checkbox("Auto-name intervals as PatientID_vwave_N", value=True, key=f"auto_vwave_labels_{case_key}")

        default_interval_label = next_vwave_label(patient_id, st.session_state.intervals) if auto_vwave_labels else f"Interval {len(st.session_state.intervals) + 1}"
        interval_label = st.text_input("Interval label", value=default_interval_label, key=f"interval_label_{case_key}")

        add_col, clear_col = st.columns(2)
        with add_col:
            add_interval = st.button("Add V wave label" if auto_vwave_labels else "Add label")
        with clear_col:
            clear_intervals = st.button("Clear labels")

        if clear_intervals:
            st.session_state.intervals = []
            st.rerun()

    if not uploaded_files:
        st.info("Upload PW6 files to begin. Suggested set: ECG/WAV_000 if available plus RA, RV, PA, and PCWP/PW files. Optional Xper PDF/JPEG/PNG outputs can be selected at the same time.")
        st.stop()

    uploaded = [up for up in uploaded_files if Path(up.name).suffix.lower() == ".pw6"]
    reference_uploads = [up for up in uploaded_files if is_reference_file(up.name)]

    if not uploaded:
        st.error("No PW6 files were uploaded. Select the patient/case PW6 files along with any optional reference PDFs or images.")
        st.stop()

    upload_signature = tuple(sorted((up.name, getattr(up, "size", None)) for up in uploaded_files))
    if st.session_state.last_upload_signature != upload_signature:
        st.session_state.intervals = []
        st.session_state.last_upload_signature = upload_signature

    pending_restore = st.session_state.get("pending_restored_interval_set")
    if pending_restore and st.session_state.applied_restored_interval_key != pending_restore.get("key"):
        restored_intervals = normalize_interval_records(pending_restore.get("intervals", []), duration_s=float(duration_s))
        st.session_state.intervals = restored_intervals
        st.session_state.applied_restored_interval_key = pending_restore.get("key")
        saved_sources = {
            s.strip()
            for s in str(pending_restore.get("source_files", "")).split(";")
            if s.strip()
        }
        uploaded_sources = {up.name for up in uploaded}
        missing_sources = sorted(saved_sources - uploaded_sources)
        if missing_sources:
            st.warning(
                "Restored saved intervals, but these original source files are not currently uploaded: "
                + ", ".join(missing_sources)
            )
        else:
            st.success("Restored saved interval selections into the Waveform viewer. You can edit, remove, add, and re-save them.")

    records = []
    ecg_candidates = []
    alignment_rows = []
    reference_files = [
        {
            "name": up.name,
            "ext": Path(up.name).suffix.lower(),
            "data": up.getvalue(),
        }
        for up in reference_uploads
    ]

    for up in uploaded:
        data = up.getvalue()
        label = resolve_strip_label(up.name, data)

        source_ecg_layout = resolve_ecg_layout(up.name, ecg_layout)
        ecg_df = extract_ecg_from_run0(data, duration_s=duration_s, layout=source_ecg_layout)
        if ecg_df is not None:
            ecg_candidates.append((up.name, label, ecg_df))

        if is_dedicated_ecg_file(up.name):
            pressure_values, pressure_df = None, pd.DataFrame()
        else:
            pressure_values, pressure_df = extract_pressure_waveform(data, duration_s=duration_s)
        if pressure_df is not None and not pressure_df.empty and label != "ECG":
            records.append({"filename": up.name, "label": label, "df": pressure_df, "ecg_df": ecg_df})

        ecg_lead_count = 0
        ecg_samples_per_lead = 0
        if ecg_df is not None:
            ecg_lead_count = len([c for c in ecg_df.columns if c.startswith("EKG_") and c.endswith("_mV")])
            ecg_samples_per_lead = len(ecg_df)
        pressure_first_valid_s = first_valid_time(pressure_df, "pressure_mmHg")
        alignment_rows.append(
            {
                "filename": up.name,
                "detected_label": label,
                "ecg_layout_used": source_ecg_layout,
                "ecg_leads": ecg_lead_count,
                "ecg_samples_per_lead": ecg_samples_per_lead,
                "ecg_effective_hz_if_7s": float(ecg_samples_per_lead / duration_s) if duration_s else np.nan,
                "pressure_samples": len(pressure_df) if pressure_df is not None and not pressure_df.empty else 0,
                "pressure_first_valid_s": pressure_first_valid_s,
                "pressure_effective_hz_if_7s": float(len(pressure_df) / duration_s) if pressure_df is not None and not pressure_df.empty and duration_s else np.nan,
                "same_file_ecg_pressure": bool(ecg_df is not None and pressure_df is not None and not pressure_df.empty),
            }
        )

    records = sorted(records, key=lambda r: pressure_sort_key(r["label"], r["filename"]))
    ecg_candidates = sorted(ecg_candidates, key=ecg_source_sort_key)

    if not records:
        st.error("No pressure waveform could be extracted from the uploaded PW6 files.")
        st.stop()

    with st.expander("Detected pressure channels", expanded=False):
        if patient_id:
            st.caption(f"Current Patient / Case ID: {patient_id}")
        if reference_files:
            st.caption(f"Uploaded reference output file(s): {', '.join(ref['name'] for ref in reference_files)}")

        detected = pd.DataFrame(
            [
                {
                    "filename": r["filename"],
                    "detected_label": r["label"],
                    "n_samples": len(r["df"]),
                    "first_valid_s": first_valid_time(r["df"], "pressure_mmHg"),
                    "min": float(r["df"]["pressure_mmHg"].min()),
                    "mean": float(r["df"]["pressure_mmHg"].mean()),
                    "max": float(r["df"]["pressure_mmHg"].max()),
                }
                for r in records
            ]
        )
        st.dataframe(detected, width="stretch")

        st.markdown("**Alignment / time-base QC**")
        st.caption(
            "All displayed signals are resampled onto a common 0-to-strip-duration x-axis. "
            "However, ECG and pressure are truly simultaneous only when they come from the same PW6 file. "
            "For PCWP/V-wave timing, prefer the ECG source from the PW/PCWP file when available."
        )
        st.dataframe(pd.DataFrame(alignment_rows), width="stretch", hide_index=True)

    channel_options = ["Ignore", "RA", "RV", "PA", "PCWP", "AO", "LV", "Other"]
    mapped_records = []
    with st.expander("Channel mapping", expanded=False):
        st.caption("Review and correct labels if needed. Internal Xper labels are usually correct, but some cases use different WAV numbering.")
        cols = st.columns(min(4, len(records)))
        for i, r in enumerate(records):
            with cols[i % len(cols)]:
                default_label = "PCWP" if r["label"] == "PW" else r["label"]
                default_index = channel_options.index(default_label) if default_label in channel_options else channel_options.index("Other")
                chosen = st.selectbox(
                    f"{r['filename']}",
                    channel_options,
                    index=default_index,
                    key=f"map_{i}_{r['filename']}",
                )
                if chosen != "Ignore":
                    mapped_records.append({"filename": r["filename"], "label": chosen, "df": r["df"], "ecg_df": r.get("ecg_df")})

    with st.expander("EKG source", expanded=False):
        if ecg_candidates:
            chosen_ecg_idx = st.selectbox(
                "Choose ECG source; use WAV_000 if available, otherwise a pressure strip ECG trace",
                options=list(range(len(ecg_candidates))),
                format_func=lambda i: describe_ecg_candidate(ecg_candidates[i]),
                index=0,
            )
            chosen_ecg = ecg_candidates[chosen_ecg_idx][2]
            chosen_ecg_name = ecg_candidates[chosen_ecg_idx][0]
            chosen_ecg_label = ecg_candidates[chosen_ecg_idx][1]

            ecg_cols = [c for c in chosen_ecg.columns if c.startswith("EKG_") and c.endswith("_mV")]
            default_ecg = preferred_ecg_column(ecg_cols)
            ecg_col = st.selectbox("EKG channel", options=ecg_cols, index=ecg_cols.index(default_ecg))
            st.caption(
                f"Selected ECG source: {chosen_ecg_name} ({chosen_ecg_label}). "
                "This ECG is time-matched to the pressure waveform from the same file; other pressure files may be separate strips."
            )
        else:
            chosen_ecg = None
            ecg_col = None

    # Align data
    time_grid = np.arange(0, float(duration_s), 1 / float(fs))
    aligned = pd.DataFrame({"time_s": time_grid})
    pressure_source_by_col = {}
    same_file_ecg_by_pressure_col = {}

    if chosen_ecg is not None:
        aligned[ecg_col] = interpolate_to_grid(chosen_ecg, ecg_col, time_grid)

    for r in mapped_records:
        out_col = f"{r['label']}_mmHg"
        # If duplicate labels are present, preserve both with filenames.
        if out_col in aligned.columns:
            suffix = Path(r["filename"]).stem.replace(" ", "_")
            out_col = f"{r['label']}_{suffix}_mmHg"
        aligned[out_col] = interpolate_to_grid(r["df"], "pressure_mmHg", time_grid)
        pressure_source_by_col[out_col] = r

        source_ecg = r.get("ecg_df")
        if source_ecg is not None and not source_ecg.empty:
            source_ecg_cols = [c for c in source_ecg.columns if c.startswith("EKG_") and c.endswith("_mV")]
            source_ecg_col = preferred_overlay_ecg_column(source_ecg_cols)
            if source_ecg_col:
                source_ecg_values_by_col = {
                    c: interpolate_to_grid(source_ecg, c, time_grid)
                    for c in source_ecg_cols
                }
                same_file_ecg_by_pressure_col[out_col] = {
                    "filename": r["filename"],
                    "ecg_col": source_ecg_col,
                    "values": source_ecg_values_by_col[source_ecg_col],
                    "values_by_col": source_ecg_values_by_col,
                    "samples": len(source_ecg),
                }

    signal_cols = sorted(
        [c for c in aligned.columns if c != "time_s"],
        key=lambda c: (9, c) if "EKG" in c.upper() else pressure_sort_key(c),
    )

    if not signal_cols:
        st.error("No channels selected for display.")
        st.stop()

    pressure_cols = sorted([c for c in signal_cols if "EKG" not in c], key=pressure_sort_key)
    pcwp_signal_cols = [c for c in pressure_cols if "PCWP" in c.upper()]
    if not pcwp_signal_cols:
        pcwp_signal_cols = [c for c in pressure_cols if re.search(r"(^|_)PW(_|$)", c.upper())]
    analysis_signal_cols = pressure_cols

    default_display_cols = signal_cols.copy()
    with st.expander("Channel visibility", expanded=False):
        st.caption("Choose which channels to display. All selected/labeled interval exports still include all mapped channels unless you choose otherwise below.")

        display_signal_cols = st.multiselect(
            "Waveforms to display",
            options=signal_cols,
            default=default_display_cols,
            key=f"display_signal_cols_{case_key}",
        )

        export_display_only = st.checkbox(
            "Export only displayed channels",
            value=False,
            help="For full waveform export only. Labeled interval statistics include all mapped pressure channels.",
        )

        st.caption(
            "Interval extraction/statistics include all mapped pressure channels: "
            + (", ".join(analysis_signal_cols) if analysis_signal_cols else "no pressure channel is currently mapped")
        )

    if not display_signal_cols:
        st.warning("Select at least one waveform channel to display.")
        st.stop()

    displayed_pressure_cols = sorted([c for c in display_signal_cols if c in pressure_cols], key=pressure_sort_key)
    time_shift_ms_by_col = {
        c: int(st.session_state.get(f"time_shift_{case_key}_{c}", 0))
        for c in pressure_cols
    }

    for c, shift_ms in time_shift_ms_by_col.items():
        if shift_ms:
            aligned[c] = shift_aligned_values(time_grid, aligned[c].to_numpy(float), float(shift_ms) / 1000.0)

    st.subheader("Axis controls")
    st.caption("Optionally customize x-axis and y-axis ranges for visualization. Leave automatic ranges on for default behavior.")

    use_custom_x_axis = st.checkbox("Customize x-axis range", value=False)
    if use_custom_x_axis:
        x_axis_min, x_axis_max = st.slider(
            "Displayed x-axis range (s)",
            min_value=0.0,
            max_value=float(duration_s),
            value=(0.0, float(duration_s)),
            step=0.01,
            key="x_axis_range",
        )
    else:
        x_axis_min, x_axis_max = 0.0, float(duration_s)

    use_global_y_axis = st.checkbox(
        "Use one shared y-axis range for pressure channels",
        value=False,
        help="Useful when comparing pressure amplitudes across RA/RV/PA/PCWP. ECG keeps its own axis.",
    )

    custom_y_ranges = {}
    if use_global_y_axis:
        pressure_cols = sorted([c for c in display_signal_cols if "EKG" not in c], key=pressure_sort_key)
        if pressure_cols:
            global_min = float(np.nanmin(aligned[pressure_cols].to_numpy()))
            global_max = float(np.nanmax(aligned[pressure_cols].to_numpy()))
            pad = max(1.0, (global_max - global_min) * 0.10)
            y_global_min = st.number_input("Pressure y-axis minimum (mmHg)", value=float(global_min - pad), step=1.0)
            y_global_max = st.number_input("Pressure y-axis maximum (mmHg)", value=float(global_max + pad), step=1.0)
        else:
            y_global_min, y_global_max = None, None
    else:
        y_global_min, y_global_max = None, None

    use_custom_y_axis = st.checkbox("Customize y-axis range by channel", value=False)
    if use_custom_y_axis:
        st.caption("Set y-axis limits for each displayed channel.")
        for c in display_signal_cols:
            y = aligned[c].to_numpy(dtype=float)
            y_min_default = float(np.nanmin(y))
            y_max_default = float(np.nanmax(y))
            pad = max(0.1, (y_max_default - y_min_default) * 0.10)
            cc1, cc2 = st.columns(2)
            with cc1:
                ymin = st.number_input(f"{c} y-min", value=float(y_min_default - pad), step=0.5, key=f"ymin_{c}")
            with cc2:
                ymax = st.number_input(f"{c} y-max", value=float(y_max_default + pad), step=0.5, key=f"ymax_{c}")
            custom_y_ranges[c] = (float(ymin), float(ymax))

    with st.expander("EKG-pressure lag QC"):
        st.caption(
            "Estimates where pressure features best follow same-file EKG R waves. "
            "A stable non-zero lag is expected from physiology and catheter transmission; "
            "large lag changes across windows suggest timing drift or a mismatched extraction."
        )
        lag_qc_enabled = st.checkbox("Compute experimental same-file lag QC", value=False, key=f"lag_qc_enabled_{case_key}")
        lag_feature_label = st.selectbox(
            "Pressure timing feature",
            ["Positive pressure upslope", "Pressure amplitude"],
            index=0,
            key=f"lag_qc_feature_{case_key}",
        )
        lag_window_s = st.number_input(
            "Lag QC window length (s)",
            min_value=1.0,
            max_value=float(duration_s),
            value=min(4.0, float(duration_s)),
            step=0.5,
            key=f"lag_qc_window_{case_key}",
        )
        lag_max_ms = st.number_input(
            "Maximum lag search (ms)",
            min_value=100,
            max_value=1500,
            value=700,
            step=50,
            key=f"lag_qc_max_{case_key}",
        )
        if lag_qc_enabled and same_file_ecg_by_pressure_col:
            feature_key = "positive_upslope" if lag_feature_label == "Positive pressure upslope" else "pressure_amplitude"
            lag_summaries = []
            lag_windows = []
            for pressure_col in pressure_cols:
                if pressure_col not in same_file_ecg_by_pressure_col:
                    continue
                source_ecg = same_file_ecg_by_pressure_col[pressure_col]
                summary, windows = pressure_ecg_lag_qc(
                    aligned["time_s"].to_numpy(float),
                    aligned[pressure_col].to_numpy(float),
                    source_ecg["values"],
                    pressure_col=pressure_col,
                    ecg_col=source_ecg["ecg_col"],
                    window_s=float(lag_window_s),
                    step_s=max(0.5, float(lag_window_s) / 2.0),
                    max_lag_s=float(lag_max_ms) / 1000.0,
                    feature=feature_key,
                )
                if summary is not None:
                    lag_summaries.append(summary)
                    for row in windows:
                        row = {"pressure": pressure_col, "same_file_ekg": source_ecg["ecg_col"], **row}
                        lag_windows.append(row)

            if lag_summaries:
                st.dataframe(pd.DataFrame(lag_summaries), width="stretch", hide_index=True)
                if lag_windows:
                    st.dataframe(pd.DataFrame(lag_windows), width="stretch", hide_index=True)
            else:
                st.info("Lag QC could not detect enough R waves and pressure features in the same-file EKG traces.")
        elif lag_qc_enabled:
            st.info("No same-file EKG traces are available for lag QC.")

    # ECG timing features must use the ECG embedded in the PCWP/PW pressure PW6 file.
    r_wave_pressure_col, r_wave_source = pcwp_r_wave_source(pcwp_signal_cols, same_file_ecg_by_pressure_col)
    if r_wave_source is not None:
        r_waves, detected_pcwp_ecg_col = detect_r_waves_from_ecg_source(aligned["time_s"].to_numpy(), r_wave_source)
        r_wave_source_file = r_wave_source["filename"]
        r_wave_source_signal = detected_pcwp_ecg_col or r_wave_source["ecg_col"]
        if not r_waves.empty:
            r_waves.insert(0, "source_pressure_signal", r_wave_pressure_col)
            r_waves.insert(1, "source_file", r_wave_source_file)
            r_waves.insert(2, "source_ecg_signal", r_wave_source_signal)
    else:
        r_waves = pd.DataFrame(columns=["r_time_s", "r_value"])
        r_wave_source_file = ""
        r_wave_source_signal = ""

    # Add interval after data exist, so labels persist for current case.
    if add_interval:
        lo, hi = sorted([float(cursor_start), float(cursor_end)])
        final_label = next_vwave_label(patient_id, st.session_state.intervals) if auto_vwave_labels else interval_label
        st.session_state.intervals.append({"label": final_label, "start_s": lo, "end_s": hi})
        st.rerun()

    lo, hi = sorted([float(cursor_start), float(cursor_end)])
    segment = get_segment(aligned, cursor_start, cursor_end)
    stats = calculate_stats(segment, analysis_signal_cols)
    stats = add_ecg_timing_metrics(stats, r_waves, r_wave_source_file, r_wave_source_signal, timing_signal_cols=pcwp_signal_cols)

    labeled_segments, labeled_stats = build_labeled_exports(aligned, st.session_state.intervals, analysis_signal_cols)
    labeled_stats = add_ecg_timing_metrics(labeled_stats, r_waves, r_wave_source_file, r_wave_source_signal, timing_signal_cols=pcwp_signal_cols)
    raw_labeled_segments = build_labeled_raw_segments(aligned, st.session_state.intervals, signal_cols)

    # Plot
    st.subheader("Waveform viewer with synchronized dual cursors")
    st.caption("Each pressure waveform is paired with the EKG from the same PW6 file when available. The timing slider sits directly below the pressure it shifts.")

    major_grid = "rgba(65, 65, 65, 0.24)"
    minor_grid = "rgba(90, 90, 90, 0.16)"
    spike_style = dict(
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikecolor="rgba(0, 0, 0, 0.75)",
        spikedash="solid",
        spikethickness=1,
    )

    def decorate_waveform_figure(fig, bottom_row: int):
        if not r_waves.empty:
            for rt in r_waves["r_time_s"].to_list():
                fig.add_vline(x=rt, line_width=1, line_dash="dot", line_color="rgba(0,0,180,0.35)")

        fig.add_vrect(x0=lo, x1=hi, fillcolor="gray", opacity=0.15, line_width=0)
        fig.add_vline(x=lo, line_width=2, line_dash="dash", line_color="black")
        fig.add_vline(x=hi, line_width=2, line_dash="dash", line_color="black")

        for item in st.session_state.intervals:
            fig.add_vrect(
                x0=item["start_s"],
                x1=item["end_s"],
                fillcolor="LightSalmon",
                opacity=0.18,
                line_width=0,
            )
            fig.add_annotation(
                x=(item["start_s"] + item["end_s"]) / 2,
                y=1.02,
                xref="x",
                yref="paper",
                text=item["label"],
                showarrow=False,
                font=dict(size=11),
            )

        fig.update_xaxes(
            **spike_style,
            showgrid=True,
            gridcolor=major_grid,
            gridwidth=0.5,
            minor=dict(showgrid=True, gridcolor=minor_grid, gridwidth=0.5),
            zeroline=False,
        )
        fig.update_xaxes(title_text="Time (s)", row=bottom_row, col=1, rangeslider=dict(visible=False))
        if use_custom_x_axis:
            fig.update_xaxes(range=[float(x_axis_min), float(x_axis_max)])

        fig.update_yaxes(
            showgrid=True,
            gridcolor=major_grid,
            gridwidth=0.5,
            minor=dict(showgrid=True, gridcolor=minor_grid, gridwidth=0.5),
            zeroline=False,
        )

        fig.update_layout(
            height=max(280, 170 * bottom_row + 100),
            hovermode="x",
            legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5),
            margin=dict(t=70, b=95),
        )

    rendered_pressure_cols = sorted([c for c in display_signal_cols if c in displayed_pressure_cols], key=pressure_sort_key)

    for pressure_col in rendered_pressure_cols:
        pressure_label = pressure_col.replace("_mmHg", "")
        source_record = pressure_source_by_col.get(pressure_col, {})
        source_ecg = same_file_ecg_by_pressure_col.get(pressure_col)
        ecg_values = source_ecg["values"] if source_ecg is not None else None
        ecg_name = f"EKG from {pressure_label}"
        if ecg_values is None and chosen_ecg is not None and ecg_col in aligned.columns:
            ecg_values = aligned[ecg_col]
            ecg_name = f"EKG reference for {pressure_label}"

        rv_r_waves = pd.DataFrame()
        rv_derivative_analysis = None
        rv_r_wave_source_signal = ""
        rv_same_file_ecg_available = source_ecg is not None
        if is_rv_signal(pressure_col, source_record.get("filename", "")):
            if source_ecg is not None:
                rv_r_waves, rv_r_wave_source_signal = detect_r_waves_from_ecg_source(
                    aligned["time_s"].to_numpy(float),
                    source_ecg,
                )
            if not rv_r_waves.empty:
                rv_derivative_analysis = rv_single_beat_derivative_analysis(
                    aligned["time_s"].to_numpy(float),
                    aligned[pressure_col].to_numpy(float),
                    rv_r_waves["r_time_s"].to_numpy(float),
                )
                if rv_derivative_analysis is not None and rv_derivative_analysis[1].empty:
                    rv_derivative_analysis = None

        plot_rows = 2 if ecg_values is not None else 1
        subplot_titles = [ecg_name, f"Pressure from {pressure_label}"] if plot_rows == 2 else [f"Pressure from {pressure_label}"]
        pressure_row = plot_rows
        peak_fit_row = None
        dpdt_row = None
        d2pdt2_row = None
        if rv_derivative_analysis is not None:
            peak_fit_row = plot_rows + 1
            dpdt_row = plot_rows + 2
            d2pdt2_row = plot_rows + 3
            plot_rows += 3
            subplot_titles.extend(
                [
                    "RV beat peaks and first-derivative sine interpolation",
                    "First derivative method: dP/dt",
                    "Second derivative method: d2P/dt2",
                ]
            )

        fig = make_subplots(
            rows=plot_rows,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            subplot_titles=subplot_titles,
        )

        if ecg_values is not None:
            fig.add_trace(
                go.Scatter(
                    x=aligned["time_s"],
                    y=ecg_values,
                    mode="lines",
                    name=ecg_name,
                    line=dict(color="rgba(30, 90, 200, 0.75)", width=1.2),
                    hovertemplate=f"Time: %{{x:.3f}} s<br>{ecg_name}: %{{y:.3f}} mV<extra></extra>",
                ),
                row=1,
                col=1,
            )
            fig.update_yaxes(title_text="mV", row=1, col=1)

        fig.add_trace(
            go.Scatter(
                x=aligned["time_s"],
                y=aligned[pressure_col],
                mode="lines",
                name=f"Pressure from {pressure_label}",
                hovertemplate=f"Time: %{{x:.3f}} s<br>{pressure_col}: %{{y:.3f}} mmHg<extra></extra>",
            ),
            row=pressure_row,
            col=1,
        )

        if use_custom_y_axis and pressure_col in custom_y_ranges:
            ymin, ymax = custom_y_ranges[pressure_col]
            fig.update_yaxes(title_text="mmHg", range=[ymin, ymax], row=pressure_row, col=1)
        elif use_global_y_axis and y_global_min is not None and y_global_max is not None:
            fig.update_yaxes(title_text="mmHg", range=[y_global_min, y_global_max], row=pressure_row, col=1)
        else:
            fig.update_yaxes(title_text="mmHg", row=pressure_row, col=1)

        if rv_derivative_analysis is not None and peak_fit_row is not None and dpdt_row is not None and d2pdt2_row is not None:
            rv_derivatives, rv_events, rv_fits, rv_fit_samples = rv_derivative_analysis
            fig.add_trace(
                go.Scatter(
                    x=rv_derivatives["time_s"],
                    y=rv_derivatives["rv_pressure_smooth_mmHg"],
                    mode="lines",
                    name="RV pressure for beat peak/Piso review",
                    line=dict(color="rgba(20, 20, 20, 0.82)", width=1.7),
                    hovertemplate="Time: %{x:.3f} s<br>RV pressure: %{y:.2f} mmHg<extra></extra>",
                ),
                row=peak_fit_row,
                col=1,
            )
            peak_events = rv_events[rv_events["row"] == "rv_peak_fit"]
            measured_peaks = peak_events[peak_events["event"] == "RV pressure peak"]
            if not measured_peaks.empty:
                fig.add_trace(
                    go.Scatter(
                        x=measured_peaks["time_s"],
                        y=measured_peaks["value"],
                        mode="markers",
                        name="Measured RV peak",
                        marker=dict(color="rgb(220, 38, 38)", size=8, symbol="triangle-up"),
                        hovertemplate="Beat %{customdata}: measured RV peak<br>Time: %{x:.3f} s<br>Pressure: %{y:.2f} mmHg<extra></extra>",
                        customdata=measured_peaks["beat_id"],
                    ),
                    row=peak_fit_row,
                    col=1,
                )
            fit_limits = peak_events[peak_events["event"].isin(["IC onset 20% dP/dt max", "IR end 20% dP/dt min"])]
            if not fit_limits.empty:
                fig.add_trace(
                    go.Scatter(
                        x=fit_limits["time_s"],
                        y=fit_limits["value"],
                        mode="markers",
                        name="Sine interpolation limits",
                        marker=dict(color="rgb(37, 99, 235)", size=7, symbol="x"),
                        hovertemplate="%{text}<br>Beat %{customdata}<br>Time: %{x:.3f} s<br>Pressure: %{y:.2f} mmHg<extra></extra>",
                        text=fit_limits["event"],
                        customdata=fit_limits["beat_id"],
                    ),
                    row=peak_fit_row,
                    col=1,
                )
            if not rv_fit_samples.empty:
                for sample_range, sample_df in rv_fit_samples.groupby("range", sort=False):
                    fig.add_trace(
                        go.Scatter(
                            x=sample_df["time_s"],
                            y=sample_df["pressure_mmHg"],
                            mode="markers",
                            name=sample_range,
                            marker=dict(
                                color="rgba(37, 99, 235, 0.78)" if sample_range.startswith("IC") else "rgba(16, 185, 129, 0.78)",
                                size=4,
                                symbol="circle",
                            ),
                            hovertemplate=(
                                "%{text}<br>Beat %{customdata}<br>Time: %{x:.3f} s<br>"
                                "Pressure: %{y:.2f} mmHg<extra></extra>"
                            ),
                            text=sample_df["range"],
                            customdata=sample_df["beat_id"],
                        ),
                        row=peak_fit_row,
                        col=1,
                    )
            if not rv_fits.empty:
                for beat_id, fit_df in rv_fits.groupby("beat_id", sort=True):
                    fig.add_trace(
                        go.Scatter(
                            x=fit_df["time_s"],
                            y=fit_df["piso_fit_mmHg"],
                            mode="lines",
                            name=f"Beat {int(beat_id)} Piso sine interpolation",
                            line=dict(color="rgb(217, 70, 239)", width=2.6, dash="dash"),
                            hovertemplate=(
                                f"Beat {int(beat_id)} sine interpolation<br>Time: %{{x:.3f}} s<br>"
                                "Interpolated pressure: %{y:.2f} mmHg<extra></extra>"
                            ),
                            showlegend=int(beat_id) == int(rv_fits["beat_id"].min()),
                        ),
                        row=peak_fit_row,
                        col=1,
                    )
                piso_events = peak_events[peak_events["event"] == "Piso estimate"]
                if not piso_events.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=piso_events["time_s"],
                            y=piso_events["value"],
                            mode="markers",
                            name="Piso estimate",
                            marker=dict(color="rgb(217, 70, 239)", size=11, symbol="star"),
                            hovertemplate="Beat %{customdata}: Piso estimate<br>Time: %{x:.3f} s<br>Piso: %{y:.2f} mmHg<extra></extra>",
                            customdata=piso_events["beat_id"],
                        ),
                        row=peak_fit_row,
                        col=1,
                    )
            fig.update_yaxes(title_text="mmHg", row=peak_fit_row, col=1)

            fig.add_trace(
                go.Scatter(
                    x=rv_derivatives["time_s"],
                    y=rv_derivatives["rv_dpdt_mmHg_per_s"],
                    mode="lines",
                    name="RV dP/dt",
                    line=dict(color="rgb(16, 130, 92)", width=1.5),
                    hovertemplate="Time: %{x:.3f} s<br>dP/dt: %{y:.1f} mmHg/s<extra></extra>",
                ),
                row=dpdt_row,
                col=1,
            )
            first_events = rv_events[rv_events["row"] == "dpdt"]
            for _, event in first_events.iterrows():
                color = "rgb(220, 38, 38)" if event["event"] == "dP/dt max" else "rgb(37, 99, 235)"
                fig.add_trace(
                    go.Scatter(
                        x=[event["time_s"]],
                        y=[event["value"]],
                        mode="markers",
                        name=event["event"],
                        marker=dict(color=color, size=9, symbol="diamond"),
                        hovertemplate=(
                            f"Beat {int(event['beat_id'])}: {event['event']}<br>Time: %{{x:.3f}} s<br>"
                            "Value: %{y:.1f} mmHg/s<extra></extra>"
                        ),
                    ),
                    row=dpdt_row,
                    col=1,
                )
            fig.update_yaxes(title_text="mmHg/s", row=dpdt_row, col=1)

            fig.add_trace(
                go.Scatter(
                    x=rv_derivatives["time_s"],
                    y=rv_derivatives["rv_d2pdt2_mmHg_per_s2"],
                    mode="lines",
                    name="RV d2P/dt2",
                    line=dict(color="rgb(124, 58, 237)", width=1.5),
                    hovertemplate="Time: %{x:.3f} s<br>d2P/dt2: %{y:.1f} mmHg/s2<extra></extra>",
                ),
                row=d2pdt2_row,
                col=1,
            )
            second_events = rv_events[rv_events["row"] == "d2pdt2"]
            for _, event in second_events.iterrows():
                fig.add_trace(
                    go.Scatter(
                        x=[event["time_s"]],
                        y=[event["value"]],
                        mode="markers",
                        name=event["event"],
                        marker=dict(color="rgb(245, 125, 40)", size=9, symbol="circle"),
                        hovertemplate=(
                            f"Beat {int(event['beat_id'])}: {event['event']}<br>Time: %{{x:.3f}} s<br>"
                            "Value: %{y:.1f} mmHg/s2<extra></extra>"
                        ),
                    ),
                    row=d2pdt2_row,
                    col=1,
                )
            fig.update_yaxes(title_text="mmHg/s2", row=d2pdt2_row, col=1)

        decorate_waveform_figure(fig, bottom_row=plot_rows)
        st.plotly_chart(fig, width="stretch", key=f"plot_{case_key}_{pressure_col}")
        if is_rv_signal(pressure_col, source_record.get("filename", "")) and rv_derivative_analysis is None:
            if rv_same_file_ecg_available:
                st.warning(
                    "RV derivative landmarks need QRS-to-QRS beat windows from the ECG embedded in the same RV PW6 file. "
                    "No reliable R waves were detected in that RV same-file ECG, so beat-level dP/dt and d2P/dt2 markers were not added."
                )
            else:
                st.warning(
                    "RV derivative landmarks need the ECG embedded in the same RV PW6 file. A displayed fallback ECG may be visible, "
                    "but it was not used for RV beat windows because it may come from a different strip."
                )
        elif rv_derivative_analysis is not None:
            _, rv_events, rv_fits, _ = rv_derivative_analysis
            st.caption(
                "RV single-beat derivative view based on Bellofiore et al. 2017: each beat is analyzed from QRS/R wave to QRS/R wave. "
                "dP/dt max/min mark first-derivative IC/IR references; second-derivative minima mark candidate pulmonic valve opening/closing points. "
                "The RV peak/Piso row marks measured RV peaks, the IC/IR samples used for sine interpolation, 20% dP/dt interpolation limits, and the first-derivative sine interpolation used to estimate Piso. "
                f"Beat windows use the same-file RV ECG lead {rv_r_wave_source_signal or source_ecg.get('ecg_col', '')}. "
                "This is a visual feature-identification layer, not yet a final Piso/Ees calculation."
            )
            if not rv_fits.empty:
                fit_summary = (
                    rv_fits[["beat_id", "measured_peak_mmHg", "piso_mmHg", "piso_margin_mmHg", "piso_time_s", "fit_rmse_mmHg", "fit_n"]]
                    .drop_duplicates()
                    .sort_values("beat_id")
                )
                with st.expander("RV Piso sine interpolation summary", expanded=False):
                    st.dataframe(fit_summary, width="stretch", hide_index=True)
            with st.expander("RV derivative feature times", expanded=False):
                st.dataframe(
                    rv_events[["beat_id", "method", "event", "rr_start_s", "rr_end_s", "rr_duration_s", "time_s", "value", "description"]],
                    width="stretch",
                    hide_index=True,
                )
        st.slider(
            f"{pressure_col} fine time alignment (ms)",
            min_value=-1000,
            max_value=1000,
            value=int(st.session_state.get(f"time_shift_{case_key}_{pressure_col}", 0)),
            step=10,
            key=f"time_shift_{case_key}_{pressure_col}",
            help="Negative moves the pressure waveform earlier/left; positive moves it later/right. The EKG row stays fixed.",
        )
        if reference_files and is_pcwp_signal(pressure_col, source_record.get("filename", "")):
            with st.expander("PCWP reference PDF/image", expanded=show_reference_outputs):
                matched_references = sorted(
                    reference_files,
                    key=lambda ref: reference_sort_key(ref, pressure_col, source_record.get("filename", "")),
                )
                reference_names = [ref["name"] for ref in matched_references]
                selected_reference_name = st.selectbox(
                    "PCWP reference output",
                    options=reference_names,
                    index=0,
                    key=f"reference_output_{case_key}_{pressure_col}",
                    help="Shows an uploaded Xper PDF/JPEG/PNG under the PCWP waveform for visual comparison.",
                )
                selected_reference = next(ref for ref in matched_references if ref["name"] == selected_reference_name)
                render_reference_file(selected_reference)

    standalone_ecg_cols = [c for c in display_signal_cols if "EKG" in c]
    if not rendered_pressure_cols and standalone_ecg_cols:
        for ecg_display_col in standalone_ecg_cols:
            fig = make_subplots(rows=1, cols=1, subplot_titles=[ecg_display_col])
            fig.add_trace(
                go.Scatter(
                    x=aligned["time_s"],
                    y=aligned[ecg_display_col],
                    mode="lines",
                    name=ecg_display_col,
                    hovertemplate=f"Time: %{{x:.3f}} s<br>{ecg_display_col}: %{{y:.3f}} mV<extra></extra>",
                ),
                row=1,
                col=1,
            )
            fig.update_yaxes(title_text="mV", row=1, col=1)
            decorate_waveform_figure(fig, bottom_row=1)
            st.plotly_chart(fig, width="stretch", key=f"plot_{case_key}_{ecg_display_col}")

    with st.expander("Detected ECG R waves / timing reference"):
        if r_waves.empty:
            st.info("No R waves detected from the PCWP/PW file ECG, or no same-file PCWP/PW ECG channel is available.")
        else:
            st.caption(f"R waves used for timing come from {r_wave_source_file} / {r_wave_source_signal}, matched to {r_wave_pressure_col}.")
            st.dataframe(r_waves, width="stretch")

    st.subheader("Current selected-frame statistics")
    st.caption(f"Active selection: {lo:.3f}–{hi:.3f} s")

    priority_cols = [
        "signal",
        "duration_s",
        "mean",
        "peak_above_linear_baseline",
        "excess_auc_linear_baseline_positive",
        "normalized_positive_excess_auc",
        "vwave_sharpness_index",
        "area_density_index",
        "relative_vwave_amplitude",
        "vwave_burden_ratio",
        "slope_area_ratio",
        "fwhm_excess_s",
        "time_to_excess_peak_s",
        "qrs_to_excess_peak_ms",
        "cycle_normalized_excess_peak_phase",
        "excess_rise_slope_units_per_s",
        "excess_fall_slope_units_per_s",
        "symmetry_index_excess_peak",
        "raw_auc_to_zero",
    ]
    available_priority_cols = [c for c in priority_cols if c in stats.columns]

    st.markdown("**Morphology-focused summary**")
    st.dataframe(stats[available_priority_cols], width="stretch")

    with st.expander("Show full statistics table"):
        st.dataframe(stats, width="stretch")

    st.subheader("Labeled intervals")
    if st.session_state.intervals:
        intervals_df = pd.DataFrame(st.session_state.intervals)
        intervals_df.insert(0, "interval_id", range(1, len(intervals_df) + 1))
        edited_intervals_df = st.data_editor(
            intervals_df,
            width="stretch",
            hide_index=True,
            disabled=["interval_id"],
            column_config={
                "interval_id": st.column_config.NumberColumn("ID", width="small"),
                "label": st.column_config.TextColumn("Label"),
                "start_s": st.column_config.NumberColumn("Start (s)", min_value=0.0, max_value=float(duration_s), step=0.01, format="%.3f"),
                "end_s": st.column_config.NumberColumn("End (s)", min_value=0.0, max_value=float(duration_s), step=0.01, format="%.3f"),
            },
            key=f"editable_intervals_{case_key}",
        )

        if st.button("Apply interval edits", width="stretch"):
            edited_records = []
            for _, row in edited_intervals_df.sort_values("interval_id").iterrows():
                edited_records.append(
                    {
                        "label": row["label"],
                        "start_s": float(row["start_s"]),
                        "end_s": float(row["end_s"]),
                    }
                )
            st.session_state.intervals = normalize_interval_records(edited_records, duration_s=float(duration_s))
            st.rerun()

        remove_id = st.number_input("Interval ID to remove", min_value=1, max_value=len(st.session_state.intervals), value=1, step=1)
        if st.button("Remove selected interval"):
            st.session_state.intervals.pop(int(remove_id) - 1)
            st.rerun()

        st.subheader("Stats for labeled intervals")
        st.dataframe(labeled_stats, width="stretch")
    else:
        st.info("No labeled intervals yet. Set Cursor A/B, name the interval, then click Add label.")

    st.subheader("Save labeled intervals to database")
    db_save_disabled = labeled_stats.empty
    with st.container(border=True):
        st.caption("Append labeled interval statistics to the local SQLite database. This stores one row per interval per signal.")
        if db_save_disabled:
            st.info("Add at least one labeled interval before saving to the database.")
        if st.button(
            "Save labeled interval stats to database",
            disabled=db_save_disabled,
            type="primary",
            width="stretch",
        ):
            if not patient_id:
                st.warning("Patient / Case ID is blank. The rows were not saved; enter an ID first so accumulated cases stay traceable.")
            else:
                saved_rows = save_labeled_stats_to_database(
                    labeled_stats=labeled_stats,
                    raw_labeled_segments=raw_labeled_segments,
                    patient_id=patient_id,
                    procedure_date=procedure_date,
                    notes=notes,
                    source_files=[up.name for up in uploaded],
                    db_path=Path(st.session_state.database_path),
                )
                st.success(
                    f"Saved {saved_rows['stats_rows']} stat rows and {saved_rows['segment_rows']} waveform sample rows "
                    f"to {Path(st.session_state.database_path).expanduser()}."
                )

    st.subheader("Export")
    export_aligned_cols = ["time_s"] + (display_signal_cols if export_display_only else signal_cols)
    export_segment_cols = ["time_s"] + analysis_signal_cols
    export_aligned = aligned[export_aligned_cols].copy()
    export_segment = segment[export_segment_cols].copy()

    zip_outputs = package_outputs(
        export_aligned,
        export_segment,
        stats,
        labeled_segments,
        labeled_stats,
        raw_labeled_segments,
        patient_id,
        procedure_date,
        notes,
    )

    st.caption(
        "Downloads one ZIP with the current selected segment, full aligned waveforms, labeled interval stats, "
        "raw labeled waveform samples, and case metadata."
    )
    st.download_button(
        "Download export bundle",
        zip_outputs,
        "xper_hemo_export_bundle.zip",
        "application/zip",
        width="stretch",
        type="primary",
    )

    st.subheader("Notes")
    st.markdown(
        """
    - Cursor A/B are synchronized across all panels, so the same labeled interval can segment RA, RV, PA, PCWP, and any other mapped pressure channel.
    - Labels are interval-based. By default, the app auto-generates labels like `PatientID_vwave_1`, `PatientID_vwave_2`, `PatientID_vwave_3`.
    - You can turn off auto-naming to manually enter labels such as `End-expiratory wedge` or `artifact`.
    - Exported labeled segments are in long format: one row per time point per signal per interval.
    - All exported CSVs include `patient_id`, `procedure_date`, and `case_notes` columns.
    - Labeled interval statistics can be appended to a local SQLite database from the Database section.
    - `raw_auc_to_zero` is preserved, but the preferred morphology metric is `excess_auc_linear_baseline_positive`.
    - `excess_auc_linear_baseline_positive` is the area above the straight line connecting interval start and interval end.
    - Composite indices added for group comparisons: `vwave_sharpness_index`, `area_density_index`, `relative_vwave_amplitude`, `vwave_burden_ratio`, and `slope_area_ratio`.
    - ECG timing metrics are added to PCWP/PW rows when same-file R waves are detected: `qrs_to_excess_peak_ms` and `cycle_normalized_excess_peak_phase`.
    - AUC uses a NumPy trapezoid compatibility helper for older and newer NumPy versions.
    - This remains a reverse-engineered parser and should be validated against official Xper/PDF exports.
    """
    )

logo_data_uri = asset_data_uri("assets/poliomics_logo.png")
if logo_data_uri:
    st.markdown(
        f"""
        <div style="display: flex; justify-content: center; padding: 2rem 0 1rem;">
            <img src="{logo_data_uri}" alt="PoliOmics" style="width: 150px; max-width: 34vw;">
        </div>
        """,
        unsafe_allow_html=True,
    )
