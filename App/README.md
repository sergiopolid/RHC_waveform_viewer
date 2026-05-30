# Philips Xper PW6 Hemodynamic Viewer v2

This version adds synchronized dual cursors, patient/case metadata, automatic V-wave interval naming, pressure-channel interval segmentation, RV derivative feature display for single-beat method review, morphology metrics, composite indices, ECG timing features, configurable ECG lead extraction, an in-app data dictionary tab, selectable waveform channel visibility, axis controls, and a local SQLite database for accumulated labeled interval statistics.

## New features

- Dual synchronized cursors visible across all waveform panels
- Active selected frame shaded in all panels
- RV waveform panels include beat-by-beat dP/dt and reconstructed half-sine Pmax/Piso display for single-beat method review
- Label interval windows such as:
  - V wave 1
  - V wave 2
  - End-expiratory PCWP
  - PA systole sample
- Patient / Case ID, procedure date, and notes fields included in every exported CSV
- Dedicated ECG files can be interpreted as D I / D II / D III sequential leads so leads are not concatenated into one trace
- Automatic interval labels such as `PatientID_vwave_1`, `PatientID_vwave_2`, `PatientID_vwave_3`
- Export labeled intervals across all mapped pressure channels:
  - long-format segment CSV
  - per-interval/per-signal stats CSV
  - ZIP bundle
- Save labeled interval statistics to a local one-table SQLite database
- Compatible across older/newer NumPy using a trapezoid helper

## Run

```bash
cd App
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## macOS app distribution

The source code is tracked in git. The generated macOS app and zip are intentionally not committed because the packaged app bundle contains a full Python environment and is large.

Build the local app bundle:

```bash
cd App
./packaging/build_macos_app.sh
```

Refresh the shareable zip:

```bash
cd App/dist
rm -f "Xper Hemodynamic Viewer-macOS.zip"
ditto -c -k --sequesterRsrc --keepParent "Xper Hemodynamic Viewer.app" "Xper Hemodynamic Viewer-macOS.zip"
```

Recommended sharing path: upload `App/dist/Xper Hemodynamic Viewer-macOS.zip` as a GitHub Release asset, not as a regular committed file.

## Workflow

1. Upload all `.PW6` files from one case.
2. Enter Patient / Case ID, optional procedure date, and notes.
3. Confirm channel mapping.
4. Choose ECG source and ECG lead. Default ECG file layout is D I / D II / D III sequential leads.
4. Move Cursor A/B to select a time window.
5. Enter a label, e.g. `V wave 1`.
6. Click **Add label**.
7. Repeat for additional intervals.
8. Export labeled segments and stats, or append labeled interval statistics to the local database.

## Export files

- `current_selected_segment.csv`
- `current_selected_segment_stats.csv`
- `aligned_full_waveforms.csv`
- `labeled_intervals_segments_long.csv`
- `raw_labeled_waveform_segments_wide.csv`
- `labeled_intervals_stats.csv`
- `xper_hemo_export_bundle.zip`

## Local database

The Database section can append labeled interval statistics to:

`~/Documents/Xper Hemodynamic Viewer/xper_hemo_cases.sqlite`

You can override this with an environment variable or Streamlit secret:

`XPER_DATABASE_PATH=/path/to/xper_hemo_cases.sqlite`

The database uses one table, `labeled_interval_stats`, with one row per labeled interval per signal. It includes case metadata, source filenames, interval labels/timing, waveform statistics, morphology metrics, ECG timing metrics when available, and a save timestamp.

Saved intervals also write raw waveform samples to `labeled_interval_segments`, with one row per time point per labeled interval and one column per mapped waveform channel. The Database tab can restore a saved interval set into the Waveform viewer after the matching PW6 files are uploaded, so prior shaded selections can be reviewed, edited, and saved again.

## Website deployment and password

For Streamlit Cloud, use:

- Repository: `sergiopolid/RHC_waveform_viewer`
- Branch: `main`
- Main file path: `App/app.py`

Optional password protection is controlled by a Streamlit secret or environment variable:

`APP_PASSWORD=your-password-here`

If `APP_PASSWORD` is set, the app shows a password screen before the waveform viewer opens. If it is not set, the app opens normally for local use.

Important persistence note: SQLite is a file on the machine/server running the app. On a hosted Streamlit website, that file may reset when the app redeploys, restarts, or moves servers. For a fully website-based shared database, use a persistent hosted database such as Supabase/Postgres instead of local SQLite.

## Morphology metrics added in v5

For every selected or labeled interval, the stats CSV now includes morphology-focused metrics for each mapped pressure channel, such as RA, RV, PA, and PCWP/PW:

- `raw_auc_to_zero`
- `auc_above_start_baseline`
- `auc_above_horizontal_baseline`
- `excess_auc_linear_baseline_net`
- `excess_auc_linear_baseline_positive`
- `normalized_positive_excess_auc`
- `baseline_start`
- `baseline_end`
- `peak_value`
- `peak_above_linear_baseline`
- `time_to_excess_peak_s`
- `excess_rise_slope_units_per_s`
- `excess_fall_slope_units_per_s`
- `fwhm_excess_s`
- `symmetry_index_excess_peak`

The recommended primary V-wave morphology metric is:

`excess_auc_linear_baseline_positive`

This calculates the positive area above a straight line connecting the beginning and end of the selected interval, rather than area to y=0.


## Composite research metrics added in v6

Additional metrics intended for amyloid/restrictive CM vs HFrEF comparisons:

- `vwave_sharpness_index` = `peak_above_linear_baseline / fwhm_excess_s`
- `area_density_index` = `excess_auc_linear_baseline_positive / fwhm_excess_s`
- `relative_vwave_amplitude` = `peak_above_linear_baseline / mean`
- `relative_vwave_amplitude_to_median` = `peak_above_linear_baseline / median`
- `vwave_burden_ratio` = `excess_auc_linear_baseline_positive / raw_auc_to_zero`
- `slope_area_ratio` = `excess_rise_slope_units_per_s / normalized_positive_excess_auc`
- `rise_to_fall_slope_ratio` = `abs(excess_rise_slope) / abs(excess_fall_slope)`

## ECG timing features added in v6

When an ECG channel is available, the app attempts to detect R waves and adds:

- `previous_r_time_s`
- `next_r_time_s`
- `qrs_to_excess_peak_ms`
- `rr_cycle_length_ms`
- `cycle_normalized_excess_peak_phase`

These timing features help relate selected PCWP/RA V-wave morphology to the cardiac cycle.


## Data dictionary tab added in v7

The app now includes a separate **Data dictionary** tab with:

- Searchable variable definitions
- Category filtering
- Recommended feature set for amyloid/restrictive CM vs HFrEF V-wave morphology analysis
- Downloadable data dictionary CSV
- Downloadable recommended feature set CSV


## Channel visibility added in v8

The waveform viewer now includes a **Waveform channel visibility** selector.

Use it to display only selected channels, for example:

- EKG + PCWP
- EKG + PA + PCWP
- RA + PCWP
- PA only

By default, exports include all mapped channels. You can optionally check **Export only displayed channels** if you want the exported aligned waveform/segment files to include only the channels visible on the plot.


## Axis controls added in v9

The waveform viewer now includes **Axis controls**:

- Optional custom x-axis time range
- Optional shared y-axis range for all pressure channels
- Optional per-channel y-axis ranges

Examples:
- Focus the x-axis on 1.5–2.5 seconds
- Set PCWP y-axis to 0–40 mmHg
- Use a shared pressure axis for RA/RV/PA/PCWP to compare amplitudes visually
- Keep ECG on its own mV axis


## Raw segment export and saved-session restore added in v0.8.1

Labeled intervals now export pressure-channel statistics plus a wide-form raw waveform table for AI/ML workflows:

- `raw_labeled_waveform_segments_wide.csv`
- Includes `interval_id`, `interval_label`, interval start/end, `relative_time_s`, original `time_s`, and all mapped waveform channels
- Included in the ZIP bundle

When labeled intervals are saved to the SQLite database, the raw waveform samples are saved too. The Database tab can preview saved segments and restore a prior interval set into the Waveform viewer, where the shaded selections reappear and can be edited before saving/exporting again.

## RV derivative and Piso display updated in v0.8.12

Mapped RV pressure panels include additional rows for visual review of single-beat method landmarks. The RV-only rows are shown only for channels mapped as `RV`, so RA/PA panels do not inherit the RV analysis from filename numbering. Beats are identified from the RV pressure waveform itself, not from ECG R-R intervals:

- RV beat peaks and half-sine `Pmax/Piso` interpolation: smoothed RV pressure, measured RV peak, `Pes`, IC/IR samples used for interpolation, 20% dP/dt interpolation limits, half-sine isovolumic curve, and `Piso` marker
- First derivative method: RV `dP/dt`, with maximum and minimum markers for isovolumic contraction/relaxation references

This is currently a feature-identification/QC display based on Bellofiore et al. 2017. It estimates visual `Pmax/Piso` candidates using `Pbase + amplitude * sin(pi * (t - t_offset) / Tsys)`, so the fitted half-sine sits on the RV pressure baseline rather than being forced through zero. `IVO` is estimated at `dP/dt max`; `IVC` and `Pes` are estimated at `dP/dt min`.

`Piso` candidates are constrained to sit clearly above the measured RV pressure peak for that beat, using the beat pulse pressure to set a visible physiologic margin. If the sine interpolation cannot satisfy that requirement, the app skips that candidate rather than displaying a misleading low fitted peak.

Optional sidebar inputs for stroke volume and RV EDV enable single-beat mechanics:

- `Ees = (Pmax - Pes) / (EDV - SV)` when EDV is greater than SV
- `Ea = Pes / SV` when SV is supplied
- `Ees/Ea` when both Ees and Ea are available
