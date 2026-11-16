# AI Context: Cross-Repo Integration with `simulations/gui`

The companion desktop studio at `../simulations/gui` (`sim-gui`) launches, monitors, throttles, and visualizes simulations powered by `nmag-python-3`. This note captures **runtime contracts and non-obvious assumptions** across the two repositories that are not self-evident from reading `simulations/gui/README.md` or source code in isolation.

---

## 1. Output File & Telemetry Schema Contracts

`sim_gui` parses live solver output incrementally off the Qt UI thread (`gui/src/sim_gui/data/telemetry.py`, `gui/src/sim_gui/data/mesh_and_fields.py`, `gui/rust/sim_gui_accel/src/lib.rs`). Changes in `nmag-python-3` to the paths, schemas, or flush semantics below directly affect live GUI telemetry and 3D playback:

- **`simulation_dat.ndt` (`sim_gui.data.telemetry.NdtParser`, `parse_ndt_columns` in Rust)**
  - Read **incrementally by byte offset** assuming an append-only table during a run (truncation triggers a full parser reset).
  - Expects `id`, `step`, `time` (seconds), `stage_time` (seconds), `last_step_dt` (seconds), energy prefixes (`E_total*`, `E_exch*`, `E_demag*`), magnetization average vectors (`m_*_0`, `m_*_1`, `m_*_2`), and `dmdt_*_0/1/2`.
  - Any column containing a non-numeric token (e.g., partial write or string column like `localtime`) is discarded by the Rust parser on that read; `NdtParser` defensively aligns column lengths to `min(len(time), len(step))` so PyQtGraph never crashes on jagged arrays.

- **`simulation_progress.txt` (`ProgressFileParser`) & `simulation_progress.json`**
  - Parses pipe-delimited `Key = Value` cards (`Step`, `Time`, `Last step size`, `Stage`, `Stage-step`, `Stage-time`, `Convergence`, `Stage-end`, `Exit hysteresis`) and unit suffixes (`ps`, `ns`, `fs`, `s`).
  - Parses the `# Convergence log` block (`step max_dm_dt stopping_dm_dt [quality]`) to plot `log10(max |dm/dt|)` vs stopping criterion.
  - Reads `completed_states` from `simulation_progress.json` for macro multi-state progress bars.

- **Live Telemetry Hook (`relax()` vs `simulation_hysteresis()`)**
  - By default, `simulation.relax()` buffers `simulation_dat.ndt` rows until `stage_end` and throttles `simulation_progress.txt` writes to ~60s unless `progress_message_minimum_delay` is lowered.
  - To stream ~1 Hz convergence and energy curves when 3D snapshots are disabled, `--telemetry-steps N` in `simulations/majority-3-fused-triad/majority_simulation.py` calls:
    ```python
    simulation.simulation_hysteresis(
        [None],
        save=[..., (lambda sim: sim.save_data(), every("step", telemetry_interval_steps))],
        progress_message_minimum_delay=1.0,
    )
    ```
  - Note that calling `save_data()` (without `fields="all"`) must continue writing scalar `.ndt` rows cheaply without breaking subsequent `save_data(fields="all")` HDF5 dumps used by `shutil.copy2(..., snapshot_path)`.

- **HDF5 & VTK Spatial Contracts**
  - `assets/model.nmesh.h5` expects `/mesh/points` (model units) and `/mesh/simplices` `(M, 4)` tetrahedra (used by Rust `extract_exterior_faces`).
  - State/frame HDF5 files (`results/simulation-<state>-initial_dat.h5`, `results/snapshots/<state>/frame-XXXXXX_dat.h5`, `results/simulation-<state>-final_dat.h5`) are expected to contain `/fields/<name>` datasets shaped `(N, 3)` (vectors) or `(N,)` (scalars), and optional `/mesh/points` in SI meters (divided by adapter `unit_length` when mesh geometry is absent).

---

## 2. Stdout State Announcements & Write Races

`LogStateTracker` (`sim_gui/data/telemetry.py`) regex-matches solver stdout lines to track active states and decide when to reload HDF5 frames:

- `preparing ... state <STATE>`
- `relaxing state <STATE>`
- `saving state <STATE> (initial|final) snapshot`
- `saved state <STATE> snapshot: <PATH>`
- `saving ... checkpoint after state <STATE>`

**Crucial detail**: Scripts print `saving state ... initial/final snapshot` *before* calling `simulation.save_data(fields="all")` and `shutil.copy2(...)`. `MainWindow` therefore waits for `relaxing`, `saved ... snapshot:`, or `checkpoint` announcements (plus periodic background polls) before opening the `.h5` file so it does not read a partially written HDF5 file. Keep post-write log announcements intact or write snapshots atomically (`temp` + `os.replace`) if changing simulation scripts.

---

## 3. Subprocess Isolation, Thread Caps & Adaptive `SIGSTOP`/`SIGCONT` Throttling

`simulations/gui/src/sim_gui/runner/process_manager.py` runs `nmag` workflows as child processes (`QProcess`):

- **Environment discovery**: Prefers `<shape_dir>/.venv/bin/python` -> sibling `<dir>/.venv/bin/python` -> `../nmag-python-3/.venv/bin/python`.
- **Thread-pool capping**: Always injects `RAYON_NUM_THREADS`, `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `BLIS_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, and `NUMEXPR_NUM_THREADS` according to the UI **CPU Budget** (`balanced` reserves 1-2 GUI/compositor cores; `low_power` caps at 50%; `max` unlocks all logical cores), overriding any unrestricted parent shell defaults. `nmag_accel` and BLAS backends must continue respecting these environment variables at worker initialization.
- **Process group & duty-cycle throttling**:
  - Child processes are spawned in an isolated POSIX session (`os.setsid()`) with lowered `os.nice(...)` (`+10` balanced, `+15` low power).
  - Under Qt event-loop lag, high load (`os.getloadavg()`), or low RAM pressure, `ProcessManager` may briefly pause/resume the entire child process group via `SIGSTOP`/`SIGCONT` (~220-320 ms intervals) and always sends `SIGCONT` before `SIGTERM`/`SIGKILL` on cancel.
  - Solvers in `nmag-python-3` should remain safe across arbitrary `SIGSTOP`/`SIGCONT` pauses (avoid strict wall-clock watchdog timeouts inside C/Rust/Python solver loops, and keep `last_step_dt` / step-size control independent of wall-clock deltas).

---

## 4. Rust Extension Mirrors (`nmag_accel` <-> `sim_gui_accel`)

`simulations/gui/rust/sim_gui_accel` mirrors the packaging conventions of `nmag-python-3/rust/nmag_accel` (`pyo3 = "0.27"`, `numpy = "0.27"`, `rayon = "1.11"`, `maturin`, `.cargo/config.toml` macOS `dynamic_lookup` linker args, and pure-NumPy fallbacks).

- If changing HDF5/VTK export structure (`src/nmag/vtk_export.py`, `output.py`, `checkpoint.py`), check `simulations/gui/src/sim_gui/_accel.py`, `simulations/gui/rust/sim_gui_accel/src/lib.rs`, and `simulations/gui/tests/` for compatibility.

