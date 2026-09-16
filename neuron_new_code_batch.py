"""
================================================================================
NEURON Starter Script
Golgi-derived morphology -> virtual electrophysiology simulation
================================================================================

Goal:
    Load one SWC morphology file exported from Neurolucida, convert it into a
    NEURON model, add simple biophysical properties, inject current at the soma,
    and plot the membrane voltage response.

What this script is for:
    This is a first-run teaching script. It is designed to help students confirm
    that the morphology loads correctly and that basic current-clamp simulation
    works.

What this script is not yet:
    This is not a fully calibrated barrel cortex pyramidal neuron model.
    Channel densities and passive properties are starter values only.

Required packages:
    pip install neuron matplotlib numpy

Input file:
    Put one SWC file in the same folder as this script.
================================================================================
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from neuron import h


# ---------------------------------------------------------------------
# 1. Load standard NEURON tools
# ---------------------------------------------------------------------

h.load_file("stdrun.hoc")
h.load_file("import3d.hoc")


# ---------------------------------------------------------------------
# 2. User settings
# ---------------------------------------------------------------------

SWC_FILE = "HY484-FemaleR-ControlP45-slide4a-S1BF-L4-Cell1.swc"

RESTING_VM = -70.0       # mV
RA = 150.0               # ohm-cm, axial resistance
CM = 1.0                 # uF/cm2, membrane capacitance

G_PAS = 3e-5             # S/cm2, passive leak conductance
E_PAS = -70.0            # mV, passive leak reversal potential

DT = 0.025               # ms
TSTOP = 400.0            # ms

STIM_DELAY = 100.0       # ms
STIM_DURATION = 200.0    # ms
STIM_AMPLITUDE = 0.5     # nA


# ---------------------------------------------------------------------
# 3. Morphology loading
# ---------------------------------------------------------------------

def clear_existing_model():
    """
    Delete any sections already present in NEURON.

    This prevents accidental mixing of two cells if the script is run multiple
    times in the same Python session.
    """
    for sec in list(h.allsec()):
        h.delete_section(sec=sec)


def load_swc_morphology(swc_file):
    """
    Load an SWC morphology file into NEURON.

    Parameters
    ----------
    swc_file : str or Path
        Path to the SWC morphology file.

    Returns
    -------
    list
        A list of all NEURON sections created from the morphology.
    """
    swc_path = Path(swc_file)

    if not swc_path.exists():
        raise FileNotFoundError(
            f"Could not find SWC file: {swc_path}\n"
            "Make sure the SWC file is in the same folder as this script, "
            "or provide the full file path."
        )

    clear_existing_model()

    print(f"\nLoading morphology from: {swc_path}")

    importer = h.Import3d_SWC_read()
    importer.input(str(swc_path))

    gui_importer = h.Import3d_GUI(importer, 0)
    gui_importer.instantiate(None)

    sections = list(h.allsec())

    if len(sections) == 0:
        raise RuntimeError("The SWC file loaded, but no NEURON sections were created.")

    print("Successfully loaded morphology.")
    print(f"Number of sections created: {len(sections)}")

    return sections


def print_section_summary(sections):
    """
    Print a short summary of the imported morphology.
    """
    print("\nSection summary:")
    for sec in sections[:10]:
        print(f"  {sec.name():30s}  L = {sec.L:.2f} um, diam = {sec.diam:.2f} um")

    if len(sections) > 10:
        print(f"  ... {len(sections) - 10} additional sections not shown")


def find_soma_section(sections):
    """
    Find the soma section after SWC import.

    Why this is needed:
        After Import3D, NEURON does not always create a simple object called h.soma.
        The soma may be named soma[0], Cell[0].soma, or something similar.

    Strategy:
        1. First look for any section with 'soma' in its name.
        2. If no soma is found, use the section with the largest diameter as a
           fallback, but warn the user.
    """
    soma_candidates = [
        sec for sec in sections
        if "soma" in sec.name().lower()
    ]

    if len(soma_candidates) > 0:
        soma = soma_candidates[0]
        print(f"\nSoma detected: {soma.name()}")
        return soma

    # Fallback: not ideal, but useful for debugging strange SWC files
    fallback = max(sections, key=lambda sec: sec.diam)

    print("\nWARNING: No section with 'soma' in the name was found.")
    print(f"Using largest-diameter section as fallback: {fallback.name()}")
    print("Please verify this manually before trusting biological results.")

    return fallback


# ---------------------------------------------------------------------
# 4. Biophysics setup
# ---------------------------------------------------------------------

def set_nseg_by_length(sections, target_segment_length=40.0):
    """
    Set the number of computational segments for each section.

    NEURON divides every dendritic section into smaller electrical compartments.
    Longer dendrites need more compartments.

    This simple rule keeps segment length around target_segment_length.
    """
    for sec in sections:
        nseg = int(sec.L / target_segment_length) + 1

        # NEURON convention: use odd nseg values so there is a center point at 0.5
        if nseg % 2 == 0:
            nseg += 1

        sec.nseg = max(1, nseg)


def insert_passive_properties(sections):
    """
    Insert passive leak properties into all sections.

    This is used for passive signal spread and basic morphology-constrained
    voltage attenuation.
    """
    for sec in sections:
        sec.Ra = RA
        sec.cm = CM

        sec.insert("pas")
        sec.g_pas = G_PAS
        sec.e_pas = E_PAS


def insert_simple_somatic_hh(soma):
    """
    Insert Hodgkin-Huxley channels into the soma.

    This gives the model a simple ability to fire action potentials.

    Important:
        These are generic HH channels, not a fully calibrated layer 4 pyramidal
        neuron channel model.
    """
    soma.insert("hh")

    soma.gnabar_hh = 0.12
    soma.gkbar_hh = 0.036
    soma.gl_hh = 0.0003
    soma.el_hh = -54.3

    print(f"Inserted simple HH channels into soma: {soma.name()}")


def setup_biophysics(sections, soma, active_soma=True):
    """
    Apply starter biophysical parameters.

    Parameters
    ----------
    sections : list
        All sections in the morphology.
    soma : NEURON Section
        Soma section.
    active_soma : bool
        If True, insert HH channels into the soma.
        If False, keep the model passive only.
    """
    print("\nSetting up biophysics...")

    set_nseg_by_length(sections)
    insert_passive_properties(sections)

    if active_soma:
        insert_simple_somatic_hh(soma)

    print("Biophysics setup complete.")


# ---------------------------------------------------------------------
# 5. Current-clamp simulation
# ---------------------------------------------------------------------

def run_current_clamp(
    soma,
    stim_amplitude=STIM_AMPLITUDE,
    stim_delay=STIM_DELAY,
    stim_duration=STIM_DURATION,
    tstop=TSTOP,
):
    """
    Inject current into the soma and record somatic voltage.

    Parameters
    ----------
    soma : NEURON Section
        Soma section where current will be injected.
    stim_amplitude : float
        Current amplitude in nA.
    stim_delay : float
        Time before stimulation starts, in ms.
    stim_duration : float
        Duration of current injection, in ms.
    tstop : float
        Total simulation time, in ms.

    Returns
    -------
    time : np.ndarray
        Time vector in ms.
    voltage : np.ndarray
        Somatic membrane voltage in mV.
    """
    print(
        f"\nRunning current clamp: "
        f"{stim_amplitude} nA, delay = {stim_delay} ms, duration = {stim_duration} ms"
    )

    stim = h.IClamp(soma(0.5))
    stim.delay = stim_delay
    stim.dur = stim_duration
    stim.amp = stim_amplitude

    time_vec = h.Vector().record(h._ref_t)
    voltage_vec = h.Vector().record(soma(0.5)._ref_v)

    h.dt = DT
    h.tstop = tstop

    h.finitialize(RESTING_VM)
    h.run()

    time = np.array(time_vec)
    voltage = np.array(voltage_vec)

    return time, voltage


def count_spikes(voltage, threshold=0.0):
    """
    Count action potentials using upward threshold crossings.

    Parameters
    ----------
    voltage : np.ndarray
        Voltage trace in mV.
    threshold : float
        Spike detection threshold in mV.

    Returns
    -------
    int
        Number of detected spikes.
    """
    crossings = np.where((voltage[:-1] < threshold) & (voltage[1:] >= threshold))[0]
    return len(crossings)


# ---------------------------------------------------------------------
# 6. Plotting
# ---------------------------------------------------------------------

def plot_voltage_trace(time, voltage, stim_delay, stim_duration, title):
    """
    Plot somatic voltage response.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(time, voltage, label="Soma voltage")

    plt.axvspan(
        stim_delay,
        stim_delay + stim_duration,
        alpha=0.15,
        label="Current injection window",
    )

    plt.title(title)
    plt.xlabel("Time (ms)")
    plt.ylabel("Membrane potential (mV)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def run_fi_curve(soma, amplitudes):
    """
    Run a simple F-I curve.

    F-I means:
        current input -> spike frequency output

    Parameters
    ----------
    soma : NEURON Section
        Soma section.
    amplitudes : list or np.ndarray
        Current amplitudes in nA.

    Returns
    -------
    results : dict
        Contains current amplitudes, spike counts, and firing rates.
    """
    spike_counts = []
    firing_rates_hz = []

    for amp in amplitudes:
        time, voltage = run_current_clamp(
            soma,
            stim_amplitude=amp,
            stim_delay=STIM_DELAY,
            stim_duration=STIM_DURATION,
            tstop=TSTOP,
        )

        n_spikes = count_spikes(voltage)
        firing_rate = n_spikes / (STIM_DURATION / 1000.0)

        spike_counts.append(n_spikes)
        firing_rates_hz.append(firing_rate)

        print(f"  {amp:.2f} nA -> {n_spikes} spikes, {firing_rate:.1f} Hz")

    results = {
        "current_nA": np.array(amplitudes),
        "spike_count": np.array(spike_counts),
        "firing_rate_Hz": np.array(firing_rates_hz),
    }

    return results


def plot_fi_curve(fi_results):
    """
    Plot current amplitude versus firing rate.
    """
    plt.figure(figsize=(6, 5))
    plt.plot(
        fi_results["current_nA"],
        fi_results["firing_rate_Hz"],
        marker="o",
    )

    plt.title("F-I curve")
    plt.xlabel("Injected current (nA)")
    plt.ylabel("Firing rate (Hz)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------
# 7. Main script
# ---------------------------------------------------------------------

def main():
    """
    Main analysis workflow.

    This is the part students should run first.
    """
    sections = load_swc_morphology(SWC_FILE)
    print_section_summary(sections)

    soma = find_soma_section(sections)
    
      
    print("\n=== Soma geometry before standardization ===")
    print(f"  L = {soma.L:.2f} um, diam = {soma.diam:.2f} um, nseg = {soma.nseg}")
    print(f"  area = {soma(0.5).area():.2f} um^2")

    print("\n=== Standardizing soma geometry (temporary test) ===")
    soma.L = 20.0
    soma.diam = 20.0
    soma.nseg = 1
    print(f"  After: L={soma.L} um, diam={soma.diam} um, nseg={soma.nseg}, area={soma(0.5).area():.2f} um^2")

    

    setup_biophysics(
        sections=sections,
        soma=soma,
        active_soma=True,
    )   

    # First simple test: one current step
    time, voltage = run_current_clamp(
        soma=soma,
        stim_amplitude=STIM_AMPLITUDE,
        stim_delay=STIM_DELAY,
        stim_duration=STIM_DURATION,
        tstop=TSTOP,
    )

    n_spikes = count_spikes(voltage)

    print("\nSingle-trace result:")
    print(f"  Peak voltage: {np.max(voltage):.2f} mV")
    print(f"  Minimum voltage: {np.min(voltage):.2f} mV")
    print(f"  Spike count: {n_spikes}")

    plot_voltage_trace(
        time=time,
        voltage=voltage,
        stim_delay=STIM_DELAY,
        stim_duration=STIM_DURATION,
        title=f"Somatic current clamp: {SWC_FILE}",
    )

    # Optional second test: F-I curve
    print("\nRunning simple F-I curve...")
    amplitudes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    fi_results = run_fi_curve(soma, amplitudes)
    plot_fi_curve(fi_results)


if __name__ == "__main__":
    try:
        main()

    except FileNotFoundError as error:
        print("\nSETUP NEEDED")
        print(error)
        print("\nStudent checklist:")
        print("1. Export one Neurolucida reconstruction as an SWC file.")
        print("2. Put the SWC file in the same folder as this script.")
        print("3. Update SWC_FILE at the top of this script.")
        print("4. Run: python neuron_starter.py")

    except Exception as error:
        print("\nThe script stopped because of an error:")
        print(error)
        print("\nDebugging suggestions:")
        print("1. Confirm that NEURON is installed: pip install neuron")
        print("2. Confirm that the SWC file is valid and not empty.")
        print("3. Check whether the imported sections include a soma.")
        print("4. Print all section names using: [sec.name() for sec in h.allsec()]")

# ============================================================================
# BATCH ANALYSIS EXTENSION
# Original code above is intentionally preserved unchanged.
# ============================================================================

from dataclasses import dataclass, asdict
import csv
import json
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed


@dataclass
class BatchConfig:
    """Configuration for large-scale SWC morphology + electrophysiology analysis."""
    input_dir: str = "."
    output_dir: str = "neuron_batch_results"
    pattern: str = "*.swc"

    # Simulation
    resting_vm: float = -70.0
    ra: float = 150.0
    cm: float = 1.0
    g_pas: float = 3e-5
    e_pas: float = -70.0
    dt: float = 0.025
    tstop: float = 400.0
    stim_delay: float = 100.0
    stim_duration: float = 200.0

    # Active model
    active_soma: bool = True
    gnabar_hh: float = 0.12
    gkbar_hh: float = 0.036
    gl_hh: float = 0.0003
    el_hh: float = -54.3

    # Numerical discretization
    target_segment_length: float = 40.0

    # Batch stimulation
    amplitudes_nA: tuple = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)

    # Analysis
    spike_threshold_mV: float = 0.0
    save_traces: bool = True
    save_plots: bool = True
    max_workers: int = 1


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def _section_geometry(sections):
    lengths = np.array([float(sec.L) for sec in sections], dtype=float)
    diams = np.array([float(sec.diam) for sec in sections], dtype=float)

    return {
        "n_sections": len(sections),
        "total_length_um": float(np.sum(lengths)),
        "mean_section_length_um": float(np.mean(lengths)) if len(lengths) else np.nan,
        "max_section_length_um": float(np.max(lengths)) if len(lengths) else np.nan,
        "mean_diameter_um": float(np.mean(diams)) if len(diams) else np.nan,
        "max_diameter_um": float(np.max(diams)) if len(diams) else np.nan,
    }


def _morphology_type_counts(sections):
    """Best-effort section-name classification; does not alter the morphology."""
    counts = {"soma": 0, "axon": 0, "dendrite": 0, "other": 0}
    for sec in sections:
        name = sec.name().lower()
        if "soma" in name:
            counts["soma"] += 1
        elif "axon" in name:
            counts["axon"] += 1
        elif "dend" in name:
            counts["dendrite"] += 1
        else:
            counts["other"] += 1
    return counts


def _detect_spikes(time, voltage, threshold=0.0):
    """Return spike times from upward threshold crossings."""
    crossings = np.where(
        (voltage[:-1] < threshold) & (voltage[1:] >= threshold)
    )[0]

    if len(crossings) == 0:
        return np.array([], dtype=float)

    # Linear interpolation gives a better estimate of threshold-crossing time.
    spike_times = []
    for i in crossings:
        v0, v1 = voltage[i], voltage[i + 1]
        t0, t1 = time[i], time[i + 1]
        if v1 != v0:
            spike_times.append(t0 + (threshold - v0) * (t1 - t0) / (v1 - v0))
        else:
            spike_times.append(t0)
    return np.asarray(spike_times, dtype=float)


def _trace_metrics(time, voltage, stim_delay, stim_duration, threshold=0.0):
    spikes = _detect_spikes(time, voltage, threshold)
    stim_end = stim_delay + stim_duration

    baseline_mask = time < stim_delay
    stim_mask = (time >= stim_delay) & (time <= stim_end)

    baseline = voltage[baseline_mask] if np.any(baseline_mask) else voltage[:1]
    during = voltage[stim_mask] if np.any(stim_mask) else voltage

    rate_hz = len(spikes) / (stim_duration / 1000.0)

    isi_ms = np.diff(spikes)
    return {
        "spike_count": int(len(spikes)),
        "firing_rate_Hz": float(rate_hz),
        "peak_voltage_mV": float(np.max(voltage)),
        "minimum_voltage_mV": float(np.min(voltage)),
        "baseline_mean_mV": float(np.mean(baseline)),
        "baseline_std_mV": float(np.std(baseline)),
        "stim_mean_mV": float(np.mean(during)),
        "stim_peak_mV": float(np.max(during)),
        "spike_times_ms": spikes.tolist(),
        "mean_ISI_ms": float(np.mean(isi_ms)) if len(isi_ms) else np.nan,
        "CV_ISI": float(np.std(isi_ms) / np.mean(isi_ms))
        if len(isi_ms) and np.mean(isi_ms) > 0 else np.nan,
    }


def _setup_batch_biophysics(sections, soma, cfg):
    for sec in sections:
        sec.Ra = cfg.ra
        sec.cm = cfg.cm
        sec.insert("pas")
        sec.g_pas = cfg.g_pas
        sec.e_pas = cfg.e_pas

        nseg = int(sec.L / cfg.target_segment_length) + 1
        if nseg % 2 == 0:
            nseg += 1
        sec.nseg = max(1, nseg)

    if cfg.active_soma:
        soma.insert("hh")
        soma.gnabar_hh = cfg.gnabar_hh
        soma.gkbar_hh = cfg.gkbar_hh
        soma.gl_hh = cfg.gl_hh
        soma.el_hh = cfg.el_hh


def _load_batch_cell(swc_path):
    """Load one cell into a fresh NEURON process/model."""
    for sec in list(h.allsec()):
        h.delete_section(sec=sec)

    importer = h.Import3d_SWC_read()
    importer.input(str(swc_path))
    gui_importer = h.Import3d_GUI(importer, 0)
    gui_importer.instantiate(None)

    sections = list(h.allsec())
    if not sections:
        raise RuntimeError("No NEURON sections were created.")

    soma_candidates = [
        sec for sec in sections if "soma" in sec.name().lower()
    ]
    soma = soma_candidates[0] if soma_candidates else max(
        sections, key=lambda sec: sec.diam
    )
    return sections, soma


def _run_trace(soma, amplitude, cfg):
    stim = h.IClamp(soma(0.5))
    stim.delay = cfg.stim_delay
    stim.dur = cfg.stim_duration
    stim.amp = amplitude

    time_vec = h.Vector().record(h._ref_t)
    voltage_vec = h.Vector().record(soma(0.5)._ref_v)

    h.dt = cfg.dt
    h.tstop = cfg.tstop
    h.finitialize(cfg.resting_vm)
    h.run()

    return np.asarray(time_vec), np.asarray(voltage_vec)


def analyze_one_swc(swc_path, cfg):
    """
    Analyze one SWC without changing the user's original starter workflow.
    Returns a JSON/CSV-friendly dictionary.
    """
    swc_path = Path(swc_path)
    try:
        sections, soma = _load_batch_cell(swc_path)
        _setup_batch_biophysics(sections, soma, cfg)

        morph = _section_geometry(sections)
        morph.update(_morphology_type_counts(sections))

        result = {
            "file": swc_path.name,
            "path": str(swc_path.resolve()),
            "status": "OK",
            "soma_name": soma.name(),
            "soma_length_um": float(soma.L),
            "soma_diameter_um": float(soma.diam),
            **morph,
            "traces": [],
        }

        for amp in cfg.amplitudes_nA:
            time, voltage = _run_trace(soma, float(amp), cfg)
            metrics = _trace_metrics(
                time,
                voltage,
                cfg.stim_delay,
                cfg.stim_duration,
                cfg.spike_threshold_mV,
            )
            metrics["current_nA"] = float(amp)
            result["traces"].append(metrics)

            if cfg.save_traces:
                trace_dir = Path(cfg.output_dir) / "traces"
                trace_dir.mkdir(parents=True, exist_ok=True)
                trace_file = trace_dir / f"{swc_path.stem}__{amp:g}nA.csv"
                np.savetxt(
                    trace_file,
                    np.column_stack([time, voltage]),
                    delimiter=",",
                    header="time_ms,voltage_mV",
                    comments="",
                )

        # Derive compact cell-level summaries from the F-I sweep.
        rates = np.array([x["firing_rate_Hz"] for x in result["traces"]])
        counts = np.array([x["spike_count"] for x in result["traces"]])
        amps = np.array([x["current_nA"] for x in result["traces"]])

        result["max_firing_rate_Hz"] = float(np.max(rates)) if len(rates) else np.nan
        result["total_spikes_across_sweep"] = int(np.sum(counts))
        active = np.where(counts > 0)[0]
        result["first_spiking_current_nA"] = (
            float(amps[active[0]]) if len(active) else np.nan
        )

        if cfg.save_plots:
            plot_dir = Path(cfg.output_dir) / "plots"
            plot_dir.mkdir(parents=True, exist_ok=True)

            plt.figure(figsize=(8, 5))
            plt.plot(amps, rates, marker="o")
            plt.xlabel("Injected current (nA)")
            plt.ylabel("Firing rate (Hz)")
            plt.title(f"F-I curve: {swc_path.name}")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(plot_dir / f"{swc_path.stem}__FI.png", dpi=180)
            plt.close()

        return result

    except Exception as exc:
        return {
            "file": swc_path.name,
            "path": str(swc_path.resolve()),
            "status": "ERROR",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def _flatten_result(result):
    """Convert nested results into one CSV row."""
    row = {
        k: v for k, v in result.items()
        if k != "traces" and k != "traceback"
    }

    for trace in result.get("traces", []):
        amp = trace["current_nA"]
        prefix = f"amp_{amp:g}nA"
        for key, value in trace.items():
            if key in {"current_nA", "spike_times_ms"}:
                continue
            row[f"{prefix}_{key}"] = value

    return row


def save_batch_results(results, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Full machine-readable JSON, including spike times.
    with open(output_dir / "batch_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, allow_nan=True)

    rows = [_flatten_result(r) for r in results]
    if rows:
        fieldnames = sorted({k for row in rows for k in row.keys()})
        with open(output_dir / "batch_summary.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    ok = [r for r in results if r.get("status") == "OK"]
    summary = {
        "n_files": len(results),
        "n_success": len(ok),
        "n_failed": len(results) - len(ok),
    }

    with open(output_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nBatch complete:")
    print(f"  Files found: {summary['n_files']}")
    print(f"  Successful:  {summary['n_success']}")
    print(f"  Failed:      {summary['n_failed']}")
    print(f"  Results:     {output_dir.resolve()}")


def run_batch_analysis(cfg=None):
    """
    Batch entry point.

    Put as many SWC files as desired in cfg.input_dir. Each cell is processed
    independently and receives the same morphology/electrophysiology workflow.
    """
    cfg = cfg or BatchConfig()
    input_dir = Path(cfg.input_dir)
    output_dir = Path(cfg.output_dir)

    swc_files = sorted(input_dir.glob(cfg.pattern))
    if not swc_files:
        raise FileNotFoundError(
            f"No SWC files matching {cfg.pattern!r} were found in {input_dir.resolve()}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("NEURON BATCH ANALYSIS")
    print("=" * 72)
    print(f"Input:      {input_dir.resolve()}")
    print(f"SWC files:  {len(swc_files)}")
    print(f"Output:     {output_dir.resolve()}")
    print(f"Currents:   {list(cfg.amplitudes_nA)} nA")
    print(f"Workers:    {cfg.max_workers}")

    results = []

    # ProcessPoolExecutor is intentionally optional. NEURON is safest when each
    # worker owns an independent model; max_workers=1 is the robust default.
    if cfg.max_workers > 1:
        with ProcessPoolExecutor(max_workers=cfg.max_workers) as executor:
            futures = {
                executor.submit(analyze_one_swc, str(path), cfg): path
                for path in swc_files
            }
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(f"[{result.get('status')}] {result['file']}")
    else:
        for i, path in enumerate(swc_files, start=1):
            print(f"\n[{i}/{len(swc_files)}] {path.name}")
            result = analyze_one_swc(path, cfg)
            results.append(result)
            print(f"  Status: {result.get('status')}")

    results.sort(key=lambda x: x.get("file", ""))
    save_batch_results(results, output_dir)
    return results


def batch_main():
    """
    Convenience launcher for the extension.

    The original main() above is NOT called here, so using this launcher does
    not modify or replace the original single-cell experiment.
    """
    cfg = BatchConfig(
        input_dir="swc_cells",
        output_dir="neuron_batch_results",
        pattern="*.swc",
        amplitudes_nA=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0),
        save_traces=True,
        save_plots=True,
        max_workers=1,
    )
    return run_batch_analysis(cfg)


# To activate batch mode, run:
#     batch_main()
#
# Recommended command-line style:
#     python -c "import neuron_new_code as n; n.batch_main()"
#
# The original:
#     if __name__ == "__main__": ...
# remains untouched above.
