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

SWC_FILE = "young_barrel_neuron.swc"

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
    amplitudes = np.arange(0.1, 0.9, 0.1)
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
