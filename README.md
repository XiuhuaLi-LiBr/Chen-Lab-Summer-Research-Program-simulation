# Golgi-derived NEURON starter project

This project loads one Neurolucida-exported SWC morphology into NEURON and runs a simple virtual current-clamp simulation.

## Setup

Install packages:

```bash
pip install neuron matplotlib numpy
```

## First task

1. Export one reconstructed neuron from Neurolucida as `.swc`.
2. Put the `.swc` file in the same folder as `neuron_starter.py`.
3. Open `neuron_starter.py`.
4. Change this line:

```python
SWC_FILE = "young_barrel_neuron.swc"
```

to match the real SWC file name.

5. Run:

```bash
python neuron_starter.py
```

## What should happen

The script should:

1. Load the SWC morphology.
2. Print the first few NEURON sections.
3. Detect the soma.
4. Insert passive properties into the morphology.
5. Insert simple Hodgkin-Huxley channels into the soma.
6. Inject current at the soma.
7. Plot the somatic voltage response.
8. Run a simple F-I curve.

## Important note

This is a starter model, not a fully calibrated biological model.

The first goal is not to claim real electrophysiology.  
The first goal is to confirm that our real Golgi-derived morphology can be converted into a working virtual neuron.

## Next extensions

After the first script works, we can add:

1. Passive attenuation from distal dendrite to soma.
2. Spatial summation from multiple dendritic input sites.
3. Temporal summation with different input delays.
4. Young vs aged morphology comparisons.
5. Morphology-constrained F-I curve comparisons across many reconstructed neurons.
