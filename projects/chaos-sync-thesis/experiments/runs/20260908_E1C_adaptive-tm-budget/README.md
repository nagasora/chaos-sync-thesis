# E1C finite-grid adaptive TM budget pilot

Status: preregistered; implementation and focused tests may proceed, but the
experiment calculation is blocked until the audit reviewer approves these
conditions.

## Scope and question

Compare a finite-grid QR/least-squares adaptive TM representation with a fixed
TM basis and a low-frequency Fourier basis for lossily reconstructing short
windows of saved tan-map trajectories under an explicit digital bit budget.

The question is whether adaptive pole selection reduces reconstruction error at
the same real-coordinate budget. This run does not test input decoding, VAE
advantages, synchronization, or greedy optimality.

## Input and split

- Source: the verified E0 representative orbit artifact from
  `20260906_E1_tangent-tm-complex-plane/artifacts/representative_orbits.npz`.
- Cases: `b1.01_float64`, `b1.1_float64`, `b1.5_float64`, and `b2.0_float64`.
- Windows: 12 non-overlapping windows of length `T=128` per beta, selected by
  the fixed config rule. These are windows from one saved representative orbit,
  not independent seeds or a population-level test set.
- Observation: `y_t = q_1(x_t)` with the scale-one Cayley map. The DFT grid is
  `xi_t = exp(2*pi*i*t/T)`.

## Fixed conditions

- Pole candidates: `a=0`, and radii `0.3`, `0.6`, `0.85` at 16 equally spaced
  angles, for 49 candidates including the origin.
- Adaptive basis: TM columns are generated recursively. At each step the
  candidate maximizing the QR ordinary-residual gain is selected; all selected
  coefficients are then refit by complex QR/least squares.
- The selection residual and any AFD reduced remainder are distinct objects.
  This pilot uses the ordinary finite-grid residual only and does not claim the
  AFD greedy theorem or AMO's MLP pole-selection procedure.
- Baselines: fixed `a=0` TM columns (nonnegative-frequency convention) and
  two-sided low-frequency DFT order `0, 1, -1, 2, -2, ...`.
- Diagnostics: `m=2,4,8`; equal real-coordinate budgets `d=8,16,32`, meaning
  fixed `m=d/2` and adaptive `m=d/4` because adaptive storage also includes a
  pole index per term.

## Codec contract

The true bit codec stores complex coefficients as float32 (64 bits each),
adaptive candidate indices as uint8 (8 bits each), and a shared versioned
header containing `T`, `m`, and method. Decoding must use only the saved
bitstream and the shared grid convention; it must not access the original
window or recompute selection. Raw complex float32 is the unstructured control.
Quantizer resolution is intentionally not swept in this pilot.

## Required artifacts and gates

The implementation must save source and reconstructed windows, selected poles,
coefficients, bitstreams, per-case CSV metrics, plots, source/code hashes, and a
validator result. The validator must check bitstream-only decoding, finite
values, QR residual orthogonality in the selected finite-grid space, exact
budget accounting, source hashes, and artifact hashes.

The primary reported quantities are reconstruction MSE and actual serialized
bytes/bits. A result is a pilot observation over four representative cases and
12 windows per case; it is not a claim of universal adaptive-TM superiority.

## Limits and falsification

The run must retain the distinction between waveform compression and input
recovery. It must state that VAE superiority is unmeasured, greedy optimality is
unproved, and finite precision is part of the codec condition. A fixed or DFT
baseline with lower MSE at the same measured bit budget falsifies the primary
pilot expectation for that budget; a codec reconstruction mismatch falsifies
the bitstream-only implementation contract.

## Execution gate

Do not run the pilot until the audit reviewer accepts the conditions above and
the implementation review confirms that only this run directory and the
existing E1 TM test file are modified.


## Observed pilot results

The validator passed 432 primary rows (48 windows x 3 methods x 3 byte budgets). Actual bytes were fixed TM/Fourier 48, 80, 144 and adaptive 43, 79, 142 because adaptive pole indices use 9 bytes per term. Mean decoded MSE by method and requested budget was fixed 0.715074/0.668318/0.598507, Fourier 0.702869/0.632138/0.524760, and adaptive 0.686948/0.633493/0.585196 for B=48/80/144. The raw complex64 control was 1040 bytes with mean MSE 4.09e-16. These are finite-window observations over one representative orbit per beta and do not establish adaptive TM superiority.
