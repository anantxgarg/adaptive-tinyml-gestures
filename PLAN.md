# Plan: Adaptive TinyML Gesture Recognizer (on-device personalization)

## Problem
A wrist/handheld IMU (3-axis accelerometer, 50 Hz) recognizes gestures. A model trained
in the "factory" on a set of users degrades when deployed to a new user who wears the
device rotated, moves faster, and with smaller amplitude. The device must **adapt on its
own, on the MCU**, using a handful of labeled examples — no cloud, no retraining server.
It should also **learn a brand-new gesture** the factory model never saw.

## Constraints (target: Cortex-M4F class MCU, e.g. nRF52840 / STM32L4)
- Flash budget < 8 KB for model + code data, RAM < 4 KB for model state.
- No dynamic allocation, no external libraries; pure C99.
- Integer (int8) inference for the frozen part; tiny float head for learning (M4F has an FPU).

## Architecture
```
 accel window (100 x 3)  ->  23 handcrafted features  ->  int8 backbone 23->32->16 (frozen, flash)
                                                          -> float head 16->K   (trainable, RAM)
```
- **Features** (identical in Python & C): per-axis mean/std/min/max/mean|diff|/zero-crossing
  rate, magnitude mean/std, axis correlations.
- **Backbone**: MLP, post-training int8 quantization (symmetric per-tensor, int32 accumulators,
  float requantization multipliers). Stays frozen.
- **Adaptive head**: linear softmax layer, up to 8 class slots, trained on-device with SGD.
- **Replay buffer**: per-class ring of int8 embeddings, pre-seeded with a few factory exemplars
  (stored in flash). User samples overwrite factory ones, which prevents catastrophic
  forgetting of classes the user hasn't re-taught yet.
- **Regularization**: L2 pull toward factory head weights.
- **Confidence output**: low-confidence predictions can trigger "please label this" prompts.

## Steps
1. `python/data.py` - synthetic IMU gesture generator with per-user style (orientation,
   speed, amplitude, bias, noise). Gestures: idle, wave, circle, shake, punch (+ `flip`
   held out as the "new" gesture).
2. `python/features.py` - feature extraction (mirrors C).
3. `python/train.py` - numpy MLP training, int8 quantization, Python int8 reference
   inference, export `firmware/model_data.h` (weights, scales, normalization, exemplars).
4. `firmware/` - `features.c`, `tinyml.c` (int8 inference + on-device learning),
   `main.c` (device simulator with a line protocol over stdin), `Makefile`.
5. `python/demo.py` - end-to-end evaluation driving the compiled firmware:
   - parity check: C int8 vs Python int8 predictions
   - factory accuracy on training-like users vs. on the new user
   - learning curve: accuracy after 1, 2, 3, 5, 8 shots per class
   - add the new `flip` gesture on-device, accuracy on all 6 classes
   - memory footprint report
6. `README.md` - how to run, and how to port to a real board (IMU ring buffer -> `tml_predict`).

## Success criteria
- C and Python int8 predictions agree on >= 99% of windows.
- New-user accuracy improves substantially after <= 5 shots/class.
- New gesture learned with <= 5 shots without destroying the old classes.
- Model + learning state fits the flash/RAM budget above.
