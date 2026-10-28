# Adaptive TinyML Gesture Recognizer

A gesture recognizer for a 3-axis accelerometer that **adapts to its user on the
microcontroller itself**. A factory model ships in ~2.3 KB of flash. When a new user
wears the device differently (rotated, faster, smaller moves), accuracy drops. A few
labeled examples, taught on-device, recover it, and the user can add **brand-new
gestures** the factory never saw. No cloud and no retraining server.

See [PLAN.md](PLAN.md) for the design plan.

## Quick start
```bash
cd python
python3 demo.py        # trains, quantizes, exports C header, builds firmware, runs the demo
```
Requires only Python 3 + numpy and gcc. The demo drives the **compiled C firmware**
(`firmware/device_sim`) over a line protocol, exactly as an MCU would be driven.

## How it works
```
accel 100x3 (2 s @ 50 Hz)
  -> 23 features (stats, zero-crossings, magnitude, axis correlations)     features.c
  -> int8 MLP backbone 23 -> 32 -> 16   frozen, flash, ~1.2k int8 MACs        tinyml.c
  -> float linear head 16 -> K          trainable in RAM, K <= 8 classes      tinyml.c
```
**Adaptation (`tml_learn`)**: the window's int8 embedding goes into a per-class replay
ring buffer, and the head is trained with a few SGD epochs over the buffer.
- The replay buffer is **pre-seeded with factory exemplars** (4 per class, in flash), so
  teaching one class doesn't make the head forget the others.
- An L2 pull toward the factory head weights keeps learning stable with very few shots.
- `tml_add_class()` opens a new class slot; teaching it works the same way.

## Results (`python3 demo.py`)
| | accuracy |
|---|---|
| Factory model, factory-like users | 100% |
| Factory model, **new user** | **60%** (wave read as circle, punch as idle) |
| After **1 shot**/class on-device | 98.0% |
| After **2 shots**/class on-device | 99.7% |
| New gesture `flip`, 3 shots (all 6 classes) | 100% |

- **C vs Python int8 reference:** embeddings match bit for bit on 100% of windows.
- **Footprint:** 2,284 B flash (model constants), 1,588 B RAM (trainable head + replay buffer),
  about 1.3k MACs per inference, 1 inference every 2 s.
- **Held-out check:** the learning settings were tuned on the demo user. They were then
  confirmed on two other deployment styles (yaw -110°/slow/big, yaw 40°/fast/noisy) and
  fresh random data. Each reached ≥99.7% after 2 shots per class.

### Trade-offs
- **Personalization vs. generality.** Once a class has more than 4 user samples, they start
  overwriting that class's factory exemplars. The device becomes *this user's* device:
  accuracy on factory-style users falls from 100% to 80–89% after 8 shots per class.
  To keep both, reserve some replay slots for factory exemplars.
- **One shot is not always enough for a new gesture.** On one held-out user, `flip` was
  only 3% after 1 shot and 100% after 3. Plan on 3+ shots.
- **The factory model is confidently wrong** on the new user (0.86–0.99 mean confidence
  at 60% accuracy). So confidence alone can't detect the need to adapt; user feedback
  (a "wrong gesture" button) is the trigger.
- The data is **synthetic** (`python/data.py`). On real hardware, collect factory data
  from real users and retrain with `train.py`. Nothing else changes.

## Layout
```
python/data.py      synthetic IMU gestures with per-user style (rotation, speed, amplitude, bias, noise)
python/features.py  feature extraction (mirror of features.c)
python/train.py     numpy MLP training, int8 quantization, int8 reference, exports model_data.h
python/demo.py      end-to-end evaluation against the C firmware
firmware/features.c feature extraction
firmware/tinyml.c   int8 inference + on-device learning (no malloc, no libraries)
firmware/main.c     host simulator: stdin line protocol (P/E/L/A/R/I)
firmware/model_data.h  generated weights, scales, exemplars
```
Learning settings can be overridden at compile time, e.g.
`make CFLAGS="-std=c99 -O2 -DLEARN_RATE=0.02f -DLEARN_EPOCHS=15"`.

## Porting to a real board (e.g. nRF52840, STM32L4, ESP32-S3)
Compile `features.c`, `tinyml.c` and `model_data.h` into your project and replace `main.c`:
```c
static float win[TML_WIN][3]; static int n;
void imu_sample_isr(float ax, float ay, float az) {      // 50 Hz
    win[n][0] = ax; win[n][1] = ay; win[n][2] = az;
    if (++n == TML_WIN) { n = 0; window_ready = 1; }
}
// main loop
tml_init();
if (window_ready) {
    float conf; int g = tml_predict(win, &conf);
    if (user_pressed_correct_button) tml_learn(win, user_label);
}
```
To keep the adaptation across reboots, write `head_w`, `head_b` and the replay buffer
(~1.6 KB) to flash or EEPROM.
