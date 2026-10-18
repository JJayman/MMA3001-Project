# Run time and memory of every method

Session 13. `python -m treeskel.benchmark data/raw --trees 8` on the student's PC: NVIDIA GeForce GTX
1080 Ti, an 8-thread Intel CPU (family 6, model 158), Windows 11, PyTorch 2.14 (CUDA 12.6). Output:
`output.txt`; all numbers: `report.json`.

## How it was measured

- **The same way for every method, on one machine.** Each time is the median of 10 runs after warm-up
  runs, with the GPU synchronised before the clock is read (5 for the ground truths: 3 per tree on 8
  development trees; 2 for training steps on the CPU).
- **Networks get random weights and random photos:** time and memory do not depend on the values, and
  nothing needs downloading.
- **Prediction** is one photo and its mirror image, at batch 1, as every model here predicts.
- **A training step** is at the recipe's batch: 16 photos for the angle CNN, 8 for the U-Net, with AdamW.
- **Memory:** the GPU's peak (weights, activations and, in training, gradients and optimiser state);
  for the CPU methods, the peak of what Python and NumPy allocate.
- **Check of the method:** one U-Net trained on all 449 trees for 34 epochs, estimated from the measured
  step, takes 2.00 min. The final test recorded 2.0 min.

## Results

| Method | Accuracy (development, V-opening skill) | Parameters | Prediction per photo: GPU / CPU | Peak GPU memory: predict / train | Training |
|---|---|---|---|---|---|
| Mean shape | 0 (by definition) | 10 | – | – | 50 µs |
| Ridge on fixed ResNet-18 features | +0.10 | 30,730 (+ 11.2 M fixed) | 5.9 / 51 ms (the features) | 75 MB / – | 71 s for 5 folds (CPU), features of 449 photos 3 s (GPU) |
| Angle CNN 240x320 | +0.16 | 11.2 M (45 MB) | 6.1 / 53 ms | 83 MB / 0.80 GB | 0.7 min one network; 6.2 min 5 folds |
| Angle CNN 480x640 | +0.18 | 11.2 M (45 MB) | 7.7 / 181 ms | 110 MB / 2.46 GB | 2.0 min; 19.2 min |
| **U-Net 256x256 (named)** | **+0.75** | 14.3 M (57 MB) | **9.7 / 116 ms** (+14 ms joining and measuring, CPU) | 109 MB / 0.93 GB | **2.0 min; 20.4 min** |
| U-Net 480x640 | +0.74 | 14.3 M (57 MB) | 18.7 / 463 ms | 191 MB / 3.66 GB | 5.9 min; 69.7 min |

"One network": all 449 development trees at the median epochs the development run chose, estimated from
the measured step. "5 folds": the recorded development run, each fold with its inner fold choosing the
epochs.

The ground truths, per winter tree (CPU):

| Method | Accuracy | Time per tree | Peak memory |
|---|---|---|---|
| Automatic baseline (`winter_gt`) | traced on 43% of trees (`docs/results/winter_baseline/`) | 0.16 s | 13 MB |
| Guided tracing (`guided_gt`) | the ground truth | 1.2 s + about 15 s of clicking | 42 MB |

## What it shows

- **The named U-Net costs little more than the much weaker models:** 3.6 ms more per photo on the GPU
  than the angle CNN at 240x320, and 2 minutes to train one network, as long as the angle CNN at
  480x640. Its gain in accuracy (+0.75 against +0.16–0.18) does not come from more computation.
- **The full-size U-Net costs 3x to train and 4x the memory, for no gain** (+0.74): the naming rule's
  choice of the 256x256 network was also the cheaper one.
- **It would run on a robot without a GPU:** 0.13 s per photo on this CPU with the joining and
  measuring, about 7 photos a second; with a GPU, under 0.2 GB of its memory.
- **All the networks fit an ordinary GPU:** the largest training run needs 3.7 GB.
- **The ground truth's cost is the person:** the guided tracing computes in 1.2 s per tree, but the
  clicks took about 15 s per tree; the automatic baseline is 8x faster still but right on only 4 trees
  in 10.

## Limits

- One machine; times on another GPU or CPU scale, but the comparison between methods should hold.
- Prediction at batch 1 is what a robot taking one photo at a time does; batched prediction is faster
  per photo.
- The CPU memory of the networks (weights plus activations) is not measured; their weights take 45–57 MB.

## FLOPs, memory traffic and the profiler (session 18)

`python -m treeskel.benchmark --flops` explains the times above: `flops.json` (all numbers) and
`flops_output.txt`. It leaves `report.json` alone and reads its prediction times.

**The files here come from the assistant's environment** (no GPU, 4 CPU threads, PyTorch 2.14.1). The
FLOP counts, the traffic and the roofline do not depend on the machine. The speeds on the GTX 1080 Ti and
the student's CPU come from `report.json`. Only the GPU profile is missing: kernels launched and the time
the GPU spends running them. To add it, run the same command on the student's PC, which overwrites both
files with that machine's run.

### How it was measured

- **FLOPs:** PyTorch's FLOP counter (`torch.utils.flop_counter.FlopCounterMode`) on one forward pass at
  batch 1, random weights. It counts convolutions and matrix products, 2 FLOPs per multiply-add, from the
  tensor shapes. Batch norm, ReLU, additions, pooling and interpolation are not counted: about 0.5% of
  the convolutions' count in the named U-Net (a few FLOPs per element they write). A prediction is twice
  this, because the mirror image is predicted too.
- **Memory traffic:** every operation PyTorch dispatches is taken to read all its input tensors from
  memory, weights included, and to write all its outputs back. Views move nothing and are left out.
  Real kernels may fuse steps or find data still in cache. Caching can only help a little here: the
  GTX 1080 Ti's L2 cache holds 2.75 MB, and the U-Net's largest activation is 8.4 MB (the decoder's
  joined features at half size, 128 channels). So this is an upper bound on the traffic, and the
  intensity is a lower bound. It is a model, not a hardware measurement; NVIDIA's Nsight Compute would
  measure the bytes.
- **Roofline:** the GTX 1080 Ti's rated peaks, from NVIDIA's specifications: 11.34 TFLOP/s in 32-bit
  (3584 cores x 2 x 1582 MHz boost) and 484 GB/s (11 Gbit/s x 352 bits). Its ridge point is
  23.4 FLOP/byte. Work above the ridge is limited by arithmetic, not by memory. At the peak rates, a
  prediction needs at least FLOPs / peak seconds of compute and bytes / bandwidth seconds of memory
  traffic.
- **Profiler:** `torch.profiler` over one prediction after two warm-up predictions. It counts the
  PyTorch operators called (the outermost `aten::` calls). On a GPU it also counts the kernels and sums
  their durations, through NVIDIA's CUPTI. "GPU busy" is that sum divided by the timed prediction.

### Results

| Network | GFLOP per pass | MB moved per pass | FLOP/byte | Operators per photo | GPU (report.json): ms per photo, share of peak | CPU (report.json): GFLOP/s |
|---|---|---|---|---|---|---|
| ResNet-18 features (ridge) | 5.64 | 152 | 37.0 | 133 | 5.9 ms, 17% | 220 |
| Angle CNN 240x320 | 5.64 | 152 | 37.0 | 139 | 6.1 ms, 16% | 214 |
| Angle CNN 480x640 | 22.21 | 474 | 46.8 | 139 | 7.7 ms, 51% | 245 |
| **U-Net 256x256 (named)** | **10.78** | **313** | **34.4** | **215** | **9.7 ms, 20%** | **186** |
| U-Net 480x640 | 50.53 | 1256 | 40.2 | 215 | 18.7 ms, 48% | 218 |

At the peak rates, a prediction of the named U-Net (21.56 GFLOP, 628 MB) needs 1.90 ms of arithmetic and
1.30 ms of memory traffic. It takes 9.7 ms.

### What it shows

- **Memory bandwidth limits none of the networks.** All are above the ridge point (34–47 FLOP/byte against
  23.4), even with the traffic overcounted.
- **At batch 1, the GPU is mostly not doing arithmetic.** The 256x256 U-Net reaches 20% of the peak and the
  240x320 angle CNN 16%. The 480x640 networks reach 48–51%: the U-Net does 4.7x the work in 1.9x the time.
  Neither the arithmetic (1.90 ms) nor the memory (1.30 ms) accounts for the named U-Net's 9.7 ms. Most of
  the rest should be fixed costs per operation: 215 operators per photo, each launching at least one
  kernel, whose cost does not shrink with the input. The GPU profile from the student's PC will measure
  how much of the 9.7 ms the GPU spends running kernels.
- **On the CPU, time follows the work.** Every network runs at about the same speed (186–245 GFLOP/s), so a
  robot without a GPU gains almost fully from the smaller input.
- **These are the report's figures** (Section 7.0, Appendix A): 10.8 GFLOP per pass, 313 MB, about
  34 FLOP/byte, 20% of the peak, 1.9 ms and 1.3 ms at the peak rates, about 190 and 220 GFLOP/s on the CPU.
  `tests/test_benchmark.py` checks them. The report's "more than 200 kernel launches per photo" is the
  operator count (215). It is not a profiler's count of kernels; the GPU profile will give that count.

### Limits

- Batch 1 only, as in the timings.
- The FLOP counter leaves out the element-wise operations (about 0.5%).
- The traffic is a model with no fusion or caching, not a hardware measurement.
- The peak rates are rated values at the boost clock; the clock a card actually runs at varies.
