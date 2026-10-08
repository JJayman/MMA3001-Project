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
