# SmallFlood-CDNet

Implementation scaffold for lightweight, small-region-aware urban flood change detection.

The primary contract is a sub-5M-parameter Siamese model with size-aware supervision,
thin boundary refinement, event-held-out evaluation, and TensorRT FP16 deployment on a
Jetson Orin Nano.

The raw dataset is intentionally not bundled. See `data/README.md` for the expected layout.

