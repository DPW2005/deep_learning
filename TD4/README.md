# DeepLearning_4 — TP4: Advanced Vision, Segmentation and 3D Data

This repository contains the work for Practical Work 4 (TP4). The project
implements a small U-Net for semantic segmentation, Dice and IoU metrics,
and a demonstration Conv3D experiment. The code is intentionally minimal and
designed to run quickly with dummy data to validate the pipeline and MLflow
tracking.

## Contents

- src/unet_segmentation.py : U-Net implementation, Dice & IoU metrics, and a
	tiny training run on random data which logs to MLflow.
- src/conv3d_experiment.py : Simple Conv3D model and simulated MLflow logging.
- doc/Responses.MD : Written answers to the theoretical questions (in
	English) required by the TP.

1. Run the U-Net quick check (uses random data):

```bash
python unet_segmentation.py
```

2. Run the Conv3D simulated experiment:

```bash
python conv3d_experiment.py
```

## Author

**DONGMO PRINCE WILLIAMS**
M2-GI – Machine Learning Practical Work
