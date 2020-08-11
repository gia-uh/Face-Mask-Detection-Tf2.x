# Face-Mask-Detection

> Adapted from https://github.com/TanyaChutani/Face-Mask-Detection-Tf2.x

This project has been transformed from a Jupyter notebook to a couple of Python scripts to make it easier to train and deploy.
Plus, all dependencies have been packed in a Docker image.

## Instructions

1. Run `docker-compose` build to create the Docker image.
2. **(Optional)** Run `scripts/train.py` to train the network from scratch. You will need:
    - The training data (in `data/train`)
    - Optionally, the testing data (in `data/test`)
3. Run `scripts/run.py <image.jpg>` to annotate one image. You will need:
    - The training weights (in `data/mask_classification_model.h5`)
4. The output image will be in `data/output.jpg`.

If you want to use it as a module, the `run.py` file exposes a method `annotate_image(path)` that does all the heavy work.

## Data

All necessary data is shared in a [release](https://github.com/gia-uh/face-mask-detect/releases/tag/v0.1).
