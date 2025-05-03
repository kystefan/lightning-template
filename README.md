# Lightning Project Template

A project template to help you create AI models on multi-node clusters with PyTorch Lightning.

## Features

- **Focus on model creation:** Simple and clean *separation of concerns* that maps each file with a single unit of work and lets you focus on model creation.

- **Includes examples of:**

    - Custom dataset creation and data transformations.

    - Automatic checkpointing and resume training.

    - Evaluation with TorchMetrics.

    - Logging with TensorBoard.

    - Early stopping.

    - Exporting the model.

## Structure

- **dataset.py:** custom dataset creation

- **data.py:** dataloader setup

- **model.py:** the model

- **train.py:** training setup

- **main.py:** parsing cli arguments

## Requirements

```
pip install pillow torch torchvision tensorboard onnx lightning torchmetrics[image] deepspeed
```

*\* Check torch-cuda compatibility and install the right torch/torchvision for your machine.*

*\*\* Example dataset is any arbitrary directory of images.*

## Usage

1. Create a new repo for your research project and make an initial commit.

2. Copy the contents of this repo to your new repo.

3. Create and train your own model, customise to your needs.

    - Start a new training session:

        ```
        python main.py --data_dir path-to-data --experiment my-experiment --mode train
        ```

    - Resume a training session:

        ```
        python main.py --data_dir path-to-data --experiment my-experiment --version my-version --mode train --resume
        ```

    - Test a checkpoint:

        ```
        python main.py --data_dir path-to-data --experiment my-experiment --version my-version --mode test --ckpt path-to-checkpoint
        ```

        or the latest:

        ```
        python main.py --data_dir path-to-data --experiment my-experiment --version my-version --mode test --resume
        ```

    - Predict with a checkpoint:

        ```
        python main.py --data_dir path-to-data --experiment my-experiment --version my-version --mode predict --ckpt path-to-checkpoint
        ```

        or the latest:

        ```
        python main.py --data_dir path-to-data --experiment my-experiment --version my-version --mode predict --resume
        ```

    - Export a checkpoint to ONNX:

        ```
        python main.py --experiment my-experiment --version my-version --mode export --ckpt path-to-checkpoint
        ```

        or TorchScript:
        
        ```
        python main.py --experiment my-experiment --version my-version --mode export --export_type TorchScript --ckpt path-to-checkpoint
        ```

    
## License

This project is released under the [MIT License](LICENSE).
