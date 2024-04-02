# Lightning Project Template

A starter template for creating PyTorch Lightning projects.

## Features

- **Focus on model creation:** Simple and clean *separation of concerns* that maps each file with a single unit of work and lets you focus on model creation.

- **Includes examples of:**

    - Custom dataset creation and data transformations.

    - Automatic checkpointing and resume training.

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
pip install pillow torch torchvision tensorboard onnx lightning
```

*\* Check torch-cuda compatibility and install the right torch/torchvision for your machine*

*Example dataset from https://stylegan-human.github.io*

## Usage

1. Clone this repo.

2. Change origin to target your remote repo.

3. Create and train your own model, customise to your needs.

    - Start a new training session:

        ```
        python main.py --data_dir path-to-data --experiment my-experiment --mode train --n_train 32000 --n_val 4800 --n_test 3200
        ```

    - Resume a training session:

        ```
        python main.py --data_dir path-to-data --experiment my-experiment --version 0.0.1 --mode train --n_train 32000 --n_val 4800 --n_test 3200 --resume
        ```

    - Test a checkpoint:

        ```
        python main.py --data_dir path-to-data --experiment my-experiment --mode test --n_train 32000 --n_val 4800 --n_test 3200 --ckpt path-to-checkpoint
        ```

        or the latest:

        ```
        python main.py --data_dir path-to-data --experiment my-experiment --version my-version --mode test --n_train 32000 --n_val 4800 --n_test 3200 --resume
        ```

    - Export a checkpoint to ONNX:

        ```
        python main.py --data_dir path-to-data --experiment my-experiment --mode export --n_train 32000 --n_val 4800 --n_test 3200 --ckpt path-to-checkpoint
        ```

        or the latest:
        
        ```
        python main.py --data_dir path-to-data --experiment my-experiment --version my-version --mode export --n_train 32000 --n_val 4800 --n_test 3200 --resume
        ```

    
## License

This project is released under the [MIT License](LICENSE).
