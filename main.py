import argparse

import os
import glob
from pathlib import Path

import lightning as L
import torch

from data import DummyDataModule
from train import LightningAE

from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, required=True, help='experiment name')
    parser.add_argument('--version', type=str, required=False, help='experiment version')
    parser.add_argument('--mode', type=str, required=True, help='train/test/predict/export')
    parser.add_argument('--resume', action='store_true', required=False, help='resume from latest checkpoint')
    parser.add_argument('--ckpt', type=str, default=None, required=False, help='path to checkpoint')
    parser.add_argument('--export_type', type=str, required=False, help='ONNX or TorchScript')
    parser.add_argument('--data_dir', type=str, required=False, help='path to dataset directory')
    parser.add_argument('--n_train', type=float, default=0.7, required=False, help='number of images for training (%)')
    parser.add_argument('--n_val', type=float, default=0.15, required=False, help='number of images for validation (%)')
    parser.add_argument('--n_test', type=float, default=0.15, required=False, help='number of images for testing (%)')
    parser.add_argument('--img_width', type=int, default=64, required=False, help='image width')
    parser.add_argument('--img_height', type=int, default=64, required=False, help='image height')
    parser.add_argument('--img_channels', type=int, default=3, required=False, help='image channels')
    parser.add_argument('--batch_size', type=int, default=32, required=False, help='batch size')
    parser.add_argument('--max_epochs', type=int, default=1000, required=False, help='max epochs for training')
    parser.add_argument('--lr', type=float, default=5e-5, required=False, help='learning rate')
    parser.add_argument('--accelerator', type=str, default="auto", required=False, help='cpu/gpu/tpu/..')
    parser.add_argument('--nodes', type=int, default=1, required=False, help='number of cluster nodes')
    parser.add_argument('--devices', type=str, default="auto", required=False, help='number of cpus/gpus/tpus/.. to use')
    parser.add_argument('--strategy', type=str, default="auto", required=False, help='multi-node strategy (ddp/fsdp/deepspeed_stage_1-3)')
    parser.add_argument('--precision', type=str, default="32-true", required=False, help='32/16/32-true/16-true/16-mixed/bf16-mixed')
    parser.add_argument('--workers', type=int, default=4, required=False, help='number of dataloader workers')
    parser.add_argument('--pin_memory', action='store_true', required=False, help='pin application memory')
    args = parser.parse_args()

    model = None
    ckpt_latest = None
    ckpt = args.ckpt is not None and args.ckpt.endswith('.ckpt')
    resume = False
    if args.resume:
        if args.version is None:
            print('No experiment version specified')
            quit()
        ckpts = glob.glob("./logs/" + args.experiment + "/" + args.version + "/checkpoints/*")
        ckpt_latest = max(ckpts, key=os.path.getctime)
        print('Latest checkpoint is', ckpt_latest)
        resume = True
        print('Resuming session...')
        if Path(ckpt_latest).is_file() and args.mode == "export":
            model = LightningAE.load_from_checkpoint(
                ckpt_latest,
                batch_size=args.batch_size,
                img_channels=args.img_channels,
                img_height=args.img_height,
                img_width=args.img_width,
                lr=args.lr)
    else:
        if ckpt:
            if args.version is None:
                print('No experiment version specified')
                quit()
            print('Loading checkpoint', args.ckpt)
            if Path(args.ckpt).is_file() and args.mode == "export":
                model = LightningAE.load_from_checkpoint(
                    args.ckpt,
                    batch_size=args.batch_size,
                    img_channels=args.img_channels,
                    img_height=args.img_height,
                    img_width=args.img_width,
                    lr=args.lr)
        else:
            print('Starting new session...')
            
    if model is None:
        model = LightningAE(
            img_channels=args.img_channels,
            img_height=args.img_height,
            img_width=args.img_width,
            lr=args.lr)

    if args.mode == "export":
        Path("./exports").mkdir(parents=True, exist_ok=True)
        if not args.export_type:
            input_sample = torch.randn((1, args.img_channels, args.img_height, args.img_width))
            model.to_onnx('./exports/model.onnx', input_sample, export_params=True)
        else:
            if args.export_type == 'ONNX':
                input_sample = torch.randn((1, args.img_channels, args.img_height, args.img_width))
                model.to_onnx('./exports/model.onnx', input_sample, export_params=True)
            elif args.export_type == 'TorchScript':
                script = model.to_torchscript()
                torch.jit.save(script, "./exports/model.pt")
    else:
        if not args.data_dir:
            print('No dataset directory specified')
            quit()

        data = DummyDataModule(
            data_dir=args.data_dir,
            num_workers=args.workers,
            pin_memory=args.pin_memory,
            batch_size=args.batch_size,
            img_width=args.img_width,
            img_height=args.img_height,
            n_train=args.n_train,
            n_val=args.n_val,
            n_test=args.n_test)
        if args.mode == "predict":
            data.setup(args.mode)
        else:
            data.setup()
        
        Path("./logs").mkdir(parents=True, exist_ok=True)
        if not args.version:
            logger = TensorBoardLogger("logs", name=args.experiment)
        else:
            logger = TensorBoardLogger("logs", name=args.experiment, version=args.version)
        trainer = L.Trainer(accelerator=args.accelerator,
                            num_nodes=args.nodes,
                            devices=args.devices,
                            strategy=args.strategy,
                            precision=args.precision,
                            logger=logger,
                            max_epochs=args.max_epochs,
                            callbacks=[EarlyStopping(monitor="val_loss", mode="min")])

        if args.mode == 'train':
            if resume:
                trainer.fit(model, data.train_dataloader(), data.val_dataloader(), ckpt_path=ckpt_latest)
            elif ckpt:
                trainer.fit(model, data.train_dataloader(), data.val_dataloader(), ckpt_path=args.ckpt)
            else:
                trainer.fit(model, data.train_dataloader(), data.val_dataloader())
        elif args.mode == 'test':
            if resume:
                trainer.test(model, data.test_dataloader(), ckpt_path=ckpt_latest)
            elif ckpt:
                trainer.test(model, data.test_dataloader(), ckpt_path=args.ckpt)
            else:
                print('No checkpoint loaded')
        elif args.mode == 'predict':
            if resume:
                trainer.predict(model, data.predict_dataloader(), ckpt_path=ckpt_latest)
            elif ckpt:
                trainer.predict(model, data.predict_dataloader(), ckpt_path=args.ckpt)
            else:
                print('No checkpoint loaded')