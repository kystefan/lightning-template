import lightning as L
import torch
import torchmetrics

from torchvision.utils import make_grid

from model import AE

class LightningAE(L.LightningModule):
    def __init__(self,
                 img_channels,
                 img_height,
                 img_width,
                 lr):
        super().__init__()
        self.img_log_step = -1
        self.lr = lr
        self.img_channels = img_channels
        self.img_height = img_height
        self.img_width = img_width
        self.model = AE(
            c=img_channels,
            h=img_height,
            w=img_width)

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch):
        inputs, targets = batch
        outputs = self(inputs)
        loss = torch.nn.functional.mse_loss(outputs, targets['original'])
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch):
        inputs, targets = batch
        outputs = self(inputs)
        loss = torch.nn.functional.mse_loss(outputs, targets['original'])
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        if self.img_log_step != self.global_step:
            grid_generated = make_grid(outputs)
            self.logger.experiment.add_image("generated_images_val", grid_generated, self.global_step)
            self.img_log_step = self.global_step
        return loss
    
    @torch.jit.ignore
    def on_test_start(self):
        self.fid = torchmetrics.image.fid.FrechetInceptionDistance(normalize=True).to(self.device)
    
    @torch.jit.ignore
    def test_step(self, batch):
        inputs, targets = batch
        outputs = self(inputs)
        loss = torch.nn.functional.mse_loss(outputs, targets['original'])
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.fid.update(targets['original'], real=True)
        self.fid.update(torch.clamp(outputs,0,1), real=False)
        self.log("fid", self.fid, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        grid_generated = make_grid(outputs)
        self.img_log_step += 1
        self.logger.experiment.add_image("generated_images_test", grid_generated, self.img_log_step)
        return loss
    
    def predict_step(self, batch):
        inputs, _ = batch
        outputs = self(inputs)
        b,c,h,w = inputs.shape
        canvas = torch.zeros(2*b,c,h,w,dtype=outputs.dtype).to(self.device)
        canvas[range(0,2*b,2)] = inputs.type(outputs.dtype)
        canvas[range(1,2*b,2)] = outputs
        grid_generated = make_grid(canvas)
        self.img_log_step += 1
        self.logger.experiment.add_image("generated_images_pred", grid_generated, self.img_log_step)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)