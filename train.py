import lightning as L
import torch

from torchvision.utils import make_grid

from model import AE

class LightningAE(L.LightningModule):
    def __init__(self,
                 img_channels,
                 img_height,
                 img_width,
                 lr):
        super().__init__()
        self.lr = lr
        self.model = AE(
            c=img_channels,
            h=img_height,
            w=img_width)

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch):
        inputs, target = batch
        output = self(inputs)
        loss = torch.nn.functional.mse_loss(output, target['silhouette'])
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch):
        inputs, target = batch
        output = self(inputs)
        loss = torch.nn.functional.mse_loss(output, target['silhouette'])
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        grid_generated = make_grid(output)
        self.logger.experiment.add_image("generated_images_val", grid_generated, self.global_step)
        return loss
    
    def test_step(self, batch):
        inputs, target = batch
        output = self(inputs)
        loss = torch.nn.functional.mse_loss(output, target['silhouette'])
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        grid_generated = make_grid(output)
        self.logger.experiment.add_image("generated_images_test", grid_generated, self.global_step)
        return loss
    
    def predict_step(self, batch):
        inputs, _ = batch
        output = self(inputs)
        grid_generated = make_grid(output)
        self.logger.experiment.add_image("generated_images_pred", grid_generated, self.global_step)

    def configure_optimizers(self):
        return torch.optim.SGD(self.model.parameters(), lr=self.lr)