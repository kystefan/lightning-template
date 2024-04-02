import torch.nn as nn

class AE(nn.Module):
    def __init__(self, c,h,w):
        super(AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(c*h*w, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(), 
            nn.Linear(512, 256))
        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, c*h*w),
            nn.Tanh())

    def forward(self, inputs):
        _, c,h,w = inputs.shape
        z = self.encoder(inputs.view(-1, c*h*w))
        outputs = self.decoder(z)
        return outputs.view(-1, c,h,w)