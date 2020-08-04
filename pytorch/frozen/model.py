import torch
import torch.nn.functional as F

class IrisNet(torch.nn.Module):

    def __init__(self, input_dim, num_classes=3):
        super(IrisNet, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 16),
            torch.nn.Linear(16,16),
            torch.nn.Linear(16, num_classes)
        )

    def forward(self, x):
        return self.model(x)
