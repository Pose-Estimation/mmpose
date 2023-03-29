import torch
import numpy as np
import TorchSUL.Model as M


class IntegrationNet(M.Model):

    def initialize(self):
        self.fc1 = M.Dense(512, activation=M.PARAM_GELU)
        self.fc2 = M.Dense(512, activation=M.PARAM_GELU)
        self.fc4 = M.Dense(42)
        self.dropout = torch.nn.Dropout(0.1)
        self.sigmoid = torch.sigmoid

    def forward(self, pts):
        x = torch.cat([pts], dim=1)
        x = self.dropout(self.fc1(x))
        x = self.dropout(self.fc2(x))
        x = self.sigmoid(self.fc4(x))

        return x.reshape(pts.shape[0], 3, 14)
