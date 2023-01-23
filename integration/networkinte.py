import torch 
import numpy as np 
import TorchSUL.Model as M 

class IntegrationNet(M.Model):
	def initialize(self):
		self.fc1 = M.Dense(512, activation=M.PARAM_GELU)
		self.fc2 = M.Dense(512, activation=M.PARAM_GELU)
		self.fc3 = M.Dense(512, activation=M.PARAM_GELU)
		self.fc4 = M.Dense(2)

	def forward(self, pts):
		x = torch.cat([pts], dim=1)
		x = self.fc1(x)
		x = self.fc2(x)
		x = self.fc3(x)
		x = self.fc4(x)

		bsize = pts.shape[0]
		pts = pts.reshape(bsize, 2, -1)
		w_pt = x
		w_pt = torch.sigmoid(w_pt)
		pts = w_pt * pts[:,0] + (1 - w_pt) * pts[:,1]  # use a weighted-sum term to increase the robustness
		pts = pts.reshape(bsize, 3, 14)

		return pts