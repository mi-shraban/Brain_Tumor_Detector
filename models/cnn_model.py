import torch.nn as nn


class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()
		self.cnn_model = nn.Sequential(
			# 1st conv layer
			nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),
			nn.Tanh(),
			nn.AvgPool2d(kernel_size=2, stride=5),
			# 2nd conv layer
			nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
			nn.Tanh(),
			nn.AvgPool2d(kernel_size=2, stride=5),
		)
		self.fc_model = nn.Sequential(
			# 1st linear layer
			nn.Linear(in_features=256, out_features=120),
			nn.Tanh(),
			# 2nd linear layer
			nn.Linear(in_features=120, out_features=84),
			nn.Tanh(),
			# 3rd linear layer
			nn.Linear(in_features=84, out_features=4)
		)

	def forward(self, x):
		x = self.cnn_model(x)
		x = x.view(x.size(0), -1)
		x = self.fc_model(x)
		return x