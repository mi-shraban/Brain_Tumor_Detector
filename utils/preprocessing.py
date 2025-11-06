import numpy as np
import cv2
import torch


def preprocess_image(image, device):
	"""
	Preprocess the input image for model prediction

	Args:
		image: numpy array of the image (RGB)
		device: torch device (cpu or cuda)

	Returns:
		Preprocessed image tensor
	"""
	# Resize to 128x128
	img = cv2.resize(image, (128, 128))

	# Ensure RGB format
	if len(img.shape) == 2:  # Grayscale
		img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
	elif img.shape[2] == 4:  # RGBA
		img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

	# Normalize to [0, 1]
	img = img.astype(np.float32) / 255.0

	# Transpose to (C, H, W) format
	img = np.transpose(img, (2, 0, 1))

	# Convert to tensor and add batch dimension
	img_tensor = torch.FloatTensor(img).unsqueeze(0).to(device)

	return img_tensor