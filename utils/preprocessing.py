import numpy as np
from PIL import Image


def preprocess_image_onnx(image: Image.Image):
	"""
	Preprocesses a PIL Image object for ONNX inference without using OpenCV.

	Args:
		image: A PIL Image object (e.g., from Image.open(io.BytesIO(bytes))).

	Returns:
		A numpy array ready for ONNX inference (shape: 1, C, H, W).
	"""
	# 1. Convert to RGB and Resize (H, W) to (128, 128)
	# The .convert("RGB") handles Grayscale/RGBA conversion automatically.
	img = image.convert("RGB").resize((128, 128))

	# 2. Convert to NumPy array (shape: H, W, C)
	img_array = np.array(img)

	# 3. Normalize (from 0-255 to 0-1) and convert to float32
	img_normalized = img_array.astype(np.float32) / 255.0

	# 4. Transpose to (C, H, W) format (PyTorch/ONNX standard)
	img_transposed = np.transpose(img_normalized, (2, 0, 1))

	# 5. Add batch dimension (1, C, H, W)
	final_input = np.expand_dims(img_transposed, axis=0)

	return final_input