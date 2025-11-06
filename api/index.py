# from fastapi import FastAPI, File, UploadFile, Request
# from fastapi.responses import HTMLResponse, JSONResponse
# from fastapi.staticfiles import StaticFiles
# from fastapi.templating import Jinja2Templates
# import numpy as np
# import io
# from PIL import Image
# import sys
# import os
# import onnxruntime as ort
#
# # Add parent directory to path
# sys.path.append(os.path.dirname(os.path.dirname(__file__)))
#
# from utils.preprocessing import preprocess_image_onnx
#
# app = FastAPI(title="Brain Tumor Detector API")
#
# # Mount static files and templates
# app.mount("/static", StaticFiles(directory="static"), name="static")
# templates = Jinja2Templates(directory="templates")
#
# # Load ONNX model
# ort_session = None
#
#
# def load_model():
# 	global ort_session
# 	if ort_session is None:
# 		# Load ONNX model
# 		ort_session = ort.InferenceSession(
# 			'tumor_detector.onnx',
# 			providers=['CPUExecutionProvider']  # Vercel only supports CPU
# 		)
# 	return ort_session
#
#
# # Label mapping
# LABELS = {
# 	0: 'No Tumor',
# 	1: 'Glioma',
# 	2: 'Meningioma',
# 	3: 'Pituitary'
# }
#
#
# def softmax(x):
# 	"""Compute softmax values for array x"""
# 	exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
# 	return exp_x / np.sum(exp_x, axis=1, keepdims=True)
#
#
# @app.get("/", response_class=HTMLResponse)
# async def home(request: Request):
# 	"""Serve the main HTML page"""
# 	return templates.TemplateResponse("index.html", {"request": request})
#
#
# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
# 	"""
# 	Endpoint to predict brain tumor type from uploaded MRI scan
# 	"""
# 	try:
# 		# Read image file
# 		contents = await file.read()
# 		image = Image.open(io.BytesIO(contents))
#
# 		# Convert PIL Image to numpy array
# 		img_array = np.array(image)
#
# 		# Preprocess image (returns numpy array for ONNX)
# 		processed_img = preprocess_image_onnx(img_array)
#
# 		# Load model and predict
# 		session = load_model()
#
# 		# Run inference
# 		input_name = session.get_inputs()[0].name
# 		output_name = session.get_outputs()[0].name
#
# 		ort_inputs = {input_name: processed_img}
# 		ort_outputs = session.run([output_name], ort_inputs)
#
# 		# Get predictions
# 		output = ort_outputs[0]
# 		probabilities = softmax(output)
# 		pred_idx = np.argmax(probabilities, axis=1)[0]
# 		confidence = probabilities[0][pred_idx]
#
# 		# Prepare response
# 		result = {
# 			"prediction": LABELS[pred_idx],
# 			"confidence": float(confidence),
# 			"probabilities": {
# 				LABELS[i]: float(probabilities[0][i])
# 				for i in range(4)
# 			}
# 		}
#
# 		return JSONResponse(content=result)
#
# 	except Exception as e:
# 		return JSONResponse(
# 			status_code=500,
# 			content={"error": str(e)}
# 		)
#
#
# @app.get("/health")
# async def health_check():
# 	"""Health check endpoint"""
# 	return {"status": "healthy", "model_type": "ONNX"}

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import numpy as np
import io
from PIL import Image
import sys
import os
import onnxruntime as ort

# Add parent directory to path
# NOTE: This path adjustment might be specific to your local environment/IDE.
# For Vercel, ensuring your directory structure is correct is usually sufficient.
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Assuming utils/preprocessing.py is correctly resolved
from utils.preprocessing import preprocess_image_onnx

app = FastAPI(title="Brain Tumor Detector API")

# Mount static files and templates (Ensure 'static' and 'templates' directories exist)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load ONNX model
ort_session = None


def load_model():
	global ort_session
	if ort_session is None:
		# Load ONNX model
		ort_session = ort.InferenceSession(
			'tumor_detector.onnx',
			providers=['CPUExecutionProvider']  # Vercel only supports CPU
		)
	return ort_session


# Label mapping
LABELS = {
	0: 'No Tumor',
	1: 'Glioma',
	2: 'Meningioma',
	3: 'Pituitary'
}


def softmax(x):
	"""Compute softmax values for array x"""
	# Ensures numerical stability
	exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
	return exp_x / np.sum(exp_x, axis=1, keepdims=True)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
	"""Serve the main HTML page"""
	return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
	"""
	Endpoint to predict brain tumor type from uploaded MRI scan
	"""
	try:
		# Read image file
		contents = await file.read()

		# 1. Create the PIL Image object
		image = Image.open(io.BytesIO(contents))

		# 2. FIX: Pass the PIL Image object directly to the preprocessing function.
		# The function will handle the resizing, RGB conversion, and array conversion internally.
		processed_img = preprocess_image_onnx(image)

		# Load model and predict
		session = load_model()

		# Run inference
		input_name = session.get_inputs()[0].name
		output_name = session.get_outputs()[0].name

		ort_inputs = {input_name: processed_img}
		ort_outputs = session.run([output_name], ort_inputs)

		# Get predictions
		output = ort_outputs[0]
		probabilities = softmax(output)
		pred_idx = np.argmax(probabilities, axis=1)[0]
		confidence = probabilities[0][pred_idx]

		# Prepare response
		result = {
			"prediction": LABELS[pred_idx],
			"confidence": float(confidence),
			"probabilities": {
				LABELS[i]: float(probabilities[0][i])
				for i in range(4)
			}
		}

		return JSONResponse(content=result)

	except Exception as e:
		return JSONResponse(
			status_code=500,
			content={"error": str(e)}
		)


@app.get("/health")
async def health_check():
	"""Health check endpoint"""
	return {"status": "healthy", "model_type": "ONNX"}