from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import torch
import torch.nn as nn
import numpy as np
import cv2
import io
from PIL import Image
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.cnn_model import CNN
from utils.preprocessing import preprocess_image

app = FastAPI(title="Brain Tumor Detector API")

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Device configuration
device = torch.device('cpu')  # Vercel doesn't support GPU

# Load model
model = None


def load_model():
	global model
	if model is None:
		model = CNN().to(device)
		checkpoint = torch.load('tumor_detector.pth', map_location=device)
		model.load_state_dict(checkpoint['model_state_dict'])
		model.eval()
	return model


# Label mapping
LABELS = {
	0: 'No Tumor',
	1: 'Glioma',
	2: 'Meningioma',
	3: 'Pituitary'
}


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
		image = Image.open(io.BytesIO(contents))

		# Convert PIL Image to numpy array
		img_array = np.array(image)

		# Preprocess image
		processed_img = preprocess_image(img_array, device)

		# Load model and predict
		model = load_model()

		with torch.no_grad():
			output = model(processed_img)
			probabilities = torch.softmax(output, dim=1)
			pred_idx = torch.argmax(probabilities, dim=1).item()
			confidence = probabilities[0][pred_idx].item()

		# Prepare response
		result = {
			"prediction": LABELS[pred_idx],
			"confidence": float(confidence),
			"probabilities": {
				LABELS[i]: float(probabilities[0][i].item())
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
	return {"status": "healthy"}