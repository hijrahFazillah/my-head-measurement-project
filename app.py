# app.py

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import io
from pipeline import CoinMeasurementPipeline # Import your class

# --- MODEL PATHS ---
# These paths are relative to the root of your project folder
DETECTION_MODEL_PATH = "models/object_detection.torchscript"
CLASSIFICATION_MODEL_PATH = "models/image_classification_savedmodel"
LABELS_PATH = "models/image_classification_savedmodel/labels.txt"

# --- FASTAPI APP INITIALIZATION ---
app = FastAPI(title="Head Measurement API")

# --- LOAD MODELS ON STARTUP ---
# This is crucial for performance. The models are loaded only once when the
# server starts, not for every request.
try:
    pipeline = CoinMeasurementPipeline(
        detection_model_path=DETECTION_MODEL_PATH,
        classification_model_path=CLASSIFICATION_MODEL_PATH,
        labels_path=LABELS_PATH
    )
    print("Pipeline initialized successfully.")
except Exception as e:
    pipeline = None
    print(f"FATAL: Could not initialize pipeline. Error: {e}")

# --- API ENDPOINTS ---
@app.get("/", summary="Root endpoint to check server status")
def read_root():
    """A simple endpoint to confirm the server is running."""
    return {"status": "ok", "message": "Head Measurement API is running."}

@app.post("/process-image/", summary="Process an image to measure head circumference")
async def process_image_endpoint(file: UploadFile = File(...)):
    """
    Receives an image file, processes it through the pipeline, and returns
    the measurement results or an error message in JSON format.
    """
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Server Error: Pipeline is not available.")

    # 1. Read image data from the uploaded file
    # The 'await' keyword is used because this is an asynchronous operation.
    image_data = await file.read()
    
    # 2. Convert the image data (bytes) into a NumPy array that OpenCV can use
    try:
        image_np_array = np.frombuffer(image_data, np.uint8)
        image_cv = cv2.imdecode(image_np_array, cv2.IMREAD_COLOR)
        if image_cv is None:
            raise ValueError("Could not decode image. The file may be corrupt or in an unsupported format.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    # 3. Run the pipeline with the image
    print("Processing uploaded image...")
    result = pipeline.process_image(image_cv)
    print(f"Processing complete. Success: {result['pipelineSuccess']}")

    # 4. Return the result as a JSON response
    if result['pipelineSuccess']:
        return JSONResponse(content=result)
    else:
        # If the pipeline failed, return a 422 Unprocessable Entity error
        # with the specific error message from the pipeline.
        raise HTTPException(status_code=422, detail=result['errorMessage'])