from pathlib import Path
import shutil
import logging

from fastapi import FastAPI, UploadFile, File

from src.predict import predict_image


# logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(asctime)s | %(message)s"
)

logger = logging.getLogger(__name__)

# app
app = FastAPI(
    title="Chest X-Ray Pneumonia Detection API"
)


UPLOAD_DIR = Path("temp")
UPLOAD_DIR.mkdir(exist_ok=True)


# root
@app.get("/")
def root():
    logger.info("Root endpoint called")
    return {
        "message": "Pneumonia Detection API is running"
    }


# prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    logger.info(f"Received file: {file.filename}")
    temp_path = UPLOAD_DIR / file.filename

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = predict_image(str(temp_path))

        logger.info(
            f"Prediction completed | class={result['class']} "
            f"| confidence={result['confidence']}"
        )

        return result

    except Exception as e:
        logger.exception("Prediction failed")
        return {
            "error": str(e)
        }

# health check
@app.get("/health")
def health():
    logger.info("Health endpoint called")
    return {
        "status": "ok"
    }
