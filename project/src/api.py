from pathlib import Path
import shutil

from fastapi import FastAPI, UploadFile, File

from src.predict import predict_image


app = FastAPI(
    title="Chest X-Ray Pneumonia Detection API"
)


UPLOAD_DIR = Path("temp")
UPLOAD_DIR.mkdir(exist_ok=True)


@app.get("/")
def root():
    return {
        "message": "Pneumonia Detection API is running"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    temp_path = UPLOAD_DIR / file.filename

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = predict_image(str(temp_path))

    return result