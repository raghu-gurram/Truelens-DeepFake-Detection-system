from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import uvicorn

app = FastAPI()

# Load the model
model = tf.keras.models.load_model('models/xception_model.h5')

def preprocess_image(file: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(file)).convert("RGB")
    image = image.resize((299, 299))
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")

    try:
        image_bytes = await file.read()
        image_array = preprocess_image(image_bytes)

        # Predict
        prediction = model.predict(image_array)
        result = {"prediction": int(prediction[0][0] > 0.5)}
        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("your_script_name:app", host="0.0.0.0", port=8000, reload=True)
