from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import numpy as np
import tensorflow as tf

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

MODEL = tf.keras.models.load_model(
    "/Users/yesaswimadabattula/Documents/helmet_nohelmet_dl/saved_models/hybrid_model_resnet_mobilenet_efficientnet.keras"
)
CLASS_NAMES = ["with_helmet", "without_helmet"]
IMAGE_SIZE = 256

def preprocess_image_tf(byte_data):
    image = tf.io.decode_image(byte_data, channels=3, expand_animations=False)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    return image

@app.get("/", response_class=HTMLResponse)
async def form_get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.post("/", response_class=HTMLResponse)
async def predict_image(request: Request, file: UploadFile = File(...)):
    byte_data = await file.read()
    image_tensor = preprocess_image_tf(byte_data)
    image_batch = tf.expand_dims(image_tensor, axis=0)

    predictions = MODEL.predict(image_batch)
    pred_index = int(np.argmax(predictions[0]))
    predicted_class = CLASS_NAMES[pred_index]
    confidence = float(np.max(predictions[0])) * 100

    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": {
            "class": predicted_class,
            "confidence": f"{confidence:.2f}%"
        }
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=5000)
