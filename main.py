from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
import json
import numpy as np
from tensorflow import keras

app = FastAPI()

model = keras.models.load_model('mobilenet.keras', compile=False)

with open("class_names_resnet.json", "r") as f:
    class_names = json.load(f)

@app.post('/predict')
async def predict(file: UploadFile = File(...)):

    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    image = image.resize((180, 180)) 

    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)

    # Предсказание класса
    prediction = model.predict(image_array)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    predicted_class_name = class_names[predicted_class_index]

    return {"predicted_class": predicted_class_name, "confidence": float(prediction[0][predicted_class_index])}