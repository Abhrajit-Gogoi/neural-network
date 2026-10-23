import io
import base64
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout, Input

app = Flask(__name__)

# Rebuild your exact model architecture
model = Sequential([
    Input(shape=(784,)),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])
# Load your pre-trained weights
model.load_weights('model.weights.h5')

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    image_data_url = data["image"]
    
    # Decode base64 image from frontend canvas
    header, encoded = image_data_url.split(",", 1)
    image_bytes = base64.b64decode(encoded)
    
    # Open image, convert to grayscale, and resize to 28x28 pixels
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    image = image.resize((28, 28))
    
    # Convert to array and normalize (0.0 to 1.0)
    img_array = np.array(image).astype(np.float32) / 255.0
    
    # Flatten to match your model's input_shape = (784,) with batch size 1 -> (1, 784)
    flat_input = img_array.reshape(1, 784)
    
    # Run prediction through your model
    predictions = model.predict(flat_input)
    predicted_digit = int(np.argmax(predictions[0]))
    
    return jsonify({"prediction": predicted_digit})

if __name__ == "__main__":
    app.run(debug=True)