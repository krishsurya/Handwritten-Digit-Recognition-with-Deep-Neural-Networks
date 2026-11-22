"""Handwritten Digit Recognition using CNN, MNIST and Gradio.

Install: pip install -r requirements.txt
Run:     python handwritten_digit_recognition.py
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image

MODEL_PATH = "mnist_digit_cnn.keras"


def build_model():
    model = keras.Sequential([
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def get_model():
    if os.path.exists(MODEL_PATH):
        print("Loading saved model:", MODEL_PATH)
        return keras.models.load_model(MODEL_PATH)

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32")[..., None] / 255.0
    x_test = x_test.astype("float32")[..., None] / 255.0

    model = build_model()
    model.summary()
    model.fit(x_train, y_train, validation_split=0.1,
              epochs=10, batch_size=128,
              callbacks=[keras.callbacks.EarlyStopping(
                  monitor="val_loss", patience=2,
                  restore_best_weights=True)])
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test loss: {loss:.4f} | Test accuracy: {accuracy:.4f}")
    model.save(MODEL_PATH)
    return model


def to_pil(image_input):
    if image_input is None:
        raise ValueError("Draw a digit first.")
    if isinstance(image_input, dict):
        image_input = image_input.get("composite")
        if image_input is None:
            image_input = image_input.get("image")
    if image_input is None:
        raise ValueError("No drawing received.")
    if isinstance(image_input, Image.Image):
        return image_input.convert("RGBA")
    arr = np.asarray(image_input)
    if arr.size == 0:
        raise ValueError("The drawing is empty.")
    if np.issubdtype(arr.dtype, np.floating):
        if arr.max() <= 1:
            arr = arr * 255
    arr = np.clip(arr, 0, 255).astype("uint8")
    return Image.fromarray(arr).convert("RGBA")


def preprocess_drawing(image_input):
    """Prepare a drawing as a centered 28x28 MNIST-style image."""
    image = to_pil(image_input)
    white = Image.new("RGBA", image.size, (255, 255, 255, 255))
    gray = Image.alpha_composite(white, image).convert("L")
    arr = np.asarray(gray, dtype=np.uint8)

    # Convert dark-on-light drawings to MNIST's light-on-dark format.
    if arr.mean() < 127:
        arr = 255 - arr

    mask = arr > 35
    if not mask.any():
        raise ValueError("No digit detected. Draw a thicker digit.")
    ys, xs = np.where(mask)
    crop = arr[ys.min():ys.max()+1, xs.min():xs.max()+1]

    h, w = crop.shape
    scale = 20.0 / max(h, w)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = Image.fromarray(crop).resize((nw, nh), Image.Resampling.LANCZOS)

    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(resized, ((28-nw)//2, (28-nh)//2))
    return (np.asarray(canvas, dtype="float32") / 255.0).reshape(1, 28, 28, 1)


def main():
    model = get_model()
    import gradio as gr

    def predict(image):
        x = preprocess_drawing(image)
        probs = model.predict(x, verbose=0)[0]
        return {str(i): float(probs[i]) for i in range(10)}

    app = gr.Interface(
        fn=predict,
        inputs=gr.Sketchpad(),
        outputs=gr.Label(num_top_classes=5, label="Predictions"),
        title="Handwritten Digit Recognition",
        description="Draw one digit from 0 to 9.",
    )
    app.launch()


if __name__ == "__main__":
    main()
