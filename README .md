<div align="center">

# ✍️ Handwritten Digit Recognition using Deep Neural Networks

**Recognizing handwritten digits (0–9) with a Convolutional Neural Network (CNN)**

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-FF6F00?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-Neural%20Networks-D00000?logo=keras&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-MNIST-blue)
![Interface](https://img.shields.io/badge/UI-Gradio-orange)

</div>

---

## 📌 Project Overview

This project uses deep learning to recognize handwritten digits from **0 to 9**. A **Convolutional Neural Network (CNN)** is trained using the **MNIST dataset**, which contains grayscale images of handwritten digits.

The model processes an input image, identifies visual patterns, and predicts the digit. An interactive **Gradio interface** allows a user to draw a digit and receive a prediction.

## 🎯 Objectives

- Load and explore the MNIST handwritten-digit dataset.
- Preprocess images for neural-network training.
- Build and train a CNN classification model.
- Evaluate the model on test data.
- Provide an interactive interface for handwritten digit predictions.

## 🧰 Technologies Used

- **Python**
- **TensorFlow**
- **Keras**
- **NumPy**
- **Matplotlib**
- **Gradio**
- **MNIST dataset**

## 🧠 Model Architecture

The CNN uses common image-classification layers:

1. **Convolutional layers** – learn image features such as edges and shapes.
2. **Max-pooling layers** – reduce feature-map dimensions.
3. **Dropout** – helps reduce overfitting during training.
4. **Flatten** – converts feature maps into a one-dimensional vector.
5. **Dense layer** – learns higher-level patterns.
6. **Softmax output layer** – produces class probabilities for digits 0–9.

## 🔄 Workflow

1. Load the MNIST dataset.
2. Normalize image pixel values and prepare labels.
3. Build the CNN model using TensorFlow/Keras.
4. Train the model using the training images.
5. Evaluate classification performance using the test dataset.
6. Use the trained model to predict handwritten digits.
7. Launch the Gradio interface for interactive predictions.

## 🚀 Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/krishsurya/Handwritten-Digit-Recognition-with-Deep-Neural-Networks.git
cd Handwritten-Digit-Recognition-with-Deep-Neural-Networks
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the notebook or Python script

Open the notebook in Jupyter or Google Colab and run the cells in order. If the project is run as a Python script, use the script filename present in the repository.

## 🖼️ Interactive Prediction

The Gradio interface lets users draw a handwritten digit. The input image is preprocessed and passed to the trained model, which returns the predicted digit and its confidence score.

> Prediction quality can vary depending on handwriting style, image centering, stroke thickness, and preprocessing.

## 📊 Model Evaluation

Evaluate the trained model using the test dataset. Add the actual test accuracy, loss, and any other metrics produced by your run here.

| Metric | Result |
|---|---|
| Test accuracy | Add your measured result |
| Test loss | Add your measured result |

## 📁 Suggested Repository Structure

```text
Handwritten-Digit-Recognition-with-Deep-Neural-Networks/
│
├── README.md
├── requirements.txt
├── handwritten_digit_recognition.ipynb
└── mnist.keras                 # Optional: saved trained model
```

The filenames above are suggested. Keep the filenames that actually exist in your repository.

## 👨‍💻 Author

**Krishnasurya**

M.Sc. Business Statistics | Python | Machine Learning | Deep Learning

GitHub: [@krishsurya](https://github.com/krishsurya)

---

<div align="center">

⭐ If you find this project useful, consider starring the repository.

</div>
