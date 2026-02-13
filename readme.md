# 🐶 Dog Breed Classifier (TensorFlow.js Node)

A custom Deep Learning model that distinguishes between dog breeds using **Transfer Learning**.
Built with **Node.js** and **TensorFlow.js (Universal/CPU)**, bypassing the need for Python or C++ native bindings.

## 🚀 Features

- **Dataset Used:** https://www.kaggle.com/datasets/yaswanthgali/dog-images
- **Transfer Learning:** Uses MobileNet (v1 0.25) as a feature extractor.
- **Custom Head:** A trained dense layer classifier for specific breeds.
- **Universal Backend:** Runs on pure JavaScript/WASM (no `gyp` or C++ compiler needed).
- **API:** Browser endpoint easy to use with query params.

## 🛠️ Installation

1. **Clone the repository**

   ```bash
   git clone [https://github.com/pranavtelangade/dog-breed-classifier.git] (https://github.com/pranavtelangade/dog-breed-classifier.git)
   cd dog-breed-classifier
   ```

2. **Install Dependencies**

```bash
npm install
Note: This uses @tensorflow/tfjs (CPU version) instead of tfjs-node to ensure compatibility with all Windows environments.
```

## 🏃‍♂️ Usage

**Start the Inference Server**

```bash
node server.js
The server will start on http://localhost:3000 and load the model into memory.
```

**Make a Prediction**
Send a GET request to /predict with an image file.

Using cURL:

```bash
http://localhost:3000/predict?path=path\to\image
```

## 🧠 Training (Optional)

If you want to retrain the model on your own dataset:

Place your images in a dataset/ folder, organized by breed name.

Run the training script:

```Bash
node train.js
Note: Training runs on CPU and may take time for large datasets.
```

## 📂 Project Structure

server.js - The Inference API.
train.js - The Transfer Learning training script.
model/ - Contains the saved TensorFlow.js model artifacts (model.json, weights.bin).
