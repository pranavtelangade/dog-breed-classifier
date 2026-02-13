process.env.NODE_TLS_REJECT_UNAUTHORIZED = '0';
const express = require('express');
const tf = require('@tensorflow/tfjs');
const { Jimp } = require('jimp');
const fs = require('fs');
const path = require('path');

const app = express();

// CONFIGURATION
const PORT = 3000;
const MODEL_DIR = path.join(__dirname, 'models', 'dog-classification-model');
const LABELS_PATH = path.join(MODEL_DIR, 'labels.json');

let baseModel; // MobileNet Feature Extractor
let headModel; // Your trained dog classifier
let labels;

// Serve the trained model files
app.use('/model', express.static(MODEL_DIR));

async function loadResources() {
  try {
    console.log('Loading MobileNet Base...');
    // 1. Load the same MobileNet version used in training
    const mobilenet = await tf.loadLayersModel(
      'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json',
    );

    // 2. Re-create the Feature Extractor (up to 'conv_pw_13_relu')
    const layer = mobilenet.getLayer('conv_pw_13_relu');
    baseModel = tf.model({ inputs: mobilenet.inputs, outputs: layer.output });
    console.log('MobileNet Base loaded!');

    console.log('Loading Custom Head Model...');
    // 3. Load your trained "Head" model
    headModel = await tf.loadLayersModel(
      `http://localhost:${PORT}/model/model.json`,
    );
    console.log('Custom Head Model loaded!');

    if (fs.existsSync(LABELS_PATH)) {
      labels = JSON.parse(fs.readFileSync(LABELS_PATH, 'utf-8'));
      console.log('Labels loaded:', labels);
    }
  } catch (err) {
    console.error('Failed to load resources:', err.message);
  }
}

async function processImage(filePath) {
  const image = await Jimp.read(filePath);
  image.cover({ w: 224, h: 224 }); // Ensure 224x224
  const buffer = image.bitmap.data;

  return tf.tidy(() => {
    const tensor = tf.tensor3d(new Uint8Array(buffer), [224, 224, 4]);
    return tensor
      .slice([0, 0, 0], [-1, -1, 3]) // Remove Alpha
      .toFloat()
      .div(127.5)
      .sub(1)
      .expandDims(0); // [1, 224, 224, 3]
  });
}

app.get('/predict', async (req, res) => {
  if (!req.query.path || !headModel) {
    return res.status(503).json({ error: 'Models are still loading...' });
  }

  // Handle both File Upload (Postman) and Query Param (Browser/Test)
  const imagePath = req.file ? req.file.path : req.query.path;
  if (!imagePath) return res.status(400).json({ error: 'Image required.' });

  try {
    // 1. Process Image
    const imageTensor = await processImage(imagePath);

    // 2. Extract Features using MobileNet Base
    // Input: [1, 224, 224, 3] -> Output: [1, 7, 7, 256]
    const features = baseModel.predict(imageTensor);

    // 3. Predict Breed using your Head Model
    // Input: [1, 7, 7, 256] -> Output: Probabilities
    const prediction = headModel.predict(features);

    const pIndex = prediction.argMax(-1).dataSync()[0];
    const pScores = await prediction.data();

    // Clean up Tensors
    imageTensor.dispose();
    features.dispose(); // Important!
    prediction.dispose();

    if (req.file) fs.unlinkSync(imagePath);

    const result = {
      breed: labels ? labels[pIndex] : `Class ${pIndex}`,
      confidence: (pScores[pIndex] * 100).toFixed(2) + '%',
      all_scores: labels
        ? labels.map((l, i) => ({ breed: l, score: pScores[i].toFixed(4) }))
        : pScores,
    };

    res.json(result);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

app.listen(PORT, async () => {
  console.log(`Server running on http://localhost:${PORT}`);
  await loadResources();
});
