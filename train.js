process.env.NODE_TLS_REJECT_UNAUTHORIZED = '0';
const tf = require('@tensorflow/tfjs');
const { Jimp } = require('jimp');
const fs = require('fs');
const path = require('path');
const path = require('path');

const DATASET_PATH = path.join(__dirname, 'dataset');
const IMAGE_SIZE = 224;

async function loadAndProcessImage(filePath) {
  const image = await Jimp.read(filePath);
  image.cover({ w: IMAGE_SIZE, h: IMAGE_SIZE });
  const { data } = image.bitmap;

  return tf.tidy(() => {
    // Convert RGBA to RGB [224, 224, 3]
    const img = tf.tensor3d(new Uint8Array(data), [IMAGE_SIZE, IMAGE_SIZE, 4]);
    return img.slice([0, 0, 0], [-1, -1, 3]).div(127.5).sub(1); // Normalize to [-1, 1]
  });
}

async function saveModelToDisk(model, dirPath) {
  const fs = require('fs');
  const path = require('path');

  // Ensure the directory exists
  if (!fs.existsSync(dirPath)) {
    fs.mkdirSync(dirPath, { recursive: true });
  }

  await model.save(
    tf.io.withSaveHandler(async (artifacts) => {
      // 1. Save the weights (the binary data)
      const weightData = Buffer.from(artifacts.weightData);
      const weightFileName = 'weights.bin';
      fs.writeFileSync(path.join(dirPath, weightFileName), weightData);

      // 2. Save the topology (the JSON structure)
      const modelJSON = {
        modelTopology: artifacts.modelTopology,
        format: artifacts.format,
        generatedBy: artifacts.generatedBy,
        convertedBy: artifacts.convertedBy,
        weightsManifest: [
          {
            paths: ['./' + weightFileName],
            weights: artifacts.weightSpecs,
          },
        ],
      };
      fs.writeFileSync(
        path.join(dirPath, 'model.json'),
        JSON.stringify(modelJSON, null, 2),
      );

      return {
        modelArtifactsInfo: {
          dateSaved: new Date(),
          modelTopologyType: 'JSON',
          generatedBy: 'TensorFlow.js',
          convertedBy: null,
          weightDataBytes: weightData.byteLength,
        },
      };
    }),
  );
}

async function train() {
  // 1. Filter out hidden system files (like .DS_Store or Thumbs.db)
  const allDirs = fs.readdirSync(DATASET_PATH).filter((f) => {
    return fs.statSync(path.join(DATASET_PATH, f)).isDirectory();
  });

  // SAFETY LIMIT: Only take the first 3 folders for the first test run!
  // Change this to 'allDirs' once you confirm it works fast enough.
  const activeBreeds = allDirs.slice(0, 3);

  // Clean up the names: "n02085620-Chihuahua" -> "Chihuahua"
  const classNames = activeBreeds.map((dir) => dir.split('-')[1] || dir);
  const numClasses = classNames.length;

  console.log(`Training on ${numClasses} breeds: ${classNames.join(', ')}`);

  // ... Load MobileNet (same as before) ...
  const mobilenet = await tf.loadLayersModel(
    'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json',
  );
  const layer = mobilenet.getLayer('conv_pw_13_relu');
  const featureExtractor = tf.model({
    inputs: mobilenet.inputs,
    outputs: layer.output,
  });

  // ... Build Head Model (same as before) ...
  const model = tf.sequential({
    layers: [
      tf.layers.flatten({ inputShape: layer.outputShape.slice(1) }),
      tf.layers.dense({ units: 100, activation: 'relu' }),
      tf.layers.dense({ units: numClasses, activation: 'softmax' }),
    ],
  });

  model.compile({
    optimizer: tf.train.adam(0.0001),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  let xs = [];
  let ys = [];

  // Loop through the selected breeds
  for (let i = 0; i < activeBreeds.length; i++) {
    const breedDir = path.join(DATASET_PATH, activeBreeds[i]);
    const files = fs.readdirSync(breedDir);

    console.log(`Processing ${classNames[i]} (${files.length} images)...`);

    for (const file of files) {
      // Skip non-image files just in case
      if (!file.match(/\.(jpg|jpeg|png)$/i)) continue;

      try {
        const imgTensor = await loadAndProcessImage(path.join(breedDir, file));
        const features = featureExtractor.predict(imgTensor.expandDims(0));

        xs.push(features);
        ys.push(tf.oneHot(i, numClasses));

        imgTensor.dispose(); // Important for memory!
      } catch (err) {
        console.log(`Skipped bad image: ${file}`);
      }
    }
  }

  // ... The rest (concatenation, fit, save) remains the same ...
  const trainX = tf.concat(xs);
  const trainY = tf.stack(ys);

  console.log('Starting training...');
  await model.fit(trainX, trainY, {
    epochs: 20,
    callbacks: {
      onEpochEnd: (epoch, logs) =>
        console.log(`Epoch ${epoch}: loss = ${logs.loss.toFixed(4)}`),
    },
  });

  await saveModelToDisk(
    model,
    path.join(__dirname, 'models', 'dog-classification-model'),
  );

  fs.writeFileSync(
    path.join(__dirname, 'models', 'dog-classification-model', 'labels.json'),
    JSON.stringify(classNames),
  );
  console.log('Model saved successfully using custom handler!');
}

train();
