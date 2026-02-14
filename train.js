process.env.NODE_TLS_REJECT_UNAUTHORIZED = "0"; // Bypass SSL issues
const { Worker } = require("worker_threads");
const os = require("os");
const path = require("path");
const fs = require("fs");
const tf = require("@tensorflow/tfjs");

// --- CONFIGURATION ---
const DATASET_PATH = path.join(__dirname, "dataset");
const MODEL_SAVE_PATH = path.join(
  __dirname,
  "models",
  "dog-classification-model",
);
const BATCH_SIZE = 64; // Batch size for faster processing
const EPOCHS = 20;

// --- HELPER: Save Handler ---
async function saveModelToDisk(model, dirPath) {
  if (!fs.existsSync(dirPath)) fs.mkdirSync(dirPath, { recursive: true });

  await model.save(
    tf.io.withSaveHandler(async (artifacts) => {
      const weightData = Buffer.from(artifacts.weightData);
      const weightFileName = "weights.bin";
      fs.writeFileSync(path.join(dirPath, weightFileName), weightData);

      const modelJSON = {
        modelTopology: artifacts.modelTopology,
        format: artifacts.format,
        generatedBy: artifacts.generatedBy,
        convertedBy: artifacts.convertedBy,
        weightsManifest: [
          { paths: ["./" + weightFileName], weights: artifacts.weightSpecs },
        ],
      };
      fs.writeFileSync(
        path.join(dirPath, "model.json"),
        JSON.stringify(modelJSON, null, 2),
      );

      return {
        modelArtifactsInfo: {
          dateSaved: new Date(),
          modelTopologyType: "JSON",
          weightDataBytes: weightData.byteLength,
        },
      };
    }),
  );
}

async function train() {
  console.log("--- Starting Optimized Multi-Threaded Training (Online) ---");

  // 1. Scan Dataset
  if (!fs.existsSync(DATASET_PATH)) {
    console.error(`Error: Dataset folder not found at ${DATASET_PATH}`);
    return;
  }

  const allDirs = fs
    .readdirSync(DATASET_PATH)
    .filter((f) => fs.statSync(path.join(DATASET_PATH, f)).isDirectory());
  const activeBreeds = allDirs;
  const classNames = activeBreeds.map((dir) => dir.split("-")[1] || dir);
  const numClasses = classNames.length;

  console.log(`Classes Detected (${numClasses}): ${classNames.join(", ")}`);

  // 2. Create Task List
  let tasks = [];
  activeBreeds.forEach((breed, index) => {
    const breedDir = path.join(DATASET_PATH, breed);
    const files = fs
      .readdirSync(breedDir)
      .filter((f) => f.match(/\.(jpg|jpeg|png)$/i));
    files.forEach((file) => {
      tasks.push({ filePath: path.join(breedDir, file), labelIndex: index });
    });
  });

  console.log(`Found ${tasks.length} images total.`);

  // 3. Load Models (ONLINE)
  console.log("Loading MobileNet Base Model from Google Storage...");

  // Using the original URL
  const mobilenet = await tf.loadLayersModel(
    "https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json",
  );

  const layer = mobilenet.getLayer("conv_pw_13_relu");
  const featureExtractor = tf.model({
    inputs: mobilenet.inputs,
    outputs: layer.output,
  });

  console.log("Building Head Model...");
  const model = tf.sequential({
    layers: [
      tf.layers.flatten({ inputShape: layer.outputShape.slice(1) }),
      tf.layers.dense({ units: 100, activation: "relu" }),
      tf.layers.dense({ units: numClasses, activation: "softmax" }),
    ],
  });

  model.compile({
    optimizer: tf.train.adam(0.0001),
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  // 4. BATCH PROCESSING LOGIC
  const numCPUs = os.cpus().length;
  console.log(`Spawning ${numCPUs} worker threads...`);

  const processedFeatureBatches = [];
  const processedLabelBatches = [];
  let completedCount = 0;

  // Temporary buffers for batching
  let imageBuffer = [];
  let labelBuffer = [];

  // Function to process a full batch
  const flushBatch = () => {
    if (imageBuffer.length === 0) return;

    // FIX: We get the tensors OUT of tf.tidy first
    const batchResult = tf.tidy(() => {
      // Stack images into one big tensor [BatchSize, 224, 224, 3]
      const imgBatchTensor = tf.stack(imageBuffer);

      // Run MobileNet ONCE on the whole batch
      const features = featureExtractor.predict(imgBatchTensor);

      // Stack labels
      const labels = tf.stack(labelBuffer);

      // CRITICAL: Return them! This tells tf.tidy "Don't delete these!"
      return { features, labels };
    });

    // NOW it is safe to push them to the array
    processedFeatureBatches.push(batchResult.features);
    processedLabelBatches.push(batchResult.labels);

    // Clean up the individual input tensors (we don't need them anymore)
    imageBuffer.forEach((t) => t.dispose());
    labelBuffer.forEach((t) => t.dispose());

    // Reset buffers
    imageBuffer = [];
    labelBuffer = [];
  };

  await new Promise((resolve) => {
    let activeWorkers = 0;
    let taskIndex = 0;

    const startWorker = () => {
      if (taskIndex >= tasks.length) return;

      const worker = new Worker("./worker.js");
      activeWorkers++;

      const chunkSize = Math.ceil(tasks.length / numCPUs);
      const myTasks = tasks.slice(taskIndex, taskIndex + chunkSize);
      taskIndex += chunkSize;

      if (myTasks.length === 0) {
        worker.terminate();
        activeWorkers--;
        return;
      }

      myTasks.forEach((t) => worker.postMessage(t));
      let tasksFinished = 0;

      worker.on("message", (msg) => {
        tasksFinished++;

        if (msg.status === "success") {
          // 1. Create Tensor from raw buffer
          const imgTensor = tf.tidy(() => {
            return tf
              .tensor3d(new Uint8Array(msg.buffer), [224, 224, 4])
              .slice([0, 0, 0], [-1, -1, 3])
              .div(127.5)
              .sub(1);
          });

          // 2. Add to buffer
          imageBuffer.push(imgTensor);
          labelBuffer.push(tf.oneHot(msg.labelIndex, numClasses));

          completedCount++;
          process.stdout.write(
            `\rProgress: ${completedCount}/${tasks.length} images`,
          );

          // 3. If buffer is full, FLUSH IT
          if (imageBuffer.length >= BATCH_SIZE) {
            flushBatch();
          }
        }

        if (tasksFinished >= myTasks.length) {
          worker.terminate();
          activeWorkers--;

          if (activeWorkers === 0) {
            flushBatch(); // Process remaining
            console.log("\nAll workers finished.");
            resolve();
          }
        }
      });

      worker.on("error", (err) => console.error("Worker Error:", err));
    };

    for (let i = 0; i < numCPUs; i++) startWorker();
  });

  // 5. Training Phase
  console.log("Merging batches for training...");

  const trainX = tf.concat(processedFeatureBatches);
  const trainY = tf.concat(processedLabelBatches);

  // Clean up batches
  processedFeatureBatches.forEach((t) => t.dispose());
  processedLabelBatches.forEach((t) => t.dispose());

  console.log("Starting Model Training...");
  await model.fit(trainX, trainY, {
    epochs: EPOCHS,
    batchSize: BATCH_SIZE,
    shuffle: true,
    callbacks: {
      onEpochEnd: (epoch, logs) =>
        console.log(
          `Epoch ${epoch + 1}: loss=${logs.loss.toFixed(4)} accuracy=${logs.acc.toFixed(4)}`,
        ),
    },
  });

  // 6. Saving
  console.log("Saving Model...");
  await saveModelToDisk(model, MODEL_SAVE_PATH);
  fs.writeFileSync(
    path.join(MODEL_SAVE_PATH, "labels.json"),
    JSON.stringify(classNames),
  );

  console.log(`Success! Model saved to: ${MODEL_SAVE_PATH}`);
}

train();
