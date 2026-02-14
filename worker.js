// worker.js
const { parentPort } = require("worker_threads");
const { Jimp } = require("jimp");

// Listen for the 'task' message from the main thread
parentPort.on("message", async (task) => {
  try {
    // 1. Read the image from disk
    const image = await Jimp.read(task.filePath);

    // 2. Resize to 224x224 (MobileNet standard)
    // Note: Using the object syntax for Jimp v1.0+
    image.cover({ w: 224, h: 224 });

    // 3. Send raw data back to Main Thread
    // We send the buffer because Tensors cannot be shared between threads.
    parentPort.postMessage({
      status: "success",
      buffer: image.bitmap.data, // This is a Uint8Array (RGBA)
      labelIndex: task.labelIndex,
      filePath: task.filePath,
    });
  } catch (err) {
    // If the image is corrupt or unreadable, report the error
    parentPort.postMessage({
      status: "error",
      error: err.message,
      filePath: task.filePath,
    });
  }
});
