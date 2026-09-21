/**
 * perception-cnn/cnn.js
 * ---------------------------------------------------------------
 * Standalone perception CNN for the disaster-response-game.
 *
 * Job: look at a small "sensor tile" (an NxN patch of the world grid
 * centered on a drone/vehicle) and answer two yes/no questions:
 *   1. Is there a victim in this tile?
 *   2. Is the road/terrain in this tile blocked (flood/debris/collapse)?
 *
 * Pure @tensorflow/tfjs (CPU backend, no native bindings) so it runs
 * anywhere plain `node` runs — no tfjs-node install headaches.
 *
 * Usage:
 *   node cnn.js train      # generate synthetic data, train, save weights
 *   node cnn.js eval        # load saved weights, report accuracy
 *   node cnn.js demo        # run one prediction on a random tile
 *
 * Swap-in note: generateDataset() below is a synthetic stand-in.
 * For production, replace it with tiles sampled from real rollouts of
 * simulation-engine/env/world.js (physicsFor + buildTerrain) so the
 * CNN learns the actual flood/cyclone/earthquake physics instead of
 * this file's approximation of it.
 * ---------------------------------------------------------------
 */

import * as tf from "@tensorflow/tfjs";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const TILE_SIZE = 8; // 8x8 grid cells per sensor tile
const CHANNELS = 3; // [hazard intensity, terrain passability, motion signature]
const WEIGHTS_PATH = path.join(__dirname, "cnn.weights.json");

// ---------------------------------------------------------------
// Model
// ---------------------------------------------------------------
function buildModel() {
  const model = tf.sequential();
  model.add(
    tf.layers.conv2d({
      inputShape: [TILE_SIZE, TILE_SIZE, CHANNELS],
      filters: 8,
      kernelSize: 3,
      padding: "same",
      activation: "relu",
    })
  );
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
  model.add(tf.layers.conv2d({ filters: 16, kernelSize: 3, padding: "same", activation: "relu" }));
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({ units: 32, activation: "relu" }));
  model.add(tf.layers.dropout({ rate: 0.2 }));
  // Two independent sigmoid outputs: [victimProb, blockedProb]
  model.add(tf.layers.dense({ units: 2, activation: "sigmoid" }));

  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: "binaryCrossentropy",
    metrics: ["accuracy"],
  });
  return model;
}

// ---------------------------------------------------------------
// Synthetic dataset (stand-in for real physics-engine rollouts)
// ---------------------------------------------------------------
function generateTile(rng = Math.random) {
  const tile = Array.from({ length: TILE_SIZE }, () =>
    Array.from({ length: TILE_SIZE }, () => [0, 1, 0]) // [hazard, passable, motion]
  );

  // Base hazard field: a soft blob of intensity somewhere in the tile
  const hazardCenterX = rng() * TILE_SIZE;
  const hazardCenterY = rng() * TILE_SIZE;
  const hazardRadius = 1.5 + rng() * 3.5;
  const hazardStrength = rng();

  let hazardSum = 0;
  for (let y = 0; y < TILE_SIZE; y++) {
    for (let x = 0; x < TILE_SIZE; x++) {
      const d = Math.hypot(x - hazardCenterX, y - hazardCenterY);
      const intensity = hazardStrength * Math.max(0, 1 - d / hazardRadius);
      const passable = intensity > 0.55 ? 0 : 1;
      tile[y][x][0] = intensity + rng() * 0.05; // small sensor noise
      tile[y][x][1] = passable;
    }
  }

  // Victim: a small motion signature, more likely to be planted away
  // from the worst hazard (people cluster on drier/safer ground)
  const hasVictim = rng() < 0.45;
  let victimLabel = 0;
  if (hasVictim) {
    let vx, vy, tries = 0;
    do {
      vx = Math.floor(rng() * TILE_SIZE);
      vy = Math.floor(rng() * TILE_SIZE);
      tries++;
    } while (tile[vy][vx][0] > 0.6 && tries < 10);
    tile[vy][vx][2] = 0.8 + rng() * 0.2;
    victimLabel = 1;
  }
  // small chance of sensor noise causing a stray motion blip with no victim
  if (!hasVictim && rng() < 0.06) {
    const nx = Math.floor(rng() * TILE_SIZE);
    const ny = Math.floor(rng() * TILE_SIZE);
    tile[ny][nx][2] = 0.3 + rng() * 0.2; // weak signal, shouldn't flip label
  }

  for (let y = 0; y < TILE_SIZE; y++)
    for (let x = 0; x < TILE_SIZE; x++) hazardSum += tile[y][x][0];
  const avgHazard = hazardSum / (TILE_SIZE * TILE_SIZE);
  const blockedLabel = avgHazard > 0.28 ? 1 : 0;

  return { tile, labels: [victimLabel, blockedLabel] };
}

function generateDataset(n) {
  const tiles = [];
  const labels = [];
  for (let i = 0; i < n; i++) {
    const { tile, labels: lbl } = generateTile();
    tiles.push(tile);
    labels.push(lbl);
  }
  return {
    xs: tf.tensor4d(tiles.flat(3), [n, TILE_SIZE, TILE_SIZE, CHANNELS]),
    ys: tf.tensor2d(labels.flat(), [n, 2]),
  };
}

// ---------------------------------------------------------------
// Weight persistence as plain JSON (avoids needing tfjs-node's
// file:// I/O handler just to save/load a small model)
// ---------------------------------------------------------------
async function saveWeights(model, filePath) {
  const weights = model.getWeights().map((w) => ({
    shape: w.shape,
    data: Array.from(w.dataSync()),
  }));
  fs.writeFileSync(filePath, JSON.stringify(weights));
}

async function loadWeights(model, filePath) {
  const saved = JSON.parse(fs.readFileSync(filePath, "utf8"));
  const tensors = saved.map((w) => tf.tensor(w.data, w.shape));
  model.setWeights(tensors);
}

// ---------------------------------------------------------------
// Train / evaluate / predict
// ---------------------------------------------------------------
async function train({ epochs = 12, trainSize = 2000, valSize = 400 } = {}) {
  const model = buildModel();
  const trainData = generateDataset(trainSize);
  const valData = generateDataset(valSize);

  console.log(`Training perception CNN on ${trainSize} synthetic tiles...`);
  await model.fit(trainData.xs, trainData.ys, {
    epochs,
    batchSize: 32,
    validationData: [valData.xs, valData.ys],
    verbose: 0,
    callbacks: {
      onEpochEnd: (epoch, logs) =>
        console.log(
          `  epoch ${epoch + 1}/${epochs} — loss ${logs.loss.toFixed(4)}, ` +
            `acc ${logs.acc.toFixed(3)}, val_acc ${logs.val_acc.toFixed(3)}`
        ),
    },
  });

  await saveWeights(model, WEIGHTS_PATH);
  console.log(`Saved weights to ${WEIGHTS_PATH}`);

  tf.dispose([trainData.xs, trainData.ys, valData.xs, valData.ys]);
  return model;
}

async function evaluate({ testSize = 1000 } = {}) {
  const model = buildModel();
  await loadWeights(model, WEIGHTS_PATH);

  const { xs, ys } = generateDataset(testSize);
  const preds = model.predict(xs);
  const predLabels = preds.round();
  const correct = predLabels.equal(ys).mean().dataSync()[0];
  console.log(`Per-label accuracy on ${testSize} fresh synthetic tiles: ${(correct * 100).toFixed(1)}%`);

  tf.dispose([xs, ys, preds, predLabels]);
  return correct;
}

/** Classify one tile: tile is a TILE_SIZE x TILE_SIZE x 3 nested array. */
async function predict(model, tile) {
  const input = tf.tensor4d([tile], [1, TILE_SIZE, TILE_SIZE, CHANNELS]);
  const output = model.predict(input);
  const [victimProb, blockedProb] = await output.data();
  tf.dispose([input, output]);
  return { victimProb, blockedProb };
}

// ---------------------------------------------------------------
// CLI
// ---------------------------------------------------------------
async function main() {
  const cmd = process.argv[2] || "train";
  if (cmd === "train") {
    await train();
    await evaluate();
  } else if (cmd === "eval") {
    await evaluate();
  } else if (cmd === "demo") {
    const model = buildModel();
    await loadWeights(model, WEIGHTS_PATH);
    const { tile, labels } = generateTile();
    const result = await predict(model, tile);
    console.log("Ground truth: victim =", labels[0], " blocked =", labels[1]);
    console.log(
      "Predicted:    victim =",
      result.victimProb.toFixed(3),
      " blocked =",
      result.blockedProb.toFixed(3)
    );
  } else {
    console.log("Usage: node cnn.js [train|eval|demo]");
  }
}

export { buildModel, generateDataset, generateTile, saveWeights, loadWeights, predict, TILE_SIZE, CHANNELS };

if (process.argv[1] === __filename) {
  main();
}