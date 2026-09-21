/**
 * hazard-lstm/lstm.js
 * ---------------------------------------------------------------
 * Standalone hazard-forecasting LSTM for the disaster-response-game.
 *
 * Job: given a cell's last SEQ_LEN ticks of hazard intensity, predict
 * its next HORIZON ticks — this is what feeds the "Hazard Forecast /
 * LSTM projection, next 4 steps" chart in the frontend.
 *
 * Pure @tensorflow/tfjs (CPU backend, no native bindings).
 *
 * Usage:
 *   node lstm.js train     # generate synthetic rollouts, train, save weights
 *   node lstm.js eval       # load saved weights, compare MAE vs naive baseline
 *   node lstm.js demo       # forecast one random sequence
 *
 * Swap-in note: generateDataset() below synthesizes hazard curves with
 * logistic growth + decay + noise as a stand-in for real dynamics. For
 * production, replace it with per-cell hazard histories rolled out from
 * simulation-engine/env/world.js's actual flood/cyclone/earthquake
 * physics, sliced into (history window -> future window) pairs.
 * ---------------------------------------------------------------
 */

import * as tf from "@tensorflow/tfjs";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const SEQ_LEN = 6; // ticks of history fed in
const HORIZON = 4; // ticks forecast out (matches the frontend's "next 4 steps")
const WEIGHTS_PATH = path.join(__dirname, "lstm.weights.json");

// ---------------------------------------------------------------
// Model
// ---------------------------------------------------------------
function buildModel() {
  const model = tf.sequential();
  model.add(tf.layers.lstm({ inputShape: [SEQ_LEN, 1], units: 16, returnSequences: false }));
  model.add(tf.layers.dense({ units: 16, activation: "relu" }));
  model.add(tf.layers.dense({ units: HORIZON, activation: "linear" }));

  model.compile({
    optimizer: tf.train.adam(0.005),
    loss: "meanSquaredError",
    metrics: ["mae"],
  });
  return model;
}

// ---------------------------------------------------------------
// Synthetic dataset (stand-in for real physics-engine rollouts)
// ---------------------------------------------------------------
/** One synthetic hazard-intensity curve over `length` ticks, in [0,1]. */
function generateCurve(length, rng = Math.random) {
  const peak = 0.4 + rng() * 0.6;
  const growthRate = 0.15 + rng() * 0.35;
  const onsetTick = rng() * length * 0.3;
  const decayStart = length * (0.5 + rng() * 0.3);
  const decayRate = 0.05 + rng() * 0.15;
  const noise = () => (rng() - 0.5) * 0.03;

  const curve = [];
  for (let t = 0; t < length; t++) {
    let v;
    if (t < decayStart) {
      // logistic growth from onset
      v = peak / (1 + Math.exp(-growthRate * (t - onsetTick)));
    } else {
      const atDecayStart = peak / (1 + Math.exp(-growthRate * (decayStart - onsetTick)));
      v = atDecayStart * Math.exp(-decayRate * (t - decayStart));
    }
    curve.push(Math.max(0, Math.min(1, v + noise())));
  }
  return curve;
}

function generateDataset(numCurves, ticksPerCurve = 30) {
  const inputs = [];
  const targets = [];
  for (let i = 0; i < numCurves; i++) {
    const curve = generateCurve(ticksPerCurve);
    // slide a window across the curve to get many (history -> future) pairs
    for (let start = 0; start + SEQ_LEN + HORIZON <= ticksPerCurve; start++) {
      inputs.push(curve.slice(start, start + SEQ_LEN).map((v) => [v]));
      targets.push(curve.slice(start + SEQ_LEN, start + SEQ_LEN + HORIZON));
    }
  }
  return {
    xs: tf.tensor3d(inputs),
    ys: tf.tensor2d(targets),
  };
}

// ---------------------------------------------------------------
// Weight persistence as plain JSON
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
// Naive baseline: "tomorrow = today" repeated HORIZON times.
// The LSTM only earns its keep if it beats this.
// ---------------------------------------------------------------
function naiveBaselineMAE(xs, ys) {
  const xsData = xs.arraySync(); // [n, SEQ_LEN, 1]
  const ysData = ys.arraySync(); // [n, HORIZON]
  let total = 0;
  let count = 0;
  for (let i = 0; i < xsData.length; i++) {
    const lastValue = xsData[i][SEQ_LEN - 1][0];
    for (let h = 0; h < HORIZON; h++) {
      total += Math.abs(lastValue - ysData[i][h]);
      count++;
    }
  }
  return total / count;
}

// ---------------------------------------------------------------
// Train / evaluate / forecast
// ---------------------------------------------------------------
async function train({ epochs = 25, trainCurves = 250, valCurves = 50 } = {}) {
  const model = buildModel();
  const trainData = generateDataset(trainCurves);
  const valData = generateDataset(valCurves);

  console.log(`Training hazard LSTM on ${trainData.xs.shape[0]} synthetic windows...`);
  await model.fit(trainData.xs, trainData.ys, {
    epochs,
    batchSize: 32,
    validationData: [valData.xs, valData.ys],
    verbose: 0,
    callbacks: {
      onEpochEnd: (epoch, logs) =>
        console.log(
          `  epoch ${epoch + 1}/${epochs} — loss ${logs.loss.toFixed(5)}, ` +
            `mae ${logs.mae.toFixed(4)}, val_mae ${logs.val_mae.toFixed(4)}`
        ),
    },
  });

  await saveWeights(model, WEIGHTS_PATH);
  console.log(`Saved weights to ${WEIGHTS_PATH}`);

  tf.dispose([trainData.xs, trainData.ys, valData.xs, valData.ys]);
  return model;
}

async function evaluate({ testCurves = 80 } = {}) {
  const model = buildModel();
  await loadWeights(model, WEIGHTS_PATH);

  const { xs, ys } = generateDataset(testCurves);
  const preds = model.predict(xs);
  const modelMAE = tf.losses.absoluteDifference(ys, preds).dataSync()[0];
  const baselineMAE = naiveBaselineMAE(xs, ys);

  console.log(`Model MAE:    ${modelMAE.toFixed(4)}`);
  console.log(`Baseline MAE: ${baselineMAE.toFixed(4)} (naive "hold last value")`);
  console.log(
    modelMAE < baselineMAE
      ? `LSTM beats the naive baseline by ${(((baselineMAE - modelMAE) / baselineMAE) * 100).toFixed(1)}%`
      : `LSTM did not beat the naive baseline — needs more training/tuning`
  );

  tf.dispose([xs, ys, preds]);
  return { modelMAE, baselineMAE };
}

/** Forecast the next HORIZON values given SEQ_LEN past values (each in [0,1]). */
async function forecast(model, history) {
  if (history.length !== SEQ_LEN) {
    throw new Error(`forecast() expects exactly ${SEQ_LEN} historical values, got ${history.length}`);
  }
  const input = tf.tensor3d([history.map((v) => [v])]);
  const output = model.predict(input);
  const values = await output.data();
  tf.dispose([input, output]);
  return Array.from(values);
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
    const curve = generateCurve(SEQ_LEN + HORIZON);
    const history = curve.slice(0, SEQ_LEN);
    const actualFuture = curve.slice(SEQ_LEN);
    const predicted = await forecast(model, history);
    console.log("History:        ", history.map((v) => v.toFixed(3)).join("  "));
    console.log("Actual future:  ", actualFuture.map((v) => v.toFixed(3)).join("  "));
    console.log("Predicted:      ", predicted.map((v) => v.toFixed(3)).join("  "));
  } else {
    console.log("Usage: node lstm.js [train|eval|demo]");
  }
}

export {
  buildModel,
  generateDataset,
  generateCurve,
  saveWeights,
  loadWeights,
  forecast,
  naiveBaselineMAE,
  SEQ_LEN,
  HORIZON,
};

if (process.argv[1] === __filename) {
  main();
}