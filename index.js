const express = require('express');
const cors = require('cors');
const mongoose = require('mongoose');
const fs = require('fs');
const csv = require('csv-parser');
const { Matrix, inverse } = require('ml-matrix');
const dotenv = require('dotenv');
dotenv.config();
const app = express();
const port = 5000;

app.use(cors());
app.use(express.json());

mongoose.connect(process.env.MONGO_URI, {
  useNewUrlParser: true,
  useUnifiedTopology: true
}).then(() => console.log('✅ Connected to MongoDB'))
  .catch(err => console.error('❌ MongoDB Connection Error:', err));

const predictionSchema = new mongoose.Schema({
  timestamp: { type: Date, default: Date.now },
  waterLevel: Number,
  waterFlow: Number,
  turbineSpeed: Number,
  electricityGenerated: Number,
  batterystorage: Number,
  waterPressure: Number
});

const Prediction = mongoose.model('Prediction', predictionSchema);

const dataset = [];
let modelWeights = null;
let minValues = {}, maxValues = {};

fs.createReadStream('rainwater_data.csv')
  .pipe(csv({ separator: ',', skipEmptyLines: true }))
  .on('data', (row) => {
    const waterLevel = Math.min(parseFloat(row['Water Level (cm)']), 99);
    const waterFlow = parseFloat(row['Water Flow Speed (m/s)']);
    const turbineSpeed = Math.min(parseFloat(row['Turbine Spin (RPM)']), 1499);
    const electricityGenerated = Math.min(parseFloat(row['Electricity Generated (W)']), 9.9);
    const batterystorage = Math.min(parseFloat(row['Battery storage (W)']), 9.9);
    const waterPressure = Math.min(parseFloat(row['Water Pressure (psi)']), 30);

    if ([waterLevel, waterFlow, turbineSpeed, electricityGenerated, batterystorage, waterPressure].every(v => !isNaN(v))) {
      dataset.push({ waterLevel, waterFlow, turbineSpeed, electricityGenerated, batterystorage, waterPressure });
    }
  })
  .on('end', () => {
    console.log('✅ CSV file processed');
    if (dataset.length > 3) trainModel();
    else console.log('❌ Not enough data to train model.');
  });

function normalize(value, min, max) {
  return max === min ? 0 : (value - min) / (max - min);
}

function denormalize(value, min, max) {
  return max === min ? min : value * (max - min) + min;
}

function trainModel() {
  minValues = dataset.reduce((min, d) => ({
    waterLevel: Math.min(min.waterLevel ?? d.waterLevel, d.waterLevel),
    waterFlow: Math.min(min.waterFlow ?? d.waterFlow, d.waterFlow),
    turbineSpeed: Math.min(min.turbineSpeed ?? d.turbineSpeed, d.turbineSpeed),
    electricityGenerated: Math.min(min.electricityGenerated ?? d.electricityGenerated, d.electricityGenerated),
    batterystorage: Math.min(min.batterystorage ?? d.batterystorage, d.batterystorage),
  }), {});

  maxValues = dataset.reduce((max, d) => ({
    waterLevel: Math.max(max.waterLevel ?? d.waterLevel, d.waterLevel),
    waterFlow: Math.max(max.waterFlow ?? d.waterFlow, d.waterFlow),
    turbineSpeed: Math.max(max.turbineSpeed ?? d.turbineSpeed, d.turbineSpeed),
    electricityGenerated: Math.max(max.electricityGenerated ?? d.electricityGenerated, d.electricityGenerated),
    batterystorage: Math.max(max.batterystorage ?? d.batterystorage, d.batterystorage),
  }), {});

  const X = dataset.map(d => [
    1,
    normalize(d.waterLevel, minValues.waterLevel, maxValues.waterLevel),
    normalize(d.waterFlow, minValues.waterFlow, maxValues.waterFlow),
    normalize(d.turbineSpeed, minValues.turbineSpeed, maxValues.turbineSpeed)
  ]);

  const y = dataset.map(d => [
    normalize(d.electricityGenerated, minValues.electricityGenerated, maxValues.electricityGenerated),
    normalize(d.batterystorage, minValues.batterystorage, maxValues.batterystorage)
  ]);

  if (X.length === 0 || y.length === 0) {
    console.log('❌ Dataset is empty or not valid.');
    return;
  }

  const XMatrix = new Matrix(X);
  const yMatrix = new Matrix(y);

  try {
    const lambda = 0.01;
    const I = Matrix.eye(XMatrix.columns, XMatrix.columns).mul(lambda);

    modelWeights = inverse(XMatrix.transpose().mmul(XMatrix).add(I))
      .mmul(XMatrix.transpose())
      .mmul(yMatrix);

    console.log('✅ Model trained successfully');
  } catch (error) {
    console.error('❌ Model training failed:', error);
  }
}

let batteryLevel = 3; // Start with initial battery level at 3W

setInterval(async () => {
  if (!modelWeights) return console.log('⚠️ Model not trained yet');

  const lastData = dataset.length ? dataset[dataset.length - 1] : {
    waterLevel: 50,
    waterFlow: 2,
    turbineSpeed: 1000,
    waterPressure: 20,
    batterystorage: 3
  };

  const newWaterLevel = Math.max(10, Math.min(28, lastData.waterLevel + (Math.random() * 9 - 4)));
  const newWaterFlow = Math.max(0.5, Math.min(5, newWaterLevel * 0.2 + (Math.random() * 0.5 - 0.25)));
  const newTurbineSpeed = Math.min(1495, Math.max(500, newWaterFlow * 150 + (Math.random() * 50 - 25)));

  // Map turbine speed to electricity generation (1W to 10W)
  const minTurbineSpeed = 500; // Minimum turbine speed
  const maxTurbineSpeed = 1495; // Maximum turbine speed
  const minElectricity = 1; // Minimum electricity generated (1W)
  const maxElectricity = 10; // Maximum electricity generated (10W)

  // Scale turbine speed to electricity generated in the range of 1W to 10W
  let newElectricityGenerated = ((newTurbineSpeed - minTurbineSpeed) / (maxTurbineSpeed - minTurbineSpeed)) * (maxElectricity - minElectricity) + minElectricity;

  // Ensure the electricity doesn't exceed 10W
  newElectricityGenerated = Math.min(10, Math.max(1, newElectricityGenerated));

  // Slowly increase the battery level based on electricity generated
  if (batteryLevel < 10) {
    let batteryIncrease = Math.min(newElectricityGenerated * 0.1, 0.1); // Increase at a fixed rate (0.1 per interval)
    batteryLevel = Math.min(10, batteryLevel + batteryIncrease); // Ensure battery doesn't exceed 10W
  }

  const newBatterystorage = batteryLevel;

  // Simulate new water pressure
  const newWaterPressure = Math.min(28, lastData.waterPressure + (Math.random() * 1.0 - 0.5));

  // Create new prediction object and save it to the database
  const newPrediction = new Prediction({
    timestamp: new Date(),
    waterLevel: newWaterLevel,
    waterFlow: newWaterFlow,
    turbineSpeed: newTurbineSpeed,
    electricityGenerated: newElectricityGenerated,
    batterystorage: newBatterystorage,
    waterPressure: newWaterPressure
  });

  dataset.push(newPrediction);
  await newPrediction.save();

  console.log('✅ New Prediction Saved:', newPrediction);
}, 5000);


app.get('/', (req, res) => {
  res.send('Hello, welcome to the Rainwater Harvesting System API! 🚀');
});


app.get('/predict', async (req, res) => {
  try {
    const lastPrediction = await Prediction.findOne().sort({ timestamp: -1 });
    res.json(lastPrediction || { message: 'No predictions available' });
  } catch (error) {
    res.status(500).json({ error: 'Failed to fetch predictions' });
  }
});

app.listen(port, () => console.log(`🚀 Server running on http://localhost:${port}`));
