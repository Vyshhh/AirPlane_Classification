/*const express = require('express');
const multer = require('multer');
const { spawn } = require('child_process');
const mongoose = require('mongoose');
const path = require('path');
const cors = require('cors');



// MongoDB Setup
mongoose.connect('mongodb://localhost:27017/yolo_predictions', {
  useNewUrlParser: true,
  useUnifiedTopology: true
});

const predictionSchema = new mongoose.Schema({
  imagePath: String,
  result: String,
  timestamp: { type: Date, default: Date.now }
});

const Prediction = mongoose.model('Prediction', predictionSchema);

// Express App Setup
const app = express();
const PORT = 3000;

app.use(cors());
// Multer Storage Setup
const storage = multer.diskStorage({
  destination: './uploads/',
  filename: function (req, file, cb) {
    cb(null, 'uploaded_' + Date.now() + path.extname(file.originalname));
  }
});
const upload = multer({ storage });

// Route to Upload and Predict
app.post('/upload', upload.single('image'), (req, res) => {
  const imgPath = req.file.path;

  const python = spawn('python', ['pred.py', imgPath]);

  let prediction = '';
  python.stdout.on('data', (data) => {
    prediction += data.toString();
  });

  python.stderr.on('data', (data) => {
    console.error(`stderr: ${data}`);
  });

  python.on('close', async (code) => {
    try {
      const newPrediction = new Prediction({
        imagePath: imgPath,
        result: prediction.trim()
      });
      await newPrediction.save();
      res.send({
        message: "Prediction saved",
        data: newPrediction
      });
    } catch (err) {
      res.status(500).send({ error: "Database save failed", details: err });
    }
  });
});

// Start Server
app.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
});

*/

const express = require('express');
const multer = require('multer');
const { spawn } = require('child_process');
const mongoose = require('mongoose');
const path = require('path');
const cors = require('cors');

// MongoDB Setup
mongoose.connect('mongodb://localhost:27017/yolo_predictions', {
  useNewUrlParser: true,
  useUnifiedTopology: true
});

const predictionSchema = new mongoose.Schema({
  imagePath: String,
  result: String,
  timestamp: { type: Date, default: Date.now }
});

const Prediction = mongoose.model('Prediction', predictionSchema);

// Express App Setup
const app = express();
const PORT = 5000;

app.use(cors());
app.use(express.json());

// Multer Storage Setup
const storage = multer.diskStorage({
  destination: './uploads/',
  filename: function (req, file, cb) {
    cb(null, 'uploaded_' + Date.now() + path.extname(file.originalname));
  }
});
const upload = multer({ storage });

// Route to Upload and Predict
app.post('/upload', upload.single('image'), (req, res) => {
  const imgPath = req.file.path;

  const python = spawn('python', ['pred.py', imgPath]);

  let prediction = '';
  python.stdout.on('data', (data) => {
    prediction += data.toString();
  });

  python.stderr.on('data', (data) => {
    console.error(`stderr: ${data}`);
  });

  python.on('close', async (code) => {
    if (code !== 0) {
      return res.status(500).send({ error: `Python script exited with code ${code}` });
    }

    try {
      // Clean up "Detections: " prefix if present
      const cleanedPrediction = prediction.replace(/^Detections:\s*/i, '').trim();

      const newPrediction = new Prediction({
        imagePath: imgPath,
        result: cleanedPrediction
      });

      await newPrediction.save();

      res.send({
        message: "Prediction saved",
        data: newPrediction
      });
    } catch (err) {
      res.status(500).send({ error: "Database save failed", details: err });
    }
  });
});

// Start Server
app.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
});
