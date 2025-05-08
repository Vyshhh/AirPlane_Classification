/*import React, { useState } from 'react';
import '../style.css'; // Or import './test_button.css' if using separate CSS

const TestButton = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [preview, setPreview] = useState(null);

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedImage(file);
      setPreview(URL.createObjectURL(file));
    }
  };

  const handlePredict = () => {
    if (!selectedImage) {
      alert('Please upload an image first!');
      return;
    }

    // TODO: Replace this with actual model call
    alert('Prediction functionality not yet connected.');
  };

  return (
    <section id="test" className="test-section">
      <h2>Test the Classifier</h2>
      <input
        type="file"
        accept="image/*"
        onChange={handleImageChange}
        className="upload-input"
      />
      {preview && (
        <div className="preview-container">
          <img src={preview} alt="Preview" className="preview-image" />
        </div>
      )}
      <button onClick={handlePredict} className="predict-button">
        Predict
      </button>
    </section>
  );
};

export default TestButton;*/

/*
import React, { useState } from 'react';
import '../style.css'; // or './test_button.css' if separated
import axios from 'axios';

const TestButton = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState('');
  const [loading, setLoading] = useState(false);

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedImage(file);
      setPreview(URL.createObjectURL(file));
      setResult('');
    }
  };

  const handlePredict = async () => {
    if (!selectedImage) {
      alert('Please upload an image first!');
      return;
    }

    setLoading(true);
    const formData = new FormData();
    formData.append('image', selectedImage);

    try {
      const response = await axios.post('http://localhost:3000/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });

      setResult(response.data.data.result); // the prediction text
    } catch (error) {
      console.error('Prediction failed:', error);
      setResult('Prediction failed. Try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <section id="test" className="test-section">
      <h2>Test the Classifier</h2>
      <input
        type="file"
        accept="image/*"
        onChange={handleImageChange}
        className="upload-input"
      />
      {preview && (
        <div className="preview-container">
          <img src={preview} alt="Preview" className="preview-image" />
        </div>
      )}
      <button onClick={handlePredict} className="predict-button" disabled={loading}>
        {loading ? 'Predicting...' : 'Predict'}
      </button>
      {result && (
        <div className="prediction-result">
          <h4>Prediction Result:</h4>
          <p>{result}</p>
        </div>
      )}
    </section>
  );
};

export default TestButton;*/