import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './PredictPage.css';

export default function PredictPage() {
  const [file, setFile] = useState(null);
  const [history, setHistory] = useState([]);
  const [predictionResult, setPredictionResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [preview, setPreview] = useState(null);

  useEffect(() => {
    return () => {
      if (preview) {
        URL.revokeObjectURL(preview);
      }
      // Only revoke object URLs for history on unmount, not on every update
      history.forEach(entry => {
        URL.revokeObjectURL(entry.image);
      });
    };
    // eslint-disable-next-line
  }, []); // Only run on unmount

  const validateFile = (file) => {
    const maxSize = 5 * 1024 * 1024; // 5MB
    const allowedTypes = ['image/jpeg', 'image/png', 'image/jpg'];

    if (!allowedTypes.includes(file.type)) {
      throw new Error('Please upload a valid image file (JPEG, PNG)');
    }

    if (file.size > maxSize) {
      throw new Error('File size should be less than 5MB');
    }
  };

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      try {
        validateFile(selectedFile);
        setFile(selectedFile);
        setError(null);
        // Create preview URL
        const previewUrl = URL.createObjectURL(selectedFile);
        setPreview(previewUrl);
      } catch (err) {
        setError(err.message);
        setFile(null);
        setPreview(null);
      }
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError('Please select a file first');
      return;
    }

    setLoading(true);
    setError(null);
    const formData = new FormData();
    formData.append('image', file);

    try {
      const response = await axios.post('http://localhost:5000/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });

      if (response.data.message === "Prediction saved") {
        // Create a new object URL for this prediction
        const imageUrl = URL.createObjectURL(file);
        const newEntry = {
          image: imageUrl,
          label: response.data.data.result,
          timestamp: new Date().toLocaleString(),
        };
        // Add new entry to the beginning of history array
        setHistory(prevHistory => [newEntry, ...prevHistory]);
        setPredictionResult(response.data.data.result);
      } else {
        throw new Error(response.data.message || 'Prediction failed');
      }
    } catch (err) {
      if (err.code === 'ERR_NETWORK') {
        setError('Cannot connect to the server. Please make sure the backend server is running on port 5000.');
      } else if (err.response) {
        setError(`Server error: ${err.response.data?.message || err.response.statusText}`);
      } else if (err.request) {
        setError('No response from server. Please check if the server is running.');
      } else {
        setError(err.message || 'An unexpected error occurred');
      }
    } finally {
      setLoading(false);
      setFile(null);
      setPreview(null);
    }
  };

  return (
    <div
      className="predict-page-wrapper"
      style={{
        backgroundImage: 'url("/img4.jpg")',
        backgroundSize: 'cover',
        backgroundPosition: 'center',
        backgroundRepeat: 'no-repeat',
        minHeight: '100vh',
        padding: '2rem 0',
      }}
    >
      <div className="white-box">
        <div className="box-header logo-only">
          <div className="logo">✈️ Aerial Vehicle Detector</div>
        </div>

        <div className="box-body">
          <h1>Snap. Upload. Detect</h1>
          <p>Upload an image to classify it as Military, Civilian, UAV or Unknown</p>

          <form onSubmit={handleSubmit} className="upload-form">
            <label htmlFor="file-upload" className="drop-box">
              {preview ? (
                <div className="preview-container">
                  <img src={preview} alt="Preview" className="preview-image" />
                  <p>{file.name}</p>
                </div>
              ) : (
                '📁 Drop an image here or click to upload'
              )}
              <input
                type="file"
                id="file-upload"
                accept="image/jpeg,image/png,image/jpg"
                onChange={handleFileChange}
                hidden
              />
            </label>
            <button
              type="submit"
              className="submit-btn"
              disabled={loading || !file}
            >
              {loading ? '🚀 Predicting...' : '🚀 Predict'}
            </button>
          </form>

          {error && (
            <div className="error-message">
              {error}
            </div>
          )}

          {predictionResult && (
            <div className="prediction-result" style={{ margin: '0 auto' }}>
              <h3>Prediction Result</h3>
              <p className="result-text">{predictionResult.split('\n').filter(Boolean).pop()}</p>
            </div>
          )}
        </div>
      </div>

      {history.length > 0 && (
        <div className="history-box">
          <h2>Prediction History</h2>
          <div className="history-table-container">
            <table>
              <thead>
                <tr>
                  <th>Image</th>
                  <th>Prediction</th>
                  <th>Timestamp</th>
                </tr>
              </thead>
              <tbody>
                {history.map((entry, index) => (
                  <tr key={index}>
                    <td>
                      <img src={entry.image} alt="Prediction" className="history-img" />
                    </td>
                    <td>{entry.label.split('\n').filter(Boolean).pop()}</td>
                    <td>{entry.timestamp}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
