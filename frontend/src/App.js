import React, { useState } from "react";
import axios from "axios";
import Retina3D from "./Retina3D";
import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [prediction, setPrediction] = useState("");
  const [overlay, setOverlay] = useState("");
  const [mask, setMask] = useState(null);
  const [lesions, setLesions] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
      // Reset previous results
      setPrediction("");
      setOverlay("");
      setMask(null);
      setLesions([]);
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await axios.post("http://127.0.0.1:8000/predict", formData);

      setPrediction(res.data.prediction);
      setOverlay("data:image/png;base64," + res.data.overlay);

      // ✅ DIRECTLY USE MASK (NO CONVERSION)
      setMask(res.data.mask);
      setLesions(res.data.lesions);
    } catch (err) {
      console.error(err);
      setPrediction("Error connecting to server");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>Diabetic Retinopathy Detection</h1>
        <p>Upload a retinal image for AI-powered analysis</p>
      </header>

      <main className="app-main">
        <section className="upload-section card">
          <div className="upload-controls">
            <label className="file-input-label">
              Choose Image
              <input type="file" className="file-input" onChange={handleFileChange} accept="image/*" />
            </label>
            <span className="file-name">{file ? file.name : "No file selected"}</span>
            <button 
              className={`predict-btn ${!file || loading ? 'disabled' : ''}`} 
              onClick={handleUpload}
              disabled={!file || loading}
            >
              {loading ? "Analyzing..." : "Run Analysis"}
            </button>
          </div>
          
          {preview && !overlay && (
            <div className="image-preview-container">
              <img src={preview} alt="Upload Preview" className="preview-image" />
            </div>
          )}
        </section>

        {prediction && (
          <section className="results-section">
            <div className="card prediction-card">
              <h2>Diagnostic Result</h2>
              <div className={`prediction-badge ${prediction.toLowerCase().replace(/\s+/g, '-')}`}>
                {prediction}
              </div>
            </div>

            {overlay && (
              <div className="card visualization-card">
                <h2>Lesion Analysis Overlay</h2>
                <div className="image-container">
                  <img src={overlay} alt="Lesion Overlay" className="result-image" />
                </div>
              </div>
            )}

            {mask && (
              <div className="card model-card">
                <h2>3D Retinal Visualization</h2>
                <div className="model-container">
                  <Retina3D severity={prediction} mask={mask} lesions={lesions} />
                </div>
              </div>
            )}
          </section>
        )}
      </main>
    </div>
  );
}

export default App;