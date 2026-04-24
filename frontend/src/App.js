import React, { useState } from "react";
import axios from "axios";
import Retina3D from "./Retina3D";

function App() {
  const [file, setFile] = useState(null);
  const [prediction, setPrediction] = useState("");
  const [overlay, setOverlay] = useState("");
  const [mask, setMask] = useState(null);
  const [lesions, setLesions] = useState([]);

  const handleUpload = async () => {
    const formData = new FormData();
    formData.append("file", file);

    const res = await axios.post("http://127.0.0.1:8000/predict", formData);

    setPrediction(res.data.prediction);
    setOverlay("data:image/png;base64," + res.data.overlay);

    // ✅ DIRECTLY USE MASK (NO CONVERSION)
    setMask(res.data.mask);
    setLesions(res.data.lesions);
  };

  return (
    <div style={{ textAlign: "center" }}>
      <h1>DR Detection</h1>

      <input type="file" onChange={(e) => setFile(e.target.files[0])} />
      <br /><br />

      <button onClick={handleUpload}>Predict</button>

      <h2>{prediction}</h2>

      {overlay && (
        <div>
          <h3>Lesion Overlay</h3>
          <img src={overlay} width="300" alt="" />
        </div>
      )}

      <h3>3D Retina View</h3>

      {mask && (
  <Retina3D severity={prediction} mask={mask} lesions={lesions} />
)}
    </div>
  );
}

export default App;