import React, { useState, useEffect } from 'react';
import axios from 'axios';

function App() {
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        // Replace with your actual input data
        const inputData = [658, -4364, 6102, -1091, -229, 238, 243];

        const response = await axios.post('http://localhost:8000/predict/', {
          features: inputData
        });

        setPrediction(response.data.prediction);
      } catch (err) {
        setError("Error making prediction");
      }
    };

    fetchData();
  }, []);

  return (
    <div className="App">
      <h1>Sales Prediction</h1>
      {prediction && (
        <div>
          <h2>Prediction: {prediction}</h2>
        </div>
      )}

      {error && <div style={{ color: "red" }}>{error}</div>}
    </div>
  );
}

export default App;