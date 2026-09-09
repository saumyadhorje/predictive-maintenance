import { useState } from "react";
import "./App.css";

function App() {
  const [formData, setFormData] = useState({
    volt: 170.73008535164,
    rotate: 500.946844111252,
    pressure: 116.004889697097,
    vibration: 41.1890709557334,
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: Number(e.target.value),
    });
  };

  const handlePredict = async () => {
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          ...formData,

          hour: 6,
          day: 20,
          month: 10,
          dayofweek: 1,

          volt_mean_6h: 172.09362128176465,
          volt_std_6h: 4.468636964128972,
          rotate_mean_6h: 442.78246412417866,
          rotate_std_6h: 57.37645952503312,
          pressure_mean_6h: 99.4455484958457,
          pressure_std_6h: 18.86084334307298,
          vibration_mean_6h: 40.143750036929156,
          vibration_std_6h: 6.888124066544264,

          volt_change_1h: -1.5331445348869863,
          rotate_change_1h: 66.496730696697,
          pressure_change_1h: 36.272129950262496,
          vibration_change_1h: 7.240541539664498,

          volt_mean_24h: 173.63644546135905,
          volt_std_24h: 14.91723309910247,
          rotate_mean_24h: 452.47877234333873,
          rotate_std_24h: 50.06144219471292,
          pressure_mean_24h: 96.77482759466402,
          pressure_std_24h: 11.66275431477258,
          vibration_mean_24h: 40.38323178490091,
          vibration_std_24h: 4.958251788027852,

          volt_change_24h: 10.351806715004017,
          rotate_change_24h: 11.994177838117025,
          pressure_change_24h: 12.345391514398003,
          vibration_change_24h: 5.126291008955498,
        }),
      });

      if (!response.ok) {
        throw new Error("Prediction request failed");
      }

      const data = await response.json();

      setResult(data);
    } catch (err) {
      setError(
        "Unable to connect to the prediction API. Make sure FastAPI is running."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="header">
        <div>
          <h1>Predictive Maintenance</h1>
          <p>AI-powered machine failure prediction</p>
        </div>

        <div className="status">
          <span className="status-dot"></span>
          API Connected
        </div>
      </header>

      <main className="dashboard">

        <section className="intro">
          <h2>Machine Health Monitor</h2>
          <p>
            Enter the current machine sensor readings to estimate the
            probability of failure using the trained XGBoost model.
          </p>
        </section>

        <section className="sensor-grid">

          <div className="sensor-card">
            <label>Voltage</label>
            <span>V</span>
            <input
              type="number"
              name="volt"
              value={formData.volt}
              onChange={handleChange}
            />
          </div>

          <div className="sensor-card">
            <label>Rotation</label>
            <span>RPM</span>
            <input
              type="number"
              name="rotate"
              value={formData.rotate}
              onChange={handleChange}
            />
          </div>

          <div className="sensor-card">
            <label>Pressure</label>
            <span>kPa</span>
            <input
              type="number"
              name="pressure"
              value={formData.pressure}
              onChange={handleChange}
            />
          </div>

          <div className="sensor-card">
            <label>Vibration</label>
            <span>mm/s</span>
            <input
              type="number"
              name="vibration"
              value={formData.vibration}
              onChange={handleChange}
            />
          </div>

        </section>

        <button
          className="predict-button"
          onClick={handlePredict}
          disabled={loading}
        >
          {loading ? "Analyzing..." : "Analyze Machine"}
        </button>

        {error && (
          <div className="error">
            {error}
          </div>
        )}

        {result && (
          <section className="result-card">

            <div className="result-header">
              <div>
                <p className="result-label">PREDICTION RESULT</p>
                <h2>Machine Health Status</h2>
              </div>

              <div
                className={
                  result.prediction === "FAILURE"
                    ? "badge danger"
                    : "badge safe"
                }
              >
                {result.prediction === "FAILURE"
                  ? "⚠ FAILURE"
                  : "✓ NO FAILURE"}
              </div>
            </div>

            <div className="probability">
              <strong>
                {(result.failure_probability * 100).toFixed(2)}%
              </strong>

              <span>Failure Probability</span>
            </div>

            <div className="threshold">
              Model threshold: {(result.threshold * 100).toFixed(0)}%
            </div>

          </section>
        )}

      </main>

      <footer>
        <span>Predictive Maintenance System</span>
        <span>Powered by XGBoost + FastAPI + React</span>
      </footer>
    </div>
  );
}

export default App;