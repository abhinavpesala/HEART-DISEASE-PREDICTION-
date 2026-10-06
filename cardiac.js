import { useState } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import CardiacMesh from "./components/CardiacMesh";
import "./App.css";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export default function App() {
  const [form, setForm] = useState({
    Age: 60,
    Sex: "Male",
    DM: 1,
    HTN: 1,
    BP: 130,
    PR: 75,
    Typical_Chest_Pain: 1,
    Atypical: 0,
    Nonanginal: 0,
    Dyspnea: 0,
    EF_TTE: 50,
    Region_RWMA: "N",
    VHD: "N",
    BBB: "N",
    Function_Class: "N",
    Q_Wave: 0,
    St_Elevation: 0,
    St_Depression: 0,
    Tinversion: 0,
    LVH: 0
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  async function handlePredict() {
    setLoading(true);
    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form)
      });

      if (!res.ok) throw new Error("API error");
      const data = await res.json();
      setResult(data);
    } catch (err) {
      alert("Backend error: " + err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="container">
      <div className="sidebar">
        <h1>🫀 Cardiac Risk Engine</h1>

        <div className="input-group">
          <label>Age</label>
          <input 
            type="number" 
            value={form.Age} 
            onChange={(e) => setForm({ ...form, Age: Number(e.target.value) })} 
          />
        </div>

        <div className="input-group">
          <label>Sex</label>
          <select 
            value={form.Sex} 
            onChange={(e) => setForm({ ...form, Sex: e.target.value })}
          >
            <option>Male</option>
            <option>Female</option>
          </select>
        </div>

        <div className="input-group">
          <label>
            <input 
              type="checkbox" 
              checked={form.DM === 1} 
              onChange={(e) => setForm({ ...form, DM: e.target.checked ? 1 : 0 })} 
            />
            Diabetes
          </label>
        </div>

        <div className="input-group">
          <label>
            <input 
              type="checkbox" 
              checked={form.HTN === 1} 
              onChange={(e) => setForm({ ...form, HTN: e.target.checked ? 1 : 0 })} 
            />
            Hypertension
          </label>
        </div>

        <div className="input-group">
          <label>BP (systolic)</label>
          <input 
            type="number" 
            value={form.BP} 
            onChange={(e) => setForm({ ...form, BP: Number(e.target.value) })} 
          />
        </div>

        <div className="input-group">
          <label>PR (heart rate)</label>
          <input 
            type="number" 
            value={form.PR} 
            onChange={(e) => setForm({ ...form, PR: Number(e.target.value) })} 
          />
        </div>

        <div className="input-group">
          <label>EF-TTE (%)</label>
          <input 
            type="number" 
            value={form.EF_TTE} 
            onChange={(e) => setForm({ ...form, EF_TTE: Number(e.target.value) })} 
          />
        </div>

        <div className="input-group">
          <label>
            <input 
              type="checkbox" 
              checked={form.Typical_Chest_Pain === 1} 
              onChange={(e) => setForm({ ...form, Typical_Chest_Pain: e.target.checked ? 1 : 0 })} 
            />
            Typical Chest Pain
          </label>
        </div>

        <button onClick={handlePredict} disabled={loading} className="btn-predict">
          {loading ? "Predicting..." : "Predict Risk"}
        </button>

        {result && (
          <div className="results">
            <h3>Risk Assessment</h3>
            <div className={`risk-badge ${result.risk_label.toLowerCase()}`}>
              {result.risk_label}
            </div>
            <p><strong>Score:</strong> {(result.risk_score * 100).toFixed(1)}%</p>
            <p><strong>Confidence:</strong> {(result.confidence * 100).toFixed(1)}%</p>
            <hr />
            <h4>Branch Scores</h4>
            <p>📊 Tabular: {(result.branch_scores.tabular * 100).toFixed(1)}%</p>
            <p>❤️ ECG: {(result.branch_scores.ecg * 100).toFixed(1)}%</p>
            <p>🔊 Echo: {(result.branch_scores.echo * 100).toFixed(1)}%</p>
          </div>
        )}
      </div>

      <div className="canvas-container">
        <Canvas camera={{ position: [0, 0, 5], fov: 45 }}>
          <ambientLight intensity={0.7} />
          <directionalLight position={[2, 2, 2]} />
          <CardiacMesh risk={result ? result.risk_score : 0.4} />
          <OrbitControls />
        </Canvas>
      </div>
    </div>
  );
}