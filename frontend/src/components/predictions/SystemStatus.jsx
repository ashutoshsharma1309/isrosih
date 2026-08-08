import { useEffect, useState } from "react";
import { getHealth, getModelInfo } from "../../services/api.js";

// Transparency panel: reports real backend/model status so evaluators can
// see exactly which components are live at any point in development.
export default function SystemStatus() {
  const [health, setHealth] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([getHealth(), getModelInfo()])
      .then(([h, m]) => {
        setHealth(h);
        setModelInfo(m);
      })
      .catch((err) => setError(err.message));
  }, []);

  return (
    <div>
      <h2 className="mb-3 text-sm font-medium uppercase tracking-wider text-slate-400">
        System Status
      </h2>
      <div className="space-y-2 rounded-lg border border-slate-800 bg-slate-900 p-4 text-sm">
        {error && <p className="text-amber-400">Backend unreachable: {error}</p>}
        {health && (
          <p>
            API: <span className="text-risk-low">{health.status}</span>{" "}
            <span className="text-slate-500">
              (v{health.version}, {health.environment})
            </span>
          </p>
        )}
        {modelInfo && (
          <p>
            Prediction model:{" "}
            {modelInfo.model_loaded ? (
              <span className="text-risk-low">loaded</span>
            ) : (
              <span className="text-slate-400">not yet deployed (Phases 3-6)</span>
            )}
          </p>
        )}
      </div>
    </div>
  );
}
