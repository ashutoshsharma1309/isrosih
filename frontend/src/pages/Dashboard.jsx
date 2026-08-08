import { useEffect, useState } from "react";
import RiskMap from "../components/map/RiskMap.jsx";
import SystemStatus from "../components/predictions/SystemStatus.jsx";
import { getActiveAlerts } from "../services/api.js";

// Main monitoring view. Shows live data only — panels state clearly when
// no model is deployed or no alerts exist, rather than showing mock data.
export default function Dashboard() {
  const [alerts, setAlerts] = useState([]);
  const [alertsError, setAlertsError] = useState(null);

  useEffect(() => {
    getActiveAlerts()
      .then(setAlerts)
      .catch((err) => setAlertsError(err.message));
  }, []);

  return (
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
      <section className="lg:col-span-2">
        <h2 className="mb-3 text-sm font-medium uppercase tracking-wider text-slate-400">
          National Rainfall Risk Map
        </h2>
        <RiskMap />
      </section>

      <section className="space-y-6">
        <div>
          <h2 className="mb-3 text-sm font-medium uppercase tracking-wider text-slate-400">
            Active Warnings
          </h2>
          <div className="rounded-lg border border-slate-800 bg-slate-900 p-4 text-sm">
            {alertsError && (
              <p className="text-amber-400">Backend unreachable: {alertsError}</p>
            )}
            {!alertsError && alerts.length === 0 && (
              <p className="text-slate-400">No active early-warning alerts.</p>
            )}
            {alerts.map((alert) => (
              <div key={alert.id} className="mb-3 border-l-4 border-risk-extreme pl-3">
                <p className="font-medium">{alert.region_name}</p>
                <p className="text-slate-300">{alert.message}</p>
              </div>
            ))}
          </div>
        </div>

        <SystemStatus />
      </section>
    </div>
  );
}
