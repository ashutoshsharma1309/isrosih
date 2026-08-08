import { Routes, Route } from "react-router-dom";
import Layout from "./components/layout/Layout.jsx";
import Dashboard from "./pages/Dashboard.jsx";

// Route table for the platform. Later phases add:
//   /predictions/:id  — full prediction detail with SHAP + Grad-CAM views
//   /history          — historical rainfall analysis
//   /alerts           — alert management
export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Dashboard />} />
      </Routes>
    </Layout>
  );
}
