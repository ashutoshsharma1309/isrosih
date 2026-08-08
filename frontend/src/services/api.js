// Single API client for the VARUNA AI backend. All components go through
// these functions — no fetch calls scattered across components.
import axios from "axios";

// In dev, Vite proxies /api to the backend (see vite.config.js).
// In production, VITE_API_URL points at the deployed API.
const client = axios.create({
  baseURL: import.meta.env.VITE_API_URL || "",
  timeout: 15000,
});

const API = "/api/v1";

export async function getHealth() {
  const { data } = await client.get(`${API}/health`);
  return data;
}

export async function getModelInfo() {
  const { data } = await client.get(`${API}/predictions/model-info`);
  return data;
}

export async function getActiveAlerts() {
  const { data } = await client.get(`${API}/alerts`);
  return data;
}

export async function requestPrediction(payload) {
  const { data } = await client.post(`${API}/predictions`, payload);
  return data;
}
