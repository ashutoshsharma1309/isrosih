/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        // Risk-level palette used consistently across map zones,
        // badges, and alerts (matches RiskLevel enum in the backend).
        risk: {
          low: "#22c55e",
          moderate: "#eab308",
          heavy: "#f97316",
          extreme: "#dc2626",
        },
      },
    },
  },
  plugins: [],
};
