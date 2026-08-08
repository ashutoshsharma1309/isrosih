import { MapContainer, TileLayer } from "react-leaflet";

// India-wide risk map. Phase 8 adds region polygons colored by predicted
// risk level (from GET /api/v1/predictions) and alert markers.
const INDIA_CENTER = [22.5, 79.0];

export default function RiskMap() {
  return (
    <MapContainer
      center={INDIA_CENTER}
      zoom={5}
      className="map-container w-full rounded-lg"
      scrollWheelZoom
    >
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
    </MapContainer>
  );
}
