import { useMemo, useState } from "react";
import { MapContainer, TileLayer, Circle, CircleMarker, Polyline, Marker, Tooltip } from "react-leaflet";
import L from "leaflet";
import { gridToLatLng, anchorBounds, metersPerCell } from "../lib/geo.js";

const HAZARD_COLOR = {
  water: "#4fa8e8",
  wind: "#a78bfa",
  structural: "#f0755a",
};

const AGENT_COLOR = {
  drone: "#57e0c7",
  rescue_team: "#f0755a",
  vehicle: "#f5b94d",
};

const AGENT_SVG = {
  drone: (color) =>
    `<svg viewBox="-7 -7 14 14" width="100%" height="100%"><circle r="4.5" fill="none" stroke="${color}" stroke-width="1.4"/><circle r="1.6" fill="${color}"/></svg>`,
  rescue_team: (color) =>
    `<svg viewBox="-7 -7 14 14" width="100%" height="100%"><polygon points="0,-5.5 5,4 -5,4" fill="none" stroke="${color}" stroke-width="1.4"/></svg>`,
  vehicle: (color) =>
    `<svg viewBox="-7 -7 14 14" width="100%" height="100%"><rect x="-4.5" y="-4.5" width="9" height="9" fill="none" stroke="${color}" stroke-width="1.4"/></svg>`,
};

function agentIcon(type, selected) {
  const color = AGENT_COLOR[type] || AGENT_COLOR.vehicle;
  const size = selected ? 30 : 22;
  const svgFn = AGENT_SVG[type] || AGENT_SVG.vehicle;
  return L.divIcon({
    className: "agent-marker",
    html: svgFn(color),
    iconSize: [size, size],
    iconAnchor: [size / 2, size / 2],
  });
}

export default function TacticalMap({ world, selectedAgentId, onSelectAgent }) {
  const { grid, hazards, blocked_routes, victims, agents } = world;
  const [hoveredVictim, setHoveredVictim] = useState(null);

  const bounds = useMemo(() => anchorBounds(), []);
  const cellMeters = useMemo(() => metersPerCell(grid), [grid]);
  const toLL = (pos) => gridToLatLng(pos, grid);

  return (
    <div className="tactical-map-wrap">
      <MapContainer
        className="tactical-map"
        bounds={bounds}
        boundsOptions={{ padding: [8, 8] }}
        zoomControl={true}
        attributionControl={false}
        preferCanvas={true}
      >
        <TileLayer
          url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
          attribution="Tiles &copy; Esri"
          maxZoom={19}
        />
        <TileLayer
          url="https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}"
          maxZoom={19}
        />

        {hazards.map((h, i) => (
          <Circle
            key={i}
            center={toLL([h.x, h.y])}
            radius={cellMeters * 1.8}
            pathOptions={{
              color: HAZARD_COLOR[h.type] || HAZARD_COLOR.structural,
              fillColor: HAZARD_COLOR[h.type] || HAZARD_COLOR.structural,
              fillOpacity: Math.min(0.28, h.intensity * 0.32),
              stroke: false,
            }}
          />
        ))}

        {blocked_routes.map((b, i) => (
          <Polyline
            key={i}
            positions={[toLL(b.from), toLL(b.to)]}
            pathOptions={{ color: "#f5b94d", weight: 4, opacity: 0.85, dashArray: "2 4" }}
          />
        ))}

        {agents.map(
          (a) =>
            a.route &&
            a.route.length > 1 && (
              <Polyline
                key={`route-${a.id}`}
                positions={a.route.map(toLL)}
                pathOptions={{
                  color: AGENT_COLOR[a.type],
                  weight: a.id === selectedAgentId ? 2.5 : 1.5,
                  dashArray: "3 5",
                  opacity: a.id === selectedAgentId ? 0.9 : 0.35,
                }}
              />
            )
        )}

        {victims.map((v) => {
          if (v.status === "hidden") return null;
          const color = v.status === "rescued" ? "#3fcb8c" : v.severity > 0.7 ? "#f5b94d" : "#e7ecf5";
          return (
            <CircleMarker
              key={v.id}
              center={toLL(v.pos)}
              radius={v.status === "rescued" ? 4 : 6}
              pathOptions={{
                color,
                fillColor: color,
                fillOpacity: v.status === "rescued" ? 0.5 : 0.9,
                weight: 1.2,
              }}
              eventHandlers={{
                mouseover: () => setHoveredVictim(v.id),
                mouseout: () => setHoveredVictim(null),
              }}
            >
              {hoveredVictim === v.id && (
                <Tooltip permanent direction="right" opacity={1} className="map-tooltip">
                  {v.id} · sev {v.severity.toFixed(2)} · {v.status}
                </Tooltip>
              )}
            </CircleMarker>
          );
        })}

        {agents.map((a) => (
          <Marker
            key={a.id}
            position={toLL(a.pos)}
            icon={agentIcon(a.type, a.id === selectedAgentId)}
            eventHandlers={{
              click: () => onSelectAgent(a.id === selectedAgentId ? null : a.id),
            }}
          />
        ))}
      </MapContainer>

      <div className="map-legend">
        <span>
          <i className="dot" style={{ background: "var(--water)" }} /> flood
        </span>
        <span>
          <i className="dot" style={{ background: "var(--wind)" }} /> wind
        </span>
        <span>
          <i className="dot" style={{ background: "var(--quake)" }} /> structural
        </span>
        <span className="sep" />
        <span>
          <i className="glyph-circle" /> drone
        </span>
        <span>
          <i className="glyph-tri" /> rescue team
        </span>
        <span>
          <i className="glyph-sq" /> vehicle
        </span>
        <span className="sep" />
        <span>
          <i className="line-dash" /> blocked route
        </span>
      </div>
    </div>
  );
}
