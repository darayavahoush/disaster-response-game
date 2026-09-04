import { useMemo, useState } from "react";

const CELL = 18;

const HAZARD_COLOR = {
  water: "var(--water)",
  wind: "var(--wind)",
  structural: "var(--quake)",
};

const AGENT_COLOR = {
  drone: "var(--drone)",
  rescue_team: "var(--rescue)",
  vehicle: "var(--vehicle)",
};

function AgentGlyph({ type, color, selected }) {
  const s = selected ? 1.35 : 1;
  if (type === "drone") {
    return (
      <g transform={`scale(${s})`}>
        <circle r="4.5" fill="none" stroke={color} strokeWidth="1.4" />
        <circle r="1.6" fill={color} />
      </g>
    );
  }
  if (type === "rescue_team") {
    return (
      <g transform={`scale(${s})`}>
        <polygon points="0,-5.5 5,4 -5,4" fill="none" stroke={color} strokeWidth="1.4" />
      </g>
    );
  }
  return (
    <g transform={`scale(${s})`}>
      <rect x="-4.5" y="-4.5" width="9" height="9" fill="none" stroke={color} strokeWidth="1.4" />
    </g>
  );
}

export default function TacticalMap({ world, selectedAgentId, onSelectAgent }) {
  const { grid, hazards, blocked_routes, victims, agents } = world;
  const width = grid.width * CELL;
  const height = grid.height * CELL;
  const [hoveredVictim, setHoveredVictim] = useState(null);

  const hazardCircles = useMemo(
    () =>
      hazards.map((h, i) => (
        <circle
          key={i}
          cx={h.x * CELL}
          cy={h.y * CELL}
          r={CELL * 1.8}
          fill={HAZARD_COLOR[h.type] || "var(--quake)"}
          opacity={Math.min(0.28, h.intensity * 0.32)}
        />
      )),
    [hazards]
  );

  return (
    <div className="tactical-map-wrap">
      <svg
        className="tactical-map"
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="xMidYMid meet"
        role="img"
        aria-label="Tactical map of the disaster zone"
      >
        <defs>
          <pattern id="grid-pattern" width={CELL} height={CELL} patternUnits="userSpaceOnUse">
            <path d={`M ${CELL} 0 L 0 0 0 ${CELL}`} fill="none" stroke="var(--line-soft)" strokeWidth="0.5" />
          </pattern>
        </defs>

        <rect width={width} height={height} fill="url(#grid-pattern)" />
        {hazardCircles}

        {blocked_routes.map((b, i) => (
          <g key={i}>
            <line
              x1={b.from[0] * CELL}
              y1={b.from[1] * CELL}
              x2={b.to[0] * CELL}
              y2={b.to[1] * CELL}
              stroke="var(--alert)"
              strokeWidth="3"
              strokeLinecap="round"
              opacity="0.85"
            />
            <line
              x1={b.from[0] * CELL}
              y1={b.from[1] * CELL}
              x2={b.to[0] * CELL}
              y2={b.to[1] * CELL}
              stroke="var(--ink)"
              strokeWidth="1"
              strokeDasharray="2 2"
            />
          </g>
        ))}

        {agents.map(
          (a) =>
            a.route &&
            a.route.length > 1 && (
              <polyline
                key={`route-${a.id}`}
                points={a.route.map(([x, y]) => `${x * CELL},${y * CELL}`).join(" ")}
                fill="none"
                stroke={AGENT_COLOR[a.type]}
                strokeWidth={a.id === selectedAgentId ? 1.6 : 1}
                strokeDasharray="3 3"
                opacity={a.id === selectedAgentId ? 0.9 : 0.35}
              />
            )
        )}

        {victims.map((v) => {
          if (v.status === "hidden") return null;
          const color = v.status === "rescued" ? "var(--safe)" : v.severity > 0.7 ? "var(--alert)" : "var(--paper)";
          return (
            <g
              key={v.id}
              transform={`translate(${v.pos[0] * CELL}, ${v.pos[1] * CELL})`}
              onMouseEnter={() => setHoveredVictim(v.id)}
              onMouseLeave={() => setHoveredVictim(null)}
              style={{ cursor: "pointer" }}
            >
              <circle r="3" fill={color} opacity={v.status === "rescued" ? 0.5 : 1} />
              {v.status !== "rescued" && <circle r="6" fill="none" stroke={color} strokeWidth="0.75" opacity="0.5" />}
              {hoveredVictim === v.id && (
                <text x="8" y="3" fontSize="9" fill="var(--paper)" fontFamily="var(--font-mono)">
                  {v.id} · sev {v.severity.toFixed(2)} · {v.status}
                </text>
              )}
            </g>
          );
        })}

        {agents.map((a) => (
          <g
            key={a.id}
            transform={`translate(${a.pos[0] * CELL}, ${a.pos[1] * CELL})`}
            onClick={() => onSelectAgent(a.id === selectedAgentId ? null : a.id)}
            style={{ cursor: "pointer" }}
          >
            <AgentGlyph type={a.type} color={AGENT_COLOR[a.type]} selected={a.id === selectedAgentId} />
          </g>
        ))}
      </svg>

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
