import { Pause, Play, Radio, WifiOff } from "lucide-react";

const SCENARIOS = [
  { id: "flood", label: "Flood" },
  { id: "cyclone", label: "Cyclone" },
  { id: "earthquake", label: "Earthquake" },
];

function formatClock(seconds) {
  const s = Math.floor(seconds % 60);
  const m = Math.floor((seconds / 60) % 60);
  const h = Math.floor(seconds / 3600);
  return [h, m, s].map((n) => String(n).padStart(2, "0")).join(":");
}

export default function TopBar({ world, mode, onCommand }) {
  const { scenario, clock_seconds, paused, speed, metrics } = world;

  return (
    <header className="top-bar">
      <div className="top-bar-brand">
        <span className="brand-mark">AEGIS</span>
        <span className="brand-sub">disaster response coordination</span>
      </div>

      <div className="top-bar-scenarios" role="group" aria-label="Scenario">
        {SCENARIOS.map((s) => (
          <button
            key={s.id}
            className={`scenario-chip ${scenario === s.id ? "active" : ""}`}
            data-scenario={s.id}
            onClick={() => onCommand({ type: "set_scenario", scenario: s.id })}
          >
            {s.label}
          </button>
        ))}
      </div>

      <div className="top-bar-clock">
        <span className="mono-label">MISSION CLOCK</span>
        <span className="clock-value">{formatClock(clock_seconds)}</span>
      </div>

      <div className="top-bar-controls">
        <button
          className="icon-btn"
          aria-label={paused ? "Resume" : "Pause"}
          onClick={() => onCommand({ type: paused ? "resume" : "pause" })}
        >
          {paused ? <Play size={15} /> : <Pause size={15} />}
        </button>
        {[1, 2, 4].map((mult) => (
          <button
            key={mult}
            className={`speed-btn ${speed === mult ? "active" : ""}`}
            onClick={() => onCommand({ type: "set_speed", multiplier: mult })}
          >
            {mult}x
          </button>
        ))}
      </div>

      <div className="top-bar-metric">
        <span className="metric-value">
          {metrics.victims_rescued}
          <span className="metric-of">/{metrics.victims_total}</span>
        </span>
        <span className="mono-label">RESCUED</span>
      </div>

      <div className={`conn-indicator ${mode}`}>
        {mode === "live" ? <Radio size={13} /> : <WifiOff size={13} />}
        <span>{mode === "live" ? "LIVE ENGINE" : mode === "offline" ? "OFFLINE DEMO" : "CONNECTING"}</span>
      </div>
    </header>
  );
}
