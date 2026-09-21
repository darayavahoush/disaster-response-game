const TYPE_LABEL = { drone: "DRONE", rescue_team: "RESCUE TEAM", vehicle: "VEHICLE" };
const STATUS_LABEL = {
  idle: "idle",
  scanning: "scanning",
  moving: "en route",
  delivering: "delivering",
  returning: "returning",
};

function BatteryBar({ value }) {
  const pct = Math.round(value * 100);
  const color = pct < 25 ? "var(--alert)" : "var(--safe)";
  return (
    <div className="battery-bar" aria-label={`Battery ${pct}%`}>
      <div className="battery-fill" style={{ width: `${pct}%`, background: color }} />
    </div>
  );
}

export default function FleetPanel({ agents, missions, selectedAgentId, onSelectAgent }) {
  const missionByAgent = Object.fromEntries(missions.map((m) => [m.assigned_agent, m]));

  return (
    <div className="panel fleet-panel">
      <div className="panel-title">
        FLEET
        <span className="count">{agents.length} units</span>
      </div>
      <div className="scroll fleet-list">
        {agents.map((a) => {
          const mission = missionByAgent[a.id];
          const selected = a.id === selectedAgentId;
          return (
            <button
              key={a.id}
              className={`fleet-row type-${a.type} ${selected ? "selected" : ""}`}
              onClick={() => onSelectAgent(selected ? null : a.id)}
            >
              <div className="fleet-row-top">
                <span className="fleet-id">{a.id}</span>
                <span className="mono-label type-tag">{TYPE_LABEL[a.type]}</span>
              </div>
              <div className="fleet-row-mid">
                <span className={`status-dot status-${a.status}`} />
                <span className="fleet-status">{STATUS_LABEL[a.status] || a.status}</span>
                {a.type === "drone" && <BatteryBar value={a.battery} />}
              </div>
              {mission && (
                <div className="fleet-row-mission mono-label">
                  → {mission.victim_id} · eta {mission.eta_seconds}s
                </div>
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
}
