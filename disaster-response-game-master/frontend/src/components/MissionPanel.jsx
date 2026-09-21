export default function MissionPanel({ missions, victims, onReprioritize }) {
  const victimById = Object.fromEntries(victims.map((v) => [v.id, v]));

  return (
    <div className="panel mission-panel">
      <div className="panel-title">
        MISSION QUEUE
        <span className="count">{missions.length} active</span>
      </div>
      <div className="scroll mission-list">
        {missions.length === 0 && <div className="empty-note">No active missions — waiting on detections.</div>}
        {missions.map((m, i) => {
          const victim = victimById[m.victim_id];
          return (
            <div key={m.id} className="mission-row">
              <div className="mission-rank">{i + 1}</div>
              <div className="mission-body">
                <div className="mission-row-top">
                  <span className="mission-victim">{m.victim_id}</span>
                  <span className="mono-label">{m.assigned_agent}</span>
                </div>
                <div className="mission-row-mid">
                  <div className="priority-bar" title={`priority ${m.priority}`}>
                    <div className="priority-fill" style={{ width: `${m.priority * 100}%` }} />
                  </div>
                  <span className="mono-label">eta {m.eta_seconds}s</span>
                </div>
                {victim && <div className="mono-label">severity {victim.severity.toFixed(2)}</div>}
              </div>
              <button
                className="reprioritize-btn"
                title="Boost priority"
                onClick={() => onReprioritize(m.id, Math.min(1, m.priority + 0.2))}
              >
                ↑
              </button>
            </div>
          );
        })}
      </div>
    </div>
  );
}
