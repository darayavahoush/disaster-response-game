const LABEL = { medkits: "medkits", water: "water", evac_capacity: "evac capacity" };

export default function ResourcePanel({ resources, metrics }) {
  return (
    <div className="panel resource-panel">
      <div className="panel-title">RESOURCES</div>
      <div className="resource-body">
        {Object.entries(resources).map(([key, r]) => {
          const pct = Math.round((r.available / r.total) * 100);
          return (
            <div key={key} className="resource-row">
              <div className="resource-row-top">
                <span>{LABEL[key] || key}</span>
                <span className="mono-label">
                  {r.available}/{r.total}
                </span>
              </div>
              <div className="resource-bar">
                <div className="resource-fill" style={{ width: `${pct}%` }} />
              </div>
            </div>
          );
        })}
        <div className="metrics-strip">
          <div>
            <span className="metrics-value">{metrics.avg_response_seconds}s</span>
            <span className="mono-label">avg response</span>
          </div>
          <div>
            <span className="metrics-value">{metrics.resource_efficiency.toFixed(2)}</span>
            <span className="mono-label">efficiency</span>
          </div>
        </div>
      </div>
    </div>
  );
}
