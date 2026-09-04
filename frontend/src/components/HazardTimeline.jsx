import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";

export default function HazardTimeline({ hazards }) {
  const data = computeSeries(hazards);

  return (
    <div className="panel hazard-panel">
      <div className="panel-title">
        HAZARD FORECAST
        <span className="count">LSTM projection, next 4 steps</span>
      </div>
      <div className="hazard-chart">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 10, right: 16, bottom: 0, left: -16 }}>
            <CartesianGrid stroke="var(--line-soft)" vertical={false} />
            <XAxis dataKey="step" stroke="var(--muted)" fontSize={10} tickLine={false} axisLine={{ stroke: "var(--line)" }} />
            <YAxis domain={[0, 1]} stroke="var(--muted)" fontSize={10} tickLine={false} axisLine={{ stroke: "var(--line)" }} width={28} />
            <Tooltip
              contentStyle={{ background: "var(--panel-raised)", border: "1px solid var(--line)", fontSize: 11, fontFamily: "var(--font-mono)" }}
              labelStyle={{ color: "var(--muted)" }}
            />
            <Line type="monotone" dataKey="water" stroke="var(--water)" strokeWidth={2} dot={false} />
            <Line type="monotone" dataKey="wind" stroke="var(--wind)" strokeWidth={2} dot={false} />
            <Line type="monotone" dataKey="structural" stroke="var(--quake)" strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function computeSeries(hazards) {
  const avgByType = (type, stepIdx) => {
    const relevant = hazards.filter((h) => h.type === type);
    if (!relevant.length) return 0;
    const sum = relevant.reduce((s, h) => s + (stepIdx === -1 ? h.intensity : h.forecast[stepIdx] ?? h.intensity), 0);
    return Number((sum / relevant.length).toFixed(3));
  };

  return [-1, 0, 1, 2, 3].map((stepIdx) => ({
    step: stepIdx === -1 ? "now" : `+${stepIdx + 1}`,
    water: avgByType("water", stepIdx),
    wind: avgByType("wind", stepIdx),
    structural: avgByType("structural", stepIdx),
  }));
}
