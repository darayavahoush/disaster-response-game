import { useState } from "react";
import { useSimulation } from "./lib/useSimulation.js";
import TopBar from "./components/TopBar.jsx";
import TacticalMap from "./components/TacticalMap.jsx";
import FleetPanel from "./components/FleetPanel.jsx";
import MissionPanel from "./components/MissionPanel.jsx";
import ResourcePanel from "./components/ResourcePanel.jsx";
import HazardTimeline from "./components/HazardTimeline.jsx";
import KnowledgePanel from "./components/KnowledgePanel.jsx";

export default function App() {
  const { world, mode, sendCommand, askKnowledge } = useSimulation();
  const [selectedAgentId, setSelectedAgentId] = useState(null);

  return (
    <div className="app-shell">
      <TopBar world={world} mode={mode} onCommand={sendCommand} />

      <main className="app-grid">
        <FleetPanel
          agents={world.agents}
          missions={world.missions}
          selectedAgentId={selectedAgentId}
          onSelectAgent={setSelectedAgentId}
        />

        <div className="map-column">
          <TacticalMap world={world} selectedAgentId={selectedAgentId} onSelectAgent={setSelectedAgentId} />
        </div>

        <div className="right-column">
          <MissionPanel
            missions={world.missions}
            victims={world.victims}
            onReprioritize={(mission_id, priority) => sendCommand({ type: "reprioritize_mission", mission_id, priority })}
          />
          <ResourcePanel resources={world.resources} metrics={world.metrics} />
        </div>

        <div className="bottom-strip">
          <HazardTimeline hazards={world.hazards} />
          <KnowledgePanel askKnowledge={askKnowledge} />
        </div>
      </main>
    </div>
  );
}
