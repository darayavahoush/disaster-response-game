import { useEffect, useRef, useState } from "react";
import { createMockWorld, stepMockWorld, mockKnowledgeQuery } from "./mockEngine.js";

const GATEWAY_HTTP = import.meta.env.VITE_GATEWAY_URL || "http://localhost:4000";
const GATEWAY_WS = GATEWAY_HTTP.replace(/^http/, "ws") + "/ws";
const CONNECT_TIMEOUT_MS = 1500;

/**
 * Provides live world_state, connection mode ("live" | "offline"), and action callbacks.
 * Tries the real gateway first (docs/api-contract.md); if it's not reachable within
 * CONNECT_TIMEOUT_MS, falls back to a local mock loop so the UI is never just a blank screen.
 */
export function useSimulation() {
  const [world, setWorld] = useState(() => createMockWorld("flood"));
  const [mode, setMode] = useState("connecting"); // "connecting" | "live" | "offline"
  const wsRef = useRef(null);
  const mockIntervalRef = useRef(null);

  useEffect(() => {
    let cancelled = false;
    let fellBack = false;

    const startMock = () => {
      if (fellBack || cancelled) return;
      fellBack = true;
      setMode("offline");
      mockIntervalRef.current = setInterval(() => {
        setWorld((w) => stepMockWorld(w));
      }, 1000);
    };

    const timeout = setTimeout(startMock, CONNECT_TIMEOUT_MS);

    try {
      const ws = new WebSocket(GATEWAY_WS);
      wsRef.current = ws;
      ws.onopen = () => {
        if (cancelled) return;
        clearTimeout(timeout);
        setMode("live");
      };
      ws.onmessage = (evt) => {
        if (cancelled) return;
        try {
          const msg = JSON.parse(evt.data);
          if (msg.type === "world_state") setWorld(msg.payload);
        } catch {
          /* ignore malformed frame */
        }
      };
      ws.onerror = () => startMock();
      ws.onclose = () => startMock();
    } catch {
      startMock();
    }

    return () => {
      cancelled = true;
      clearTimeout(timeout);
      if (mockIntervalRef.current) clearInterval(mockIntervalRef.current);
      wsRef.current?.close();
    };
  }, []);

  async function sendCommand(cmd) {
    if (mode === "live") {
      try {
        await fetch(`${GATEWAY_HTTP}/command`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(cmd),
        });
        return;
      } catch {
        /* fall through to local handling below */
      }
    }
    // offline mode: apply the handful of commands the mock loop understands locally
    setWorld((w) => {
      if (cmd.type === "set_scenario") return createMockWorld(cmd.scenario);
      if (cmd.type === "pause") return { ...w, paused: true };
      if (cmd.type === "resume") return { ...w, paused: false };
      if (cmd.type === "set_speed") return { ...w, speed: cmd.multiplier };
      return w;
    });
  }

  async function askKnowledge(question) {
    if (mode === "live") {
      try {
        const r = await fetch(`${GATEWAY_HTTP}/knowledge_query`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question }),
        });
        return await r.json();
      } catch {
        /* fall through */
      }
    }
    return mockKnowledgeQuery(question);
  }

  return { world, mode, sendCommand, askKnowledge };
}
