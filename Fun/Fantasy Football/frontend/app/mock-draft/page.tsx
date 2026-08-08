import { ComingSoon } from "@/components/coming-soon";

export default function MockDraftPage() {
  return (
    <ComingSoon
      title="Mock Draft"
      phase="Phase 5"
      description="Draft live against AI opponents, then tune the draft engine's brain in a live chat."
      bullets={[
        "Live snake draft against AI agents that actually react to roster needs and the board",
        "Chat with Claude or Gemini to adjust how the engine values positions, risk, and upside",
        "Instant backtest feedback showing how a tuning change would've changed past results",
      ]}
    />
  );
}
