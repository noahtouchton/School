import { ComingSoon } from "@/components/coming-soon";

export default function AiArenaPage() {
  return (
    <ComingSoon
      title="AI Arena"
      phase="Phase 4"
      description="Watch 10 AI-controlled teams draft and play out a full season on real historical data."
      bullets={[
        "10-team AI draft with a plain-English rationale behind every pick",
        "Full season simulation: waivers, trades, weekly matchups",
        "Final standings and a look at which draft strategies won",
      ]}
    />
  );
}
