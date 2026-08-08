import { ComingSoon } from "@/components/coming-soon";

export default function TrainingPage() {
  return (
    <ComingSoon
      title="Training"
      phase="Phase 3"
      description="Run the evolutionary trainer and watch the agent's draft/roster strategy improve over generations."
      bullets={[
        "Population of agent parameter sets competing across simulated seasons",
        "Generation-over-generation fitness chart (wins + points for)",
        "Promote a trained parameter set to power your Mock Draft and AI Arena opponents",
      ]}
    />
  );
}
