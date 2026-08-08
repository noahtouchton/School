import { ComingSoon } from "@/components/coming-soon";

export default function SettingsPage() {
  return (
    <ComingSoon
      title="Settings"
      phase="Phases 5 & 6"
      description="Configure the pieces that connect this app to the outside world."
      bullets={[
        "ESPN league link (public/private) management",
        "LLM provider selection (Claude or Gemini) and key status",
        "Scoring rules and roster settings for the sandbox league",
      ]}
    />
  );
}
