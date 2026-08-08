import { ComingSoon } from "@/components/coming-soon";

export default function MyLeaguePage() {
  return (
    <ComingSoon
      title="My League"
      phase="Phase 6"
      description="Link your real ESPN fantasy league and get advice grounded in your own models."
      bullets={[
        "Link a public or private ESPN league (league ID, year, optional espn_s2/SWID)",
        "Pull your real roster, matchup, and free-agent pool",
        "Start/sit recommendations from your own trained projections",
        "Waiver-wire suggestions ranked by projected value added",
      ]}
    />
  );
}
