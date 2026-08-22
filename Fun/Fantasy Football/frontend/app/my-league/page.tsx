import { redirect } from "next/navigation";

// Phase 6 shipped as the ESPN integration, so this route is a duplicate. Kept as
// a redirect rather than deleted because it was the linked-to name for a while.
export default function MyLeaguePage() {
  redirect("/espn");
}
