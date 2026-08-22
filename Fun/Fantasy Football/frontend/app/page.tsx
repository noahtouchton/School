import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { fetchPlayers } from "@/lib/api";
import Link from "next/link";

const SHIPPED = [
  {
    title: "My ESPN Team",
    body: "Your real ESPN league is linked with browser cookies. One click sets the optimal lineup and the waiver scanner finds upgrades.",
    href: "/espn",
    phase: "Phase 6",
  },
  {
    title: "Draft Room",
    body: "Live draft assistant that ranks every pick for your roster and league settings — plus a manual mode that needs no league connected.",
    href: "/draft-room",
    phase: "Phase 5",
  },
];

const UPCOMING = [
  {
    title: "AI Arena",
    body: "Watch 10 AI-controlled teams draft and play a full season, with each pick explained in plain English.",
    phase: "Phase 4",
  },
  {
    title: "Training",
    body: "Run the evolutionary trainer across simulated seasons and watch the agent's strategy improve generation over generation.",
    phase: "Phase 3",
  },
  {
    title: "Mock Draft vs AI",
    body: "Draft against AI opponents, then talk to Claude or Gemini to tune how the draft engine values players.",
    phase: "Phase 5",
  },
];

export default async function DashboardPage() {
  let playerCount: number | null = null;
  try {
    const data = await fetchPlayers({ limit: 1 });
    playerCount = data.count;
  } catch {
    playerCount = null;
  }

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Welcome back</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          The data pipeline and player database are live. Everything else builds on top of it.
        </p>
      </div>

      <div className="grid gap-4 sm:grid-cols-3">
        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Players tracked</CardDescription>
            <CardTitle className="text-3xl">{playerCount === null ? "—" : playerCount}</CardTitle>
          </CardHeader>
          <CardContent className="text-xs text-muted-foreground">
            Live current-season rosters, refreshed on every ingestion run.
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Historical seasons</CardDescription>
            <CardTitle className="text-3xl">2016&ndash;2025</CardTitle>
          </CardHeader>
          <CardContent className="text-xs text-muted-foreground">
            Weekly stats and rosters cached for training and simulation.
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Build status</CardDescription>
            <CardTitle className="text-3xl">Phase 6</CardTitle>
          </CardHeader>
          <CardContent className="text-xs text-muted-foreground">
            Real-league play is live. Phases 3&ndash;4 (simulation and evolutionary training)
            are still to come.
          </CardContent>
        </Card>
      </div>

      <div>
        <h2 className="mb-3 text-sm font-medium text-muted-foreground">Ready to use</h2>
        <div className="grid gap-4 sm:grid-cols-2">
          {SHIPPED.map((item) => (
            <Link key={item.title} href={item.href}>
              <Card className="h-full transition-colors hover:border-primary">
                <CardHeader className="flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-base">{item.title}</CardTitle>
                  <span className="rounded-full bg-primary px-2 py-0.5 text-xs text-primary-foreground">
                    {item.phase}
                  </span>
                </CardHeader>
                <CardContent className="text-sm text-muted-foreground">{item.body}</CardContent>
              </Card>
            </Link>
          ))}
        </div>
      </div>

      <div>
        <h2 className="mb-3 text-sm font-medium text-muted-foreground">Coming up</h2>
        <div className="grid gap-4 sm:grid-cols-2">
          {UPCOMING.map((item) => (
            <Card key={item.title}>
              <CardHeader className="flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-base">{item.title}</CardTitle>
                <span className="rounded-full bg-secondary px-2 py-0.5 text-xs text-secondary-foreground">
                  {item.phase}
                </span>
              </CardHeader>
              <CardContent className="text-sm text-muted-foreground">{item.body}</CardContent>
            </Card>
          ))}
        </div>
      </div>
    </div>
  );
}
