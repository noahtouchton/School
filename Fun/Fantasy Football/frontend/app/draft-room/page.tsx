"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import {
  espnCheatsheetCsvUrl,
  fetchEspnDraft,
  fetchEspnStatus,
  fetchManualBoard,
  searchDraftPlayers,
  type BoardEntry,
  type DraftRecommendation,
  type EspnDraftState,
  type ManualBoardResponse,
} from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

const POLL_MS = 5000;
const MANUAL_KEY = "manual_draft_state_v1";

const SOURCE_LABEL: Record<string, string> = {
  model: "from your data",
  blend: "your data + market",
  market: "market only (rookie)",
  baseline: "late-round",
};

type ManualState = {
  numTeams: number;
  picksUntilNext: number;
  drafted: string[];
  myPlayers: string[];
};

const EMPTY_MANUAL: ManualState = {
  numTeams: 12,
  picksUntilNext: 11,
  drafted: [],
  myPlayers: [],
};

function RecommendationCard({
  rec,
  index,
}: {
  rec: DraftRecommendation;
  index: number;
}) {
  return (
    <div className={`rounded-lg border p-4 ${index === 0 ? "border-primary bg-primary/5" : ""}`}>
      <div className="flex items-start justify-between gap-2">
        <div className="font-semibold">
          {index + 1}. {rec.name}{" "}
          <span className="text-sm font-normal text-muted-foreground">
            {rec.position} · {rec.nfl_team}
          </span>
          {rec.injury_status && (
            <Badge variant="destructive" className="ml-2">
              {rec.injury_status}
            </Badge>
          )}
        </div>
        <Badge variant={index === 0 ? "default" : "secondary"}>
          {rec.season_points.toFixed(0)} pts
        </Badge>
      </div>
      <div className="mt-1 text-xs text-muted-foreground">
        Tier {rec.tier} · {rec.position}
        {rec.position_rank} · overall #{rec.overall_rank}
        {rec.adp ? ` · ADP ${rec.adp.toFixed(1)}` : ""} · {SOURCE_LABEL[rec.source] ?? rec.source}
      </div>
      <ul className="mt-2 space-y-1 text-sm">
        {rec.reasons.map((r) => (
          <li key={r}>• {r}</li>
        ))}
      </ul>
    </div>
  );
}

function BestAvailable({ players }: { players: BoardEntry[] }) {
  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead className="w-12">#</TableHead>
          <TableHead>Player</TableHead>
          <TableHead>Pos</TableHead>
          <TableHead className="text-right">Season</TableHead>
          <TableHead className="text-right">VORP</TableHead>
          <TableHead className="text-right">ADP</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {players.slice(0, 25).map((p) => (
          <TableRow key={`${p.name}-${p.position}`}>
            <TableCell className="text-muted-foreground">{p.overall_rank}</TableCell>
            <TableCell className="font-medium">{p.name}</TableCell>
            <TableCell>
              {p.position}
              <span className="text-xs text-muted-foreground">{p.position_rank}</span>
            </TableCell>
            <TableCell className="text-right">{p.season_points.toFixed(0)}</TableCell>
            <TableCell className="text-right">{p.vorp.toFixed(1)}</TableCell>
            <TableCell className="text-right text-muted-foreground">
              {p.adp ? p.adp.toFixed(1) : "—"}
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}

export default function DraftRoomPage() {
  const [mode, setMode] = useState<"espn" | "manual">("manual");
  const [espnConnected, setEspnConnected] = useState<boolean | null>(null);
  const [live, setLive] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);

  const [espnState, setEspnState] = useState<EspnDraftState | null>(null);
  const [manual, setManual] = useState<ManualState>(EMPTY_MANUAL);
  const [manualResult, setManualResult] = useState<ManualBoardResponse | null>(null);
  const [search, setSearch] = useState("");
  const [matches, setMatches] = useState<BoardEntry[]>([]);

  // Restore any in-progress manual draft, and default to ESPN mode if linked.
  useEffect(() => {
    const handle = setTimeout(() => {
      const raw = window.localStorage.getItem(MANUAL_KEY);
      if (raw) {
        try {
          setManual({ ...EMPTY_MANUAL, ...JSON.parse(raw) });
        } catch {
          // corrupt saved state is not worth surfacing; start fresh
        }
      }
      fetchEspnStatus()
        .then((s) => {
          setEspnConnected(s.connected);
          if (s.connected) setMode("espn");
        })
        .catch(() => setEspnConnected(false));
    }, 0);
    return () => clearTimeout(handle);
  }, []);

  useEffect(() => {
    window.localStorage.setItem(MANUAL_KEY, JSON.stringify(manual));
  }, [manual]);

  const refreshEspn = useCallback(() => {
    fetchEspnDraft()
      .then((d) => {
        setEspnState(d);
        setError(null);
        setLastUpdated(new Date());
      })
      .catch((e) => setError(String(e.message ?? e)));
  }, []);

  const refreshManual = useCallback(() => {
    fetchManualBoard({
      num_teams: manual.numTeams,
      drafted: manual.drafted,
      my_players: manual.myPlayers,
      picks_until_next: manual.picksUntilNext,
    })
      .then((r) => {
        setManualResult(r);
        setError(null);
        setLastUpdated(new Date());
      })
      .catch((e) => setError(String(e.message ?? e)));
  }, [manual]);

  useEffect(() => {
    if (mode === "espn" && espnConnected) refreshEspn();
  }, [mode, espnConnected, refreshEspn]);

  useEffect(() => {
    if (mode === "manual") refreshManual();
  }, [mode, refreshManual]);

  useEffect(() => {
    if (!live || mode !== "espn") {
      if (timer.current) clearInterval(timer.current);
      timer.current = null;
      return;
    }
    timer.current = setInterval(refreshEspn, POLL_MS);
    return () => {
      if (timer.current) clearInterval(timer.current);
    };
  }, [live, mode, refreshEspn]);

  // Player search for manual mode.
  useEffect(() => {
    const handle = setTimeout(() => {
      if (mode !== "manual" || search.trim().length < 2) {
        setMatches([]);
        return;
      }
      searchDraftPlayers(search, 8)
        .then((r) => setMatches(r.players))
        .catch(() => setMatches([]));
    }, 200);
    return () => clearTimeout(handle);
  }, [search, mode]);

  const markDrafted = (name: string, mine: boolean) => {
    setManual((prev) => ({
      ...prev,
      drafted: prev.drafted.includes(name) ? prev.drafted : [...prev.drafted, name],
      myPlayers: mine && !prev.myPlayers.includes(name) ? [...prev.myPlayers, name] : prev.myPlayers,
    }));
    setSearch("");
    setMatches([]);
  };

  const undoLast = () => {
    setManual((prev) => {
      const last = prev.drafted[prev.drafted.length - 1];
      return {
        ...prev,
        drafted: prev.drafted.slice(0, -1),
        myPlayers: prev.myPlayers.filter((n) => n !== last),
      };
    });
  };

  const resetManual = () => {
    if (!window.confirm("Clear this draft and start over?")) return;
    setManual(EMPTY_MANUAL);
  };

  const recs = mode === "espn" ? espnState?.recommendations : manualResult?.recommendations;
  const available = mode === "espn" ? espnState?.best_available : manualResult?.best_available;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Draft Room</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Pick-by-pick guidance from your own model. Keep your league&apos;s draft open in
            another window and take whoever this board says.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <div className="flex rounded-md border p-0.5">
            <Button
              size="sm"
              variant={mode === "espn" ? "default" : "ghost"}
              onClick={() => setMode("espn")}
              disabled={!espnConnected}
            >
              ESPN live
            </Button>
            <Button
              size="sm"
              variant={mode === "manual" ? "default" : "ghost"}
              onClick={() => setMode("manual")}
            >
              Manual
            </Button>
          </div>
          {espnConnected && (
            <a href={espnCheatsheetCsvUrl()} download>
              <Button variant="outline" size="sm">
                Cheat sheet CSV
              </Button>
            </a>
          )}
          {mode === "espn" && (
            <Button size="sm" variant={live ? "default" : "outline"} onClick={() => setLive(!live)}>
              {live ? "Live · every 5s" : "Go live"}
            </Button>
          )}
          <Button
            size="sm"
            variant="outline"
            onClick={mode === "espn" ? refreshEspn : refreshManual}
          >
            Refresh
          </Button>
        </div>
      </div>

      {error && (
        <div className="rounded-md border border-destructive/50 bg-destructive/10 px-4 py-3 text-sm text-destructive">
          {error}
        </div>
      )}

      {mode === "espn" && !espnConnected && (
        <Card>
          <CardHeader>
            <CardTitle>No ESPN league linked</CardTitle>
            <CardDescription>
              Link one on the{" "}
              <Link className="underline" href="/espn">
                My ESPN Team
              </Link>{" "}
              page, or use Manual mode — it gives the same recommendations, you just click each
              player as he comes off the board.
            </CardDescription>
          </CardHeader>
        </Card>
      )}

      {mode === "espn" && espnState && (
        <div className="grid gap-4 md:grid-cols-4">
          <Card>
            <CardHeader className="pb-2">
              <CardDescription>Picks made</CardDescription>
              <CardTitle className="text-lg">{espnState.drafted}</CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-muted-foreground">
              {espnState.num_teams} teams · {espnState.total_rounds} rounds
              {lastUpdated && (
                <div className="text-xs">updated {lastUpdated.toLocaleTimeString()}</div>
              )}
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardDescription>Current pick</CardDescription>
              <CardTitle className="text-lg">#{espnState.current_pick}</CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-muted-foreground">
              {espnState.my_draft_slot ? `your slot: ${espnState.my_draft_slot}` : "slot unknown"}
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardDescription>Your next pick</CardDescription>
              <CardTitle className="text-lg">
                {espnState.my_next_pick ? `#${espnState.my_next_pick}` : "—"}
              </CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-muted-foreground">
              {espnState.picks_until_my_turn} away
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardDescription>Your roster</CardDescription>
              <CardTitle className="text-lg">{espnState.drafted_by_me} players</CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-muted-foreground">
              {espnState.my_roster_positions.join(" · ") || "—"}
            </CardContent>
          </Card>
        </div>
      )}

      {mode === "manual" && (
        <Card>
          <CardHeader>
            <CardTitle>Track the draft</CardTitle>
            <CardDescription>
              Type a name and mark who took him. Your progress is saved in this browser, so a
              refresh mid-draft won&apos;t lose anything.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex flex-wrap items-center gap-3">
              <label className="text-sm text-muted-foreground">
                Teams
                <Input
                  className="ml-2 inline-block w-20"
                  value={manual.numTeams}
                  onChange={(e) =>
                    setManual((p) => ({ ...p, numTeams: Number(e.target.value) || 12 }))
                  }
                />
              </label>
              <label className="text-sm text-muted-foreground">
                Picks until your turn
                <Input
                  className="ml-2 inline-block w-20"
                  value={manual.picksUntilNext}
                  onChange={(e) =>
                    setManual((p) => ({ ...p, picksUntilNext: Number(e.target.value) || 0 }))
                  }
                />
              </label>
              <span className="text-sm text-muted-foreground">
                {manual.drafted.length} drafted · {manual.myPlayers.length} yours
              </span>
              <Button size="sm" variant="outline" onClick={undoLast} disabled={!manual.drafted.length}>
                Undo last
              </Button>
              <Button size="sm" variant="outline" onClick={resetManual}>
                Reset
              </Button>
            </div>

            <Input
              placeholder="Search a player who just got drafted…"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="max-w-md"
            />
            {matches.length > 0 && (
              <div className="space-y-1 rounded-md border p-2">
                {matches.map((m) => (
                  <div
                    key={`${m.name}-${m.position}`}
                    className="flex items-center justify-between gap-3 rounded px-2 py-1 text-sm hover:bg-muted"
                  >
                    <span>
                      {m.name}{" "}
                      <span className="text-xs text-muted-foreground">
                        {m.position} · {m.nfl_team} · #{m.overall_rank}
                      </span>
                    </span>
                    <span className="space-x-2">
                      <Button size="sm" variant="outline" onClick={() => markDrafted(m.name, false)}>
                        Someone took him
                      </Button>
                      <Button size="sm" onClick={() => markDrafted(m.name, true)}>
                        I drafted him
                      </Button>
                    </span>
                  </div>
                ))}
              </div>
            )}

            {manual.myPlayers.length > 0 && (
              <div className="text-sm">
                <span className="text-muted-foreground">Your team: </span>
                {manual.myPlayers.join(", ")}
              </div>
            )}
            {manualResult && (
              <div className="text-xs text-muted-foreground">
                Still needed:{" "}
                {Object.entries(manualResult.needs.starters_needed)
                  .filter(([, n]) => n > 0)
                  .map(([pos, n]) => `${n}× ${pos}`)
                  .join(", ") || "starters full"}
                {manualResult.needs.flex_needed > 0 && `, ${manualResult.needs.flex_needed}× flex`}
              </div>
            )}
          </CardContent>
        </Card>
      )}

      <Card className={mode === "espn" && espnState?.picks_until_my_turn === 0 ? "border-primary" : ""}>
        <CardHeader>
          <CardTitle>Take this player</CardTitle>
          <CardDescription>
            Ranked for your roster, your league&apos;s settings, and who&apos;s left.
          </CardDescription>
        </CardHeader>
        <CardContent className="grid gap-3 lg:grid-cols-2">
          {recs?.map((rec, i) => (
            <RecommendationCard key={`${rec.name}-${rec.position}`} rec={rec} index={i} />
          ))}
          {recs && recs.length === 0 && (
            <p className="text-sm text-muted-foreground">
              Nothing left to recommend — every position is full.
            </p>
          )}
          {!recs && <p className="text-sm text-muted-foreground">Loading the board…</p>}
        </CardContent>
      </Card>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Best available</CardTitle>
            <CardDescription>Top of the remaining board.</CardDescription>
          </CardHeader>
          <CardContent>{available && <BestAvailable players={available} />}</CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Pick log</CardTitle>
            <CardDescription>Newest first.</CardDescription>
          </CardHeader>
          <CardContent className="max-h-[560px] overflow-y-auto">
            {mode === "espn" ? (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="w-16">Pick</TableHead>
                    <TableHead>Player</TableHead>
                    <TableHead>Team</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {[...(espnState?.picks ?? [])].reverse().map((p) => (
                    <TableRow
                      key={`${p.round}-${p.round_pick}-${p.name}`}
                      className={p.is_mine ? "bg-primary/5" : ""}
                    >
                      <TableCell className="text-muted-foreground">
                        {p.round}.{String(p.round_pick).padStart(2, "0")}
                      </TableCell>
                      <TableCell className="font-medium">{p.name}</TableCell>
                      <TableCell className="text-muted-foreground">
                        {p.team_name}
                        {p.is_mine && (
                          <Badge variant="secondary" className="ml-1">
                            you
                          </Badge>
                        )}
                      </TableCell>
                    </TableRow>
                  ))}
                  {!espnState?.picks?.length && (
                    <TableRow>
                      <TableCell colSpan={3} className="py-6 text-center text-muted-foreground">
                        No picks yet.
                      </TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            ) : (
              <ol className="space-y-1 text-sm">
                {[...manual.drafted].reverse().map((name, i) => (
                  <li key={`${name}-${i}`} className="flex justify-between gap-2">
                    <span>
                      {manual.drafted.length - i}. {name}
                    </span>
                    {manual.myPlayers.includes(name) && (
                      <Badge variant="secondary">you</Badge>
                    )}
                  </li>
                ))}
                {manual.drafted.length === 0 && (
                  <li className="py-6 text-center text-muted-foreground">
                    No picks tracked yet.
                  </li>
                )}
              </ol>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
