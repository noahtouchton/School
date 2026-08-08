"use client";

import { useEffect, useMemo, useState } from "react";
import { fetchProjections, type Projection } from "@/lib/api";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";

const POSITIONS = ["ALL", "QB", "RB", "WR", "TE"];

const STATUS_VARIANT: Record<string, "default" | "secondary" | "destructive"> = {
  ACT: "secondary",
  Active: "secondary",
  RES: "destructive",
  CUT: "destructive",
  DEV: "default",
};

function statusBadgeVariant(status: string) {
  return STATUS_VARIANT[status] ?? "default";
}

const DIVERGENCE_THRESHOLD = 10;

export default function PlayersPage() {
  const [players, setPlayers] = useState<Projection[]>([]);
  const [season, setSeason] = useState<number | null>(null);
  const [week, setWeek] = useState<number | null>(null);
  const [count, setCount] = useState(0);
  const [search, setSearch] = useState("");
  const [position, setPosition] = useState("ALL");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const handle = setTimeout(() => {
      setLoading(true);
      fetchProjections({
        search: search || undefined,
        position: position === "ALL" ? undefined : position,
        limit: 300,
      })
        .then((data) => {
          setPlayers(data.players);
          setCount(data.count);
          setSeason(data.season);
          setWeek(data.week);
          setError(null);
        })
        .catch((err) => setError(String(err)))
        .finally(() => setLoading(false));
    }, 250);
    return () => clearTimeout(handle);
  }, [search, position]);

  const withSignal = useMemo(() => {
    // ADP rank is computed within the same position group, since raw projected points aren't
    // comparable across positions (QBs score more per game than RB/WR/TE in this format --
    // that's a real feature of fantasy scoring, not a model error). Position-relative rank is
    // what's actually meaningful here; true cross-position value (VORP) comes with the draft
    // tools in a later phase.
    const byPosition = new Map<string, Projection[]>();
    for (const p of players) {
      const list = byPosition.get(p.position) ?? [];
      list.push(p);
      byPosition.set(p.position, list);
    }

    const adpRankByPosition = new Map<string, number>();
    for (const [, list] of byPosition) {
      const byAdp = [...list].filter((p) => p.adp !== null).sort((a, b) => (a.adp as number) - (b.adp as number));
      byAdp.forEach((p, i) => adpRankByPosition.set(p.player_id, i + 1));
    }

    const posRankCounter = new Map<string, number>();
    return players.map((p) => {
      const posRank = (posRankCounter.get(p.position) ?? 0) + 1;
      posRankCounter.set(p.position, posRank);
      const consensusRank = adpRankByPosition.get(p.player_id) ?? null;
      const divergence = consensusRank !== null ? consensusRank - posRank : null;
      return { ...p, posRank, consensusRank, divergence };
    });
  }, [players]);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Players</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Projections from your own trained models
          {season && week ? ` — ${season} Week ${week}` : ""}, updated weekly with recent form,
          Vegas game-script signals, and injury-driven backup boosts. ADP is shown alongside as a
          public sanity check, not copied into the projection. Rank is within position &mdash;
          raw points aren&apos;t comparable across positions (QBs score more per game than
          RB/WR/TE in this format).
        </p>
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <Input
          placeholder="Search players..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="max-w-xs"
        />
        <Select value={position} onValueChange={(value) => setPosition(value ?? "ALL")}>
          <SelectTrigger className="w-32">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {POSITIONS.map((pos) => (
              <SelectItem key={pos} value={pos}>
                {pos}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <span className="text-sm text-muted-foreground">
          {loading ? "Loading…" : `${count} player${count === 1 ? "" : "s"}`}
        </span>
      </div>

      {error && (
        <div className="rounded-md border border-destructive/50 bg-destructive/10 px-4 py-3 text-sm text-destructive">
          Couldn&apos;t reach the backend API ({error}). Is it running on port 8000?
        </div>
      )}

      <div className="rounded-lg border">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="w-16">Pos #</TableHead>
              <TableHead>Name</TableHead>
              <TableHead>Pos</TableHead>
              <TableHead>Team</TableHead>
              <TableHead>Status</TableHead>
              <TableHead className="text-right">Proj</TableHead>
              <TableHead className="text-right">ADP</TableHead>
              <TableHead>vs. Consensus</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {withSignal.map((p) => (
              <TableRow key={p.player_id}>
                <TableCell className="text-muted-foreground">{p.posRank}</TableCell>
                <TableCell className="font-medium">
                  {p.name}
                  {p.boosted_due_to && (
                    <div className="text-xs font-normal text-primary">
                      ↑ boosted &mdash; {p.boosted_due_to} out
                    </div>
                  )}
                </TableCell>
                <TableCell>{p.position}</TableCell>
                <TableCell className="text-muted-foreground">{p.nfl_team}</TableCell>
                <TableCell className="space-x-1">
                  <Badge variant={statusBadgeVariant(p.status)}>{p.status}</Badge>
                  {p.injury_status && <Badge variant="destructive">{p.injury_status}</Badge>}
                </TableCell>
                <TableCell className="text-right font-medium">
                  {p.projected_points.toFixed(1)}
                </TableCell>
                <TableCell className="text-right text-muted-foreground">
                  {p.adp !== null ? p.adp.toFixed(1) : "—"}
                </TableCell>
                <TableCell>
                  {p.divergence === null ? (
                    <span className="text-xs text-muted-foreground">no consensus data</span>
                  ) : Math.abs(p.divergence) >= DIVERGENCE_THRESHOLD ? (
                    <Badge variant={p.divergence > 0 ? "secondary" : "destructive"}>
                      {p.divergence > 0 ? "↑" : "↓"} {Math.abs(p.divergence)} vs ADP
                    </Badge>
                  ) : (
                    <span className="text-xs text-muted-foreground">in line</span>
                  )}
                </TableCell>
              </TableRow>
            ))}
            {!loading && withSignal.length === 0 && (
              <TableRow>
                <TableCell colSpan={8} className="py-8 text-center text-muted-foreground">
                  No players match that search.
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </div>
    </div>
  );
}
