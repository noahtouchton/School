"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import {
  applyEspnLineup,
  connectEspn,
  disconnectEspn,
  espnTransaction,
  fetchEspnStatus,
  fetchEspnTeam,
  fetchEspnWaivers,
  type EspnStatus,
  type EspnTeamResponse,
  type WaiverRec,
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

function Message({ kind, children }: { kind: "error" | "ok"; children: React.ReactNode }) {
  return (
    <div
      className={
        kind === "error"
          ? "rounded-md border border-destructive/50 bg-destructive/10 px-4 py-3 text-sm text-destructive"
          : "rounded-md border border-primary/40 bg-primary/10 px-4 py-3 text-sm"
      }
    >
      {children}
    </div>
  );
}

export default function EspnPage() {
  const [status, setStatus] = useState<EspnStatus | null>(null);
  const [team, setTeam] = useState<EspnTeamResponse | null>(null);
  const [waivers, setWaivers] = useState<WaiverRec[] | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const [leagueId, setLeagueId] = useState("");
  const [year, setYear] = useState(String(new Date().getFullYear()));
  const [s2, setS2] = useState("");
  const [swid, setSwid] = useState("");

  const loadStatus = useCallback(() => {
    fetchEspnStatus()
      .then(setStatus)
      .catch((e) => setError(String(e.message ?? e)));
  }, []);

  useEffect(loadStatus, [loadStatus]);

  const loadTeam = useCallback(() => {
    if (!status?.connected) return;
    fetchEspnTeam()
      .then((data) => {
        setTeam(data);
        setError(null);
      })
      .catch((e) => setError(String(e.message ?? e)));
    fetchEspnWaivers()
      .then((d) => setWaivers(d.recommendations))
      .catch(() => setWaivers([]));
  }, [status?.connected]);

  useEffect(loadTeam, [loadTeam]);

  const connect = async () => {
    setBusy(true);
    setError(null);
    try {
      const res = await connectEspn({
        league_id: Number(leagueId),
        year: Number(year),
        espn_s2: s2 || undefined,
        swid: swid || undefined,
      });
      setNotice(
        res.identified_my_team
          ? `Connected to ${res.league_name ?? "your league"} — found your team "${res.my_team}".`
          : `Connected to ${res.league_name ?? "your league"}, but couldn't match your SWID to a team. Check the SWID cookie.`
      );
      loadStatus();
    } catch (e) {
      setError(String((e as Error).message ?? e));
    } finally {
      setBusy(false);
    }
  };

  const disconnect = async () => {
    if (!window.confirm("Disconnect this ESPN league?")) return;
    await disconnectEspn();
    setTeam(null);
    setWaivers(null);
    loadStatus();
  };

  const applyLineup = async () => {
    if (!team) return;
    const n = team.lineup.changes.length;
    if (
      !window.confirm(
        `Apply the optimal lineup to your real ESPN team for week ${team.league.current_week}? ` +
          `(${n} change${n === 1 ? "" : "s"}, +${team.lineup.improvement.toFixed(1)} projected pts)`
      )
    )
      return;
    setBusy(true);
    try {
      const res = await applyEspnLineup(team.league.current_week);
      if (!res.applied) {
        setNotice(res.detail ?? "No changes needed.");
      } else if (res.verified) {
        setNotice(`Lineup updated on ESPN and verified (${res.changes.length} moves).`);
      } else {
        setError(
          "ESPN accepted the request but the roster still shows pending changes — open ESPN and confirm your lineup manually."
        );
      }
      loadTeam();
    } catch (e) {
      setError(String((e as Error).message ?? e));
    } finally {
      setBusy(false);
    }
  };

  const runTransaction = async (rec: WaiverRec, withDrop: boolean) => {
    const label = withDrop
      ? `Add ${rec.add.name} and drop ${rec.drop.name} on your real ESPN team?`
      : `Add ${rec.add.name} to your real ESPN team?`;
    if (!window.confirm(label)) return;
    setBusy(true);
    try {
      await espnTransaction({
        add_player_id: Number(rec.add.player_key),
        drop_player_id: withDrop ? Number(rec.drop.player_key) : undefined,
      });
      setNotice(`Submitted: +${rec.add.name}${withDrop ? ` / -${rec.drop.name}` : ""}`);
      loadTeam();
    } catch (e) {
      setError(String((e as Error).message ?? e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">My ESPN Team</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Link your real ESPN league with the cookies from your own browser — no developer
            app, no approval. Draft night lives in the{" "}
            <Link className="underline" href="/draft-room">
              Draft Room
            </Link>
            .
          </p>
        </div>
        {status?.connected && (
          <Button variant="outline" size="sm" onClick={disconnect}>
            Disconnect
          </Button>
        )}
      </div>

      {error && <Message kind="error">{error}</Message>}
      {notice && <Message kind="ok">{notice}</Message>}

      {status && !status.connected && (
        <Card>
          <CardHeader>
            <CardTitle>Connect your ESPN league</CardTitle>
            <CardDescription>
              Your <strong>League ID</strong> is in the URL when you view your league:
              <code className="mx-1">…/football/team?leagueId=123456</code>.
              <br />
              For a <strong>private</strong> league you also need two cookies. In Chrome, while
              logged into ESPN: F12 → Application → Cookies → <code>espn.com</code>, then copy the
              values of <code>espn_s2</code> and <code>SWID</code> (include the curly braces on
              SWID). Public leagues can leave both blank.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex flex-wrap gap-3">
              <Input
                placeholder="League ID"
                value={leagueId}
                onChange={(e) => setLeagueId(e.target.value)}
                className="max-w-40"
              />
              <Input
                placeholder="Season year"
                value={year}
                onChange={(e) => setYear(e.target.value)}
                className="max-w-32"
              />
            </div>
            <Input
              placeholder="espn_s2 cookie (private leagues)"
              value={s2}
              onChange={(e) => setS2(e.target.value)}
            />
            <Input
              placeholder="SWID cookie, e.g. {AAAA-BBBB-…}"
              value={swid}
              onChange={(e) => setSwid(e.target.value)}
              className="max-w-md"
            />
            <Button disabled={busy || !leagueId || !year} onClick={connect}>
              {busy ? "Connecting…" : "Connect ESPN league"}
            </Button>
          </CardContent>
        </Card>
      )}

      {team && (
        <>
          <div className="grid gap-4 md:grid-cols-3">
            <Card>
              <CardHeader className="pb-2">
                <CardDescription>Team</CardDescription>
                <CardTitle className="text-lg">{team.team.name}</CardTitle>
              </CardHeader>
              <CardContent className="text-sm text-muted-foreground">
                {team.team.wins}-{team.team.losses} · {team.team.points_for.toFixed(1)} PF
                <div>
                  {team.league.name} · {team.league.year} · week {team.league.current_week}
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2">
                <CardDescription>Optimal lineup</CardDescription>
                <CardTitle className="text-lg">
                  {team.lineup.projected_total.toFixed(1)} proj pts
                  {team.lineup.improvement > 0 && (
                    <span className="ml-2 text-sm font-normal text-primary">
                      +{team.lineup.improvement.toFixed(1)}
                    </span>
                  )}
                </CardTitle>
              </CardHeader>
              <CardContent>
                <Button
                  size="sm"
                  disabled={busy || team.lineup.changes.length === 0}
                  onClick={applyLineup}
                >
                  {team.lineup.changes.length === 0
                    ? "Already optimal"
                    : `Apply ${team.lineup.changes.length} change${team.lineup.changes.length === 1 ? "" : "s"}`}
                </Button>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2">
                <CardDescription>Waiver wire</CardDescription>
                <CardTitle className="text-lg">
                  {waivers === null ? "Scanning…" : `${waivers.length} upgrades`}
                </CardTitle>
              </CardHeader>
              <CardContent className="text-sm text-muted-foreground">
                Projections from {team.projection_week.season} wk {team.projection_week.week}
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Recommended lineup</CardTitle>
              <CardDescription>
                Rows tagged &ldquo;move&rdquo; differ from what&apos;s set on ESPN right now.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="w-20">Slot</TableHead>
                    <TableHead>Player</TableHead>
                    <TableHead>Team</TableHead>
                    <TableHead className="text-right">Proj</TableHead>
                    <TableHead>Status</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {[...team.lineup.starters, ...team.lineup.bench].map((e) => (
                    <TableRow key={e.player_key} className={e.slot === "BN" ? "opacity-60" : ""}>
                      <TableCell className="font-medium">{e.slot}</TableCell>
                      <TableCell>
                        {e.name}
                        {e.slot !== e.current_slot && (
                          <Badge variant="secondary" className="ml-2">
                            move {e.current_slot} → {e.slot}
                          </Badge>
                        )}
                      </TableCell>
                      <TableCell className="text-muted-foreground">{e.nfl_team}</TableCell>
                      <TableCell className="text-right">
                        {e.projected_points !== null ? e.projected_points.toFixed(1) : "—"}
                      </TableCell>
                      <TableCell>
                        {e.injury_status && e.injury_status !== "ACTIVE" && (
                          <Badge variant="destructive">{e.injury_status}</Badge>
                        )}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Waiver-wire upgrades</CardTitle>
              <CardDescription>
                Free agents projected to outscore your weakest player at the same position.
              </CardDescription>
            </CardHeader>
            <CardContent>
              {waivers && waivers.length === 0 && (
                <p className="text-sm text-muted-foreground">No clear upgrades right now.</p>
              )}
              {waivers && waivers.length > 0 && (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Add</TableHead>
                      <TableHead className="text-right">Proj</TableHead>
                      <TableHead>Drop</TableHead>
                      <TableHead className="text-right">Gain</TableHead>
                      <TableHead className="text-right">Actions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {waivers.map((rec) => (
                      <TableRow key={rec.add.player_key}>
                        <TableCell className="font-medium">
                          {rec.add.name}{" "}
                          <span className="text-xs text-muted-foreground">
                            {rec.add.position} · {rec.add.nfl_team}
                          </span>
                        </TableCell>
                        <TableCell className="text-right">
                          {rec.add.projected_points.toFixed(1)}
                        </TableCell>
                        <TableCell className="text-muted-foreground">
                          {rec.drop.name}{" "}
                          <span className="text-xs">({rec.drop.projected_points.toFixed(1)})</span>
                        </TableCell>
                        <TableCell className="text-right font-medium text-primary">
                          +{rec.projected_gain.toFixed(1)}
                        </TableCell>
                        <TableCell className="space-x-2 text-right">
                          <Button
                            size="sm"
                            variant="outline"
                            disabled={busy}
                            onClick={() => runTransaction(rec, false)}
                          >
                            Add
                          </Button>
                          <Button size="sm" disabled={busy} onClick={() => runTransaction(rec, true)}>
                            Add / drop
                          </Button>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              )}
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}
