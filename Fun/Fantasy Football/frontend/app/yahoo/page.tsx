"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import {
  applyOptimalLineup,
  disconnectYahoo,
  executeYahooTransaction,
  fetchMyTeam,
  fetchWaivers,
  fetchYahooLeagues,
  fetchYahooLoginUrl,
  fetchYahooStatus,
  submitYahooCode,
  type MyTeamResponse,
  type WaiverRec,
  type YahooLeague,
  type YahooStatus,
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

const LEAGUE_STORAGE_KEY = "yahoo_league_key";

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

export default function YahooPage() {
  const [status, setStatus] = useState<YahooStatus | null>(null);
  const [leagues, setLeagues] = useState<YahooLeague[]>([]);
  const [leagueKey, setLeagueKey] = useState<string | null>(null);
  const [team, setTeam] = useState<MyTeamResponse | null>(null);
  const [waivers, setWaivers] = useState<WaiverRec[] | null>(null);
  const [faabBalance, setFaabBalance] = useState<string | null>(null);
  const [code, setCode] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const loadStatus = useCallback(() => {
    fetchYahooStatus()
      .then(setStatus)
      .catch((e) => setError(String(e.message ?? e)));
  }, []);

  useEffect(() => {
    loadStatus();
    const handle = setTimeout(() => {
      const stored = window.localStorage.getItem(LEAGUE_STORAGE_KEY);
      if (stored) setLeagueKey(stored);
      const params = new URLSearchParams(window.location.search);
      if (params.get("connected")) setNotice("Yahoo account connected.");
      if (params.get("error")) setError(`Yahoo auth failed: ${params.get("error")}`);
    }, 0);
    return () => clearTimeout(handle);
  }, [loadStatus]);

  useEffect(() => {
    if (!status?.connected) return;
    fetchYahooLeagues()
      .then((data) => setLeagues(data.leagues))
      .catch((e) => setError(String(e.message ?? e)));
  }, [status?.connected]);

  const loadTeam = useCallback(() => {
    if (!leagueKey) return;
    fetchMyTeam(leagueKey)
      .then((data) => {
        setTeam(data);
        setError(null);
      })
      .catch((e) => setError(String(e.message ?? e)));
    fetchWaivers(leagueKey)
      .then((data) => {
        setWaivers(data.recommendations);
        setFaabBalance(data.faab_balance);
      })
      .catch(() => setWaivers([]));
  }, [leagueKey]);

  useEffect(loadTeam, [loadTeam]);

  const connect = async () => {
    try {
      const { authorize_url } = await fetchYahooLoginUrl();
      window.location.href = authorize_url;
    } catch (e) {
      setError(String((e as Error).message ?? e));
    }
  };

  const pasteCode = async () => {
    setBusy(true);
    try {
      await submitYahooCode(code);
      setCode("");
      setNotice("Yahoo account connected.");
      loadStatus();
    } catch (e) {
      setError(String((e as Error).message ?? e));
    } finally {
      setBusy(false);
    }
  };

  const pickLeague = (key: string) => {
    window.localStorage.setItem(LEAGUE_STORAGE_KEY, key);
    setLeagueKey(key);
  };

  const applyLineup = async () => {
    if (!leagueKey || !team) return;
    const week = Number(team.league.current_week ?? 1);
    if (
      !window.confirm(
        `Apply the optimal lineup to your real Yahoo team for week ${week}? ` +
          `(${team.lineup.changes.length} change${team.lineup.changes.length === 1 ? "" : "s"}, ` +
          `+${team.lineup.improvement.toFixed(1)} projected pts)`
      )
    )
      return;
    setBusy(true);
    try {
      const res = await applyOptimalLineup(leagueKey, week);
      setNotice(res.applied ? "Lineup updated on Yahoo." : res.detail ?? "No changes needed.");
      loadTeam();
    } catch (e) {
      setError(String((e as Error).message ?? e));
    } finally {
      setBusy(false);
    }
  };

  const runTransaction = async (rec: WaiverRec, withDrop: boolean) => {
    if (!leagueKey) return;
    const faabInput = team?.league.uses_faab
      ? window.prompt(
          `FAAB bid for ${rec.add.name}? (balance: $${faabBalance ?? "?"})`,
          String(rec.suggested_faab)
        )
      : null;
    if (team?.league.uses_faab && faabInput === null) return;
    const label = withDrop
      ? `Add ${rec.add.name} and drop ${rec.drop.name} on your real Yahoo team?`
      : `Add ${rec.add.name} to your real Yahoo team?`;
    if (!window.confirm(label)) return;
    setBusy(true);
    try {
      await executeYahooTransaction(leagueKey, {
        add_player_key: rec.add.player_key,
        drop_player_key: withDrop ? rec.drop.player_key : undefined,
        faab_bid: faabInput ? Number(faabInput) : undefined,
      });
      setNotice(`Transaction submitted: +${rec.add.name}${withDrop ? ` / -${rec.drop.name}` : ""}`);
      loadTeam();
    } catch (e) {
      setError(String((e as Error).message ?? e));
    } finally {
      setBusy(false);
    }
  };

  const disconnect = async () => {
    if (!window.confirm("Disconnect your Yahoo account?")) return;
    await disconnectYahoo();
    setLeagues([]);
    setTeam(null);
    loadStatus();
  };

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">My Yahoo Team</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Link your real Yahoo Fantasy league — the AI manages lineups and waivers with your
            own trained models, and the <Link className="underline" href="/draft-room">Draft Room</Link>{" "}
            guides you live on draft night.
          </p>
        </div>
        {status?.connected && (
          <Button variant="outline" size="sm" onClick={disconnect}>
            Disconnect Yahoo
          </Button>
        )}
      </div>

      {error && <Message kind="error">{error}</Message>}
      {notice && <Message kind="ok">{notice}</Message>}

      {status && !status.has_credentials && (
        <Card>
          <CardHeader>
            <CardTitle>Set up Yahoo API access (one time)</CardTitle>
            <CardDescription>
              1. Go to{" "}
              <a className="underline" href="https://developer.yahoo.com/apps/create/" target="_blank" rel="noreferrer">
                developer.yahoo.com/apps/create
              </a>{" "}
              and create an app: <strong>Confidential Client</strong>, redirect URI{" "}
              <code>https://localhost:8000/yahoo/auth/callback</code> (Yahoo requires https).
              Leave the API-permission checkboxes unchecked — Fantasy Sports access is included
              by default.
              <br />
              2. Put the Client ID and Secret in <code>.env</code> as <code>YAHOO_CLIENT_ID</code> /{" "}
              <code>YAHOO_CLIENT_SECRET</code>, then restart the backend.
            </CardDescription>
          </CardHeader>
        </Card>
      )}

      {status && status.has_credentials && !status.connected && (
        <Card>
          <CardHeader>
            <CardTitle>Connect your Yahoo account</CardTitle>
            <CardDescription>
              You&apos;ll be sent to Yahoo to approve access. Afterwards the browser will land on
              a &ldquo;site can&apos;t be reached&rdquo; page (Yahoo forces an https redirect) —
              that&apos;s expected. Copy the <strong>entire URL</strong> from the address bar and
              paste it below.
            </CardDescription>
          </CardHeader>
          <CardContent className="flex flex-wrap items-center gap-3">
            <Button onClick={connect}>Connect Yahoo</Button>
            <Input
              placeholder="Paste the redirect URL (or just the code)"
              value={code}
              onChange={(e) => setCode(e.target.value)}
              className="max-w-md"
            />
            <Button variant="outline" disabled={!code || busy} onClick={pasteCode}>
              Submit code
            </Button>
          </CardContent>
        </Card>
      )}

      {status?.connected && (
        <Card>
          <CardHeader>
            <CardTitle>Your leagues</CardTitle>
            <CardDescription>Pick the league the AI should manage.</CardDescription>
          </CardHeader>
          <CardContent className="flex flex-wrap gap-2">
            {leagues.length === 0 && (
              <span className="text-sm text-muted-foreground">Loading leagues…</span>
            )}
            {leagues.map((l) => (
              <Button
                key={l.league_key}
                variant={l.league_key === leagueKey ? "default" : "outline"}
                onClick={() => pickLeague(l.league_key)}
              >
                {l.name} · {l.season}
                {l.draft_status === "predraft" && " · pre-draft"}
              </Button>
            ))}
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
                {team.league.name} · {team.league.season} · {team.league.num_teams} teams
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2">
                <CardDescription>Optimal lineup (week {team.league.current_week})</CardDescription>
                <CardTitle className="text-lg">
                  {team.lineup.projected_total.toFixed(1)} proj pts
                  {team.lineup.improvement > 0 && (
                    <span className="ml-2 text-sm font-normal text-primary">
                      +{team.lineup.improvement.toFixed(1)} vs current
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
                    ? "Lineup already optimal"
                    : `Apply ${team.lineup.changes.length} change${team.lineup.changes.length === 1 ? "" : "s"} on Yahoo`}
                </Button>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2">
                <CardDescription>Waivers</CardDescription>
                <CardTitle className="text-lg">
                  {team.league.uses_faab ? `$${faabBalance ?? "—"} FAAB left` : "Priority list"}
                </CardTitle>
              </CardHeader>
              <CardContent className="text-sm text-muted-foreground">
                {waivers === null ? "Scanning the wire…" : `${waivers.length} upgrade${waivers.length === 1 ? "" : "s"} found`}
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Recommended lineup</CardTitle>
              <CardDescription>
                Model projections for {team.projection_week.season} week {team.projection_week.week}.
                Rows marked &ldquo;move&rdquo; differ from your current Yahoo lineup.
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
                      <TableCell className="text-muted-foreground">
                        {e.nfl_team}
                        {e.bye_week && <span className="ml-1 text-xs">(bye {e.bye_week})</span>}
                      </TableCell>
                      <TableCell className="text-right">
                        {e.projected_points !== null ? e.projected_points.toFixed(1) : "—"}
                      </TableCell>
                      <TableCell>
                        {e.injury_status && <Badge variant="destructive">{e.injury_status}</Badge>}
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
                Available players projected to outscore your weakest player at the same position.
              </CardDescription>
            </CardHeader>
            <CardContent>
              {waivers && waivers.length === 0 && (
                <p className="text-sm text-muted-foreground">
                  No clear upgrades on the wire right now.
                </p>
              )}
              {waivers && waivers.length > 0 && (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Add</TableHead>
                      <TableHead className="text-right">Proj</TableHead>
                      <TableHead>Drop</TableHead>
                      <TableHead className="text-right">Proj</TableHead>
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
                          {rec.add.injury_status && (
                            <Badge variant="destructive" className="ml-1">
                              {rec.add.injury_status}
                            </Badge>
                          )}
                        </TableCell>
                        <TableCell className="text-right">{rec.add.projected_points.toFixed(1)}</TableCell>
                        <TableCell className="text-muted-foreground">{rec.drop.name}</TableCell>
                        <TableCell className="text-right text-muted-foreground">
                          {rec.drop.projected_points.toFixed(1)}
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
                          <Button
                            size="sm"
                            disabled={busy || rec.drop.is_undroppable}
                            onClick={() => runTransaction(rec, true)}
                          >
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
