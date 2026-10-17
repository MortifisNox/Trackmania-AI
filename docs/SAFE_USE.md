# Staying safe: avoiding bans and record problems

There is no published Ubisoft/Nadeo policy that names AI training in
Trackmania. What exists is general anti-cheat language in Ubisoft's Terms of
Use, Openplanet's own rules, and how the game treats Openplanet. The rules
below are built from those. Each one says whether it is sourced or inferred.

## The core idea

Openly-published RL projects such as TMRL have trained agents on Trackmania
2020 with exactly this setup (Openplanet telemetry plus a virtual gamepad)
since 2021 and run a public competition with it
([TMRL](https://github.com/trackmania-rl/tmrl)). I found no reports of bans
from that use *(inferred: absence of reports, not a guarantee)*. The things
that get Trackmania players in trouble are **AI or tool-assisted times showing
up as legitimate records**, cheating online, and modifying the game (for
example running it slowed down, the method behind the 2021 cheating scandal,
[PC Gamer](https://www.pcgamer.com/au/cheating-allegations-catch-up-with-some-of-trackmanias-fastest-drivers)).
So the agent stays offline, off the leaderboards, and does not touch the game's
memory or speed.

## Rules

1. **Train with Openplanet in School or Developer signature mode.**
   School mode "prevents you from playing online and submitting records to
   official leaderboards" (except whitelisted maps), and Developer mode
   implies School mode
   ([Openplanet: School mode](https://openplanet.dev/docs/school-mode),
   [Signature modes](https://openplanet.dev/docs/tutorials/signature-modes)).
   *(Sourced.)* This is the main safety net: even a mistake cannot push an AI
   time to the official leaderboards.

2. **Train on local map files, not on official campaign, Track of the Day or
   club leaderboard maps.** Use maps from your own `My Maps` folder or
   downloaded map files played through Create > Play a map. *(Inferred: these
   have no official leaderboard; School mode already blocks submission.)*

3. **Never run the agent online**: no servers, rooms, Ranked/matchmaking,
   Cup of the Day, Royal or any competition. Ubisoft's Terms of Use prohibit
   cheats, bots and unauthorised automation in online play *(inferred: I could
   not fetch the Terms text, legal.ubisoft.com blocks automated reading; this
   is standard in Ubisoft's codes of conduct)*. Competition servers can also
   detect and kick Openplanet use by signature mode
   ([Trackmania competition plugin settings](https://doc.trackmania.com/club/competition-tool/plugin-settings/)).
   *(Sourced.)*

4. **Never upload or present AI runs as human.** Do not submit AI replays to
   leaderboards, Trackmania Exchange records or speedrun sites, and label any
   video or ghost you share as AI-driven. *(Inferred from how the community
   treats tool-assisted runs.)*

5. **Do not modify the game.** No game-speed changes, memory writing, DLL
   injection beyond Openplanet itself, or edited game files. The agent only
   *reads* telemetry through Openplanet's plugin API and *presses* a virtual
   controller, like a person with a gamepad. *(Inferred; speed hacks are the
   documented cheating method above.)*

6. **Do not hammer Nadeo's web services.** Our code never calls Nadeo's
   leaderboard or account APIs. Openplanet will not approve plugins that
   request full leaderboards because of the request volume
   ([Openplanet plugin terms](https://openplanet.dev/docs/plugin-tos)).
   *(Sourced.)*

7. **Keep our telemetry plugin private.** Openplanet's plugin terms do not
   allow mostly AI-generated plugins on openplanet.dev
   ([plugin terms](https://openplanet.dev/docs/plugin-tos)), and the plugin is
   written with AI help. It runs locally in Developer mode and never gets
   uploaded there. *(Sourced.)*

8. **Leave School/Developer mode before playing normally.** Switch the
   signature mode back to **Regular** when you want to play online yourself.
   Openplanet kicks you out of the current game when you leave School mode;
   that is expected ([School mode](https://openplanet.dev/docs/school-mode)).
   *(Sourced.)*

9. **Keep Openplanet updated** after game updates. An outdated Openplanet can
   fail to load or crash the game. *(Inferred; this is a stability issue, not a
   ban risk.)*

## What the code enforces

- The dashboard only listens on `127.0.0.1` (config validation rejects other
  hosts).
- The telemetry socket is localhost only.
- No code path talks to Nadeo or Ubisoft services.
- Planned for phase 1: the trainer refuses to start unless our plugin reports
  Developer mode, if Openplanet's plugin API exposes that (to be verified).
