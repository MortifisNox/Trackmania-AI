// Saves the agent's fastest finished run on each map as a replay, so the run
// report can point at it. Files go to Documents\Trackmania\Replays\TMAI\<map>\
// as <time>ms-<date>.Replay.Gbx. Local files only: no network, and the race
// itself is never changed.
//
// Only saves the agent's runs: while training, tmai keeps ai_driving.txt in
// this plugin's storage folder (OpenplanetNext\PluginStorage\TMAI_Ghost\)
// with the lap times of the agent's recent finishes, one per line. A ghost is
// saved only if its time is in that list, so your own laps and personal-best
// or record ghosts the game loads are never saved here as AI runs.
//
// The game adds a ghost to DataFileMgr.Ghosts when a run finishes; Replay_Save
// writes it as a one-ghost replay (the same calls the public Autosave Ghosts
// plugin uses). Needs Club access, which Developer mode setups have.

const string FOLDER = "TMAI";

string g_mapUid = "";
int g_bestMs = -1;       // fastest replay saved for the current map
uint g_lastCount = 0;
dictionary g_seen;       // ghost ids already decided on
dictionary g_pending;    // ghost id -> first time seen (ms), waiting for tmai's lap list
const uint PENDING_MS = 10000;

string SafeName(const string &in name)
{
	string s = Text::StripFormatCodes(name);
	string bad = "\\/:*?\"<>|";
	for (int i = 0; i < bad.Length; i++)
	{
		s = s.Replace(bad.SubStr(i, 1), "_");
	}
	return s.Trim();
}

// True if tmai reported a finish with this time (to the millisecond).
bool IsAgentLap(int ms)
{
	string path = IO::FromStorageFolder("ai_driving.txt");
	if (!IO::FileExists(path)) return false;
	IO::File f(path, IO::FileMode::Read);
	string text = f.ReadToEnd();
	f.Close();
	auto lines = text.Split("\n");
	for (uint i = 0; i < lines.Length; i++)
	{
		if (Text::ParseInt(lines[i].Trim()) == ms) return true;
	}
	return false;
}

int BestSavedMs(const string &in folder)
{
	int best = -1;
	if (!IO::FolderExists(folder)) return best;
	auto files = IO::IndexFolder(folder, false);
	for (uint i = 0; i < files.Length; i++)
	{
		string name = Path::GetFileName(files[i]);
		int cut = name.IndexOf("ms");
		if (cut <= 0) continue;
		int ms = Text::ParseInt(name.SubStr(0, cut));
		if (ms > 0 && (best < 0 || ms < best)) best = ms;
	}
	return best;
}

void OnMap(CGameCtnChallenge@ map)
{
	g_mapUid = map.MapInfo.MapUid;
	g_seen.DeleteAll();
	g_pending.DeleteAll();
	g_lastCount = 0;
	g_bestMs = BestSavedMs(IO::FromUserGameFolder("Replays/" + FOLDER + "/" + SafeName(map.MapInfo.Name)));
}

void SaveGhost(CGameCtnChallenge@ map, CGameDataFileManagerScript@ dfm, CGameGhostScript@ ghost, int ms)
{
	string folder = FOLDER + "/" + SafeName(map.MapInfo.Name);
	IO::CreateFolder(IO::FromUserGameFolder("Replays/" + folder), true);
	string date = Time::FormatString("%Y%m%d-%H%M%S", Time::Stamp);
	string rel = folder + "/" + ms + "ms-" + date + ".Replay.Gbx";
	dfm.Replay_Save(rel, map, ghost);
	g_bestMs = ms;
	print("TMAI Ghost: saved " + rel);
}

void CheckGhosts(CGameCtnChallenge@ map, CGameDataFileManagerScript@ dfm)
{
	if (dfm.Ghosts.Length == g_lastCount && g_pending.GetSize() == 0) return;
	g_lastCount = dfm.Ghosts.Length;
	for (uint i = 0; i < dfm.Ghosts.Length; i++)
	{
		auto ghost = dfm.Ghosts[i];
		if (ghost is null || g_seen.Exists(ghost.IdName)) continue;
		int ms = int(ghost.Result.Time);
		if (ms <= 0 || (g_bestMs > 0 && ms >= g_bestMs))
		{
			g_seen[ghost.IdName] = true;   // not a new best: nothing to do
			continue;
		}
		if (IsAgentLap(ms))
		{
			g_seen[ghost.IdName] = true;
			g_pending.Delete(ghost.IdName);
			SaveGhost(map, dfm, ghost, ms);
			continue;
		}
		// tmai writes the lap time just after the finish; give it a moment.
		if (!g_pending.Exists(ghost.IdName))
		{
			g_pending[ghost.IdName] = Time::Now;
		}
		else if (Time::Now - uint(g_pending[ghost.IdName]) > PENDING_MS)
		{
			g_seen[ghost.IdName] = true;   // not the agent's run
			g_pending.Delete(ghost.IdName);
		}
	}
}

void Main()
{
	if (!Permissions::CreateLocalReplay())
	{
		warn("TMAI Ghost: this account cannot save local replays (needs Club access)");
		return;
	}
	while (true)
	{
		yield();
		auto app = GetApp();
		auto map = app.RootMap;
		if (map is null || app.Editor !is null) continue;
		if (map.MapInfo.MapUid != g_mapUid) OnMap(map);
		auto cmap = app.Network.ClientManiaAppPlayground;
		if (cmap is null || cmap.DataFileMgr is null) continue;
		CheckGhosts(map, cmap.DataFileMgr);
	}
}
