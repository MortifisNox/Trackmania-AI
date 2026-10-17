// Saves the agent's fastest finished run on each map as a replay, so the run
// report can point at it. Files go to Documents\Trackmania\Replays\TMAI\<map>\
// as <time>ms-<date>.Replay.Gbx. Local files only: no network, and the race
// itself is never changed.
//
// Only saves the agent's runs: while training, tmai keeps ai_driving.txt in
// this plugin's storage folder (OpenplanetNext\PluginStorage\TMAI_Ghost\)
// with the lap times of the agent's recent finishes, one per line. A ghost is
// saved only if its time is within MATCH_MS of a time in that list, so your
// own laps and personal-best or record ghosts the game loads are never saved
// here as AI runs. tmai samples the race clock 20 times a second, so its lap
// time can be a step off the game's exact finish time.
//
// What the plugin last did is written to status.txt next to ai_driving.txt,
// so tmai can say why no replay was saved.
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
const int MATCH_MS = 100;
string g_lastId = "";    // newest ghost seen, in case the game reuses list slots

void Status(const string &in text)
{
	try
	{
		IO::File f(IO::FromStorageFolder("status.txt"), IO::FileMode::Write);
		f.Write(Time::FormatString("%Y-%m-%d %H:%M:%S", Time::Stamp) + " " + text + "\n");
		f.Close();
	}
	catch
	{
	}
}

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

// True if tmai reported a finish within MATCH_MS of this time.
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
		string line = lines[i].Trim();
		if (line.Length > 0 && Math::Abs(Text::ParseInt(line) - ms) <= MATCH_MS) return true;
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
	g_lastId = "";
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
	Status("saved " + rel);
}

void CheckGhosts(CGameCtnChallenge@ map, CGameDataFileManagerScript@ dfm)
{
	uint n = dfm.Ghosts.Length;
	string newest = (n > 0 && dfm.Ghosts[n - 1] !is null) ? dfm.Ghosts[n - 1].IdName : "";
	if (n == g_lastCount && newest == g_lastId && g_pending.GetSize() == 0) return;
	g_lastCount = n;
	g_lastId = newest;
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
			print("TMAI Ghost: skipped a " + ms + " ms ghost (not in tmai's lap list)");
			Status("skipped a " + ms + " ms ghost: no tmai lap within " + MATCH_MS + " ms");
		}
	}
}

void Main()
{
	if (!Permissions::CreateLocalReplay())
	{
		warn("TMAI Ghost: this account cannot save local replays (needs Club access)");
		Status("cannot save local replays: this account needs Club access");
		return;
	}
	print("TMAI Ghost: watching for the agent's finishes");
	Status("running, no replay saved yet");
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
