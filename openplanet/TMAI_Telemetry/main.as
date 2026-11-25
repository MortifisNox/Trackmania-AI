// Streams the local car's state to the tmai Python process, once per frame.
// Localhost only, read-only: this plugin never changes game state.
// Wire format (little-endian) is mirrored in src/tmai/game/protocol.py.

const string HOST = "127.0.0.1";
const uint16 PORT = 9000;
const uint MAGIC = 0x49414D54;  // "TMAI"
const uint16 VERSION = 3;
const uint MAP_UID_SIZE = 64;

const uint16 FLAG_PLAYER = 1;    // player API reachable, fields are valid
const uint16 FLAG_PLAYING = 2;   // UI sequence is Playing (driving)
const uint16 FLAG_FINISHED = 4;  // UI sequence is Finish
const uint16 FLAG_GROUND = 8;    // at least one wheel touches the ground
const uint16 FLAG_FINISH_TIME = 16;  // lastCp is the exact finish result

const int CLOCK_SENTINEL = 0x7fffffff;
const int MAX_RACE_MS = 7 * 24 * 60 * 60 * 1000;
const int FINISH_CLOCK_TOLERANCE_MS = 100;

uint g_seq = 0;
string g_map = "";
int g_start = -1;
int g_cpCount = 0;
int g_lastCp = 0;
int g_finishTime = 0;
bool g_finished = false;
uint g_lastLandmark = uint(-1);
array<uint> g_seenCheckpoints;
array<uint> g_seenLinkedOrders;

void WriteVec3(MemoryBuffer@ buf, const vec3 &in v)
{
	buf.Write(v.x);
	buf.Write(v.y);
	buf.Write(v.z);
}

void WriteZeros(MemoryBuffer@ buf, int count)
{
	for (int i = 0; i < count; i++) buf.Write(float(0));
}

string MapUid(CGameCtnChallenge@ map)
{
	if (map is null) return "";
	if (map.MapInfo !is null && map.MapInfo.MapUid != "") return map.MapInfo.MapUid;
	return map.EdChallengeId;
}

void WriteFixedString(MemoryBuffer@ buf, const string &in s, uint size)
{
	uint n = s.Length < size ? uint(s.Length) : size;
	if (n > 0) buf.Write(s.SubStr(0, n));
	for (uint i = n; i < size; i++) buf.Write(uint8(0));
}

bool PickRaceTime(int apiRaceMs, int start, int netNow, int rulesNow, int &out raceMs)
{
	if (start > 0 && start < CLOCK_SENTINEL)
	{
		if (rulesNow >= 0 && rulesNow < CLOCK_SENTINEL && Math::Abs(rulesNow - start) <= MAX_RACE_MS)
		{
			raceMs = rulesNow - start;
			return true;
		}
		if (netNow >= 0 && netNow < CLOCK_SENTINEL && Math::Abs(netNow - start) <= MAX_RACE_MS)
		{
			raceMs = netNow - start;
			return true;
		}
	}
	if (apiRaceMs >= -MAX_RACE_MS && apiRaceMs <= MAX_RACE_MS)
	{
		raceMs = apiRaceMs;
		return true;
	}
	raceMs = -1;
	return false;
}

void ReadWaypoints(CSmArenaClient@ playground, CSmPlayer@ player, CSmScriptPlayer@ api,
	CSmArenaRulesMode@ rules, int apiRaceMs, int raceMs, bool playing, bool finished, int &out count, int &out lastMs)
{
	string map = playground.Map is null ? "" : playground.Map.IdName;
	int start = int(player.StartTime);
	if (map != g_map || start != g_start || (playing && g_finished))
	{
		g_map = map;
		g_start = start;
		g_cpCount = g_lastCp = g_finishTime = 0;
		g_finished = false;
		g_lastLandmark = uint(-1);
		g_seenCheckpoints.Resize(0);
		g_seenLinkedOrders.Resize(0);
	}

	int apiCount = int(api.RaceWaypointTimes.Length);
	if (playing && raceMs >= 0)
	{
		// Local Time Attack can leave RaceWaypointTimes empty. The controlled
		// player's launched-respawn landmark still changes at checkpoints.
		uint index = player.CurrentLaunchedRespawnLandmarkIndex;
		auto arena = playground.Arena;
		if (arena !is null && index != g_lastLandmark && index < arena.MapLandmarks.Length)
		{
			g_lastLandmark = index;
			auto landmark = arena.MapLandmarks[index];
			if (landmark !is null && landmark.Waypoint !is null && !landmark.Waypoint.IsFinish)
			{
				bool linked = landmark.Tag == "LinkedCheckpoint";
				if (landmark.Waypoint.IsMultiLap)
				{
					g_seenCheckpoints.Resize(0);
					g_seenLinkedOrders.Resize(0);
					// On multilap maps the launched landmark at time zero is the start.
					if (raceMs > 0)
					{
						g_cpCount++;
						g_lastCp = raceMs;
					}
				}
				else if (g_seenCheckpoints.Find(index) < 0 && (!linked || g_seenLinkedOrders.Find(landmark.Order) < 0))
				{
					g_seenCheckpoints.InsertLast(index);
					if (linked) g_seenLinkedOrders.InsertLast(landmark.Order);
					g_cpCount++;
					g_lastCp = raceMs;
				}
			}
		}
		if (apiCount > 0)
		{
			g_cpCount = apiCount;
			g_lastCp = int(api.RaceWaypointTimes[apiCount - 1]);
		}
	}

	if (finished && !g_finished)
	{
		g_finished = true;
		// Read the result of this run, including non-PBs, and release the
		// retrieved ghost immediately.
		if (rules !is null && rules.DataFileMgr !is null)
		{
			try
			{
				auto ghost = rules.Ghost_RetrieveFromPlayer(api);
				if (ghost !is null)
				{
					try
					{
						auto result = ghost.Result;
						if (result !is null)
							g_finishTime = int(result.Time);
					}
					catch
					{
						// Still release the ghost if reading its result fails.
					}
					rules.DataFileMgr.Ghost_Release(ghost.Id);
				}
			}
			catch
			{
				// Some modes or accounts do not expose a ghost result.
			}
		}
		if (g_finishTime <= 0 && apiCount > 0)
		{
			int apiLast = int(api.RaceWaypointTimes[apiCount - 1]);
			if (apiCount > g_cpCount)
				g_finishTime = apiLast;
			else if (apiCount == g_cpCount && apiLast == g_lastCp && g_lastCp > 0)
			{
				// The API may publish the finish waypoint a frame before the UI
				// changes. Match the finish clock, or use the game clock when the
				// API clock is unavailable (e.g. editor test drive).
				bool apiClockAvailable = apiRaceMs > 0 && apiRaceMs <= MAX_RACE_MS;
				if (apiLast == apiRaceMs || (!apiClockAvailable && apiLast <= raceMs
						&& raceMs - apiLast <= FINISH_CLOCK_TOLERANCE_MS))
					g_finishTime = g_lastCp;
			}
		}
		bool finishAlreadyCounted = g_finishTime > 0 && g_finishTime == g_lastCp;
		g_cpCount = Math::Max(g_cpCount + (finishAlreadyCounted ? 0 : 1), apiCount);
	}

	count = g_cpCount;
	// A finish packet's last waypoint is the finish. Zero lets Python
	// estimate its time when Openplanet has no result.
	lastMs = finished ? Math::Max(0, g_finishTime) : g_lastCp;
}

// Fills buf with one packet. Returns false when there is nothing to send yet,
// with the reason in `why` (logged when it changes, so a silent stream can be
// explained from F3 > Openplanet > Log).
bool BuildPacket(MemoryBuffer@ buf, string &out why)
{
	why = "";
	CTrackMania@ app = cast<CTrackMania>(GetApp());
	if (app is null) { why = "no app"; return false; }
	CSmArenaClient@ playground = cast<CSmArenaClient>(app.CurrentPlayground);
	if (playground is null) { why = "no map is being played"; return false; }
	if (playground.GameTerminals.Length == 0) { why = "no game terminal"; return false; }
	auto terminal = playground.GameTerminals[0];
	CSmPlayer@ player = cast<CSmPlayer>(terminal.ControlledPlayer);
	if (player is null) @player = cast<CSmPlayer>(terminal.GUIPlayer);
	if (player is null) { why = "no local player"; return false; }
	CSmScriptPlayer@ api = cast<CSmScriptPlayer>(player.ScriptAPI);
	if (api is null) { why = "player has no script API"; return false; }

	CSceneVehicleVisState@ vis = VehicleState::ViewingPlayerState();

	uint16 flags = FLAG_PLAYER;
	auto seq = terminal.UISequence_Current;
	if (seq == SGamePlaygroundUIConfig::EUISequence::Playing) flags |= FLAG_PLAYING;
	if (seq == SGamePlaygroundUIConfig::EUISequence::Finish) flags |= FLAG_FINISHED;
	if (vis !is null && vis.IsGroundContact) flags |= FLAG_GROUND;

	// Race time sources differ by mode (CurrentRaceTime reads 0 in the editor's
	// test drive), so send the raw clocks too and let Python pick a valid one.
	int netNow = -1;
	if (playground.Interface !is null && playground.Interface.ManialinkScriptHandler !is null)
		netNow = int(playground.Interface.ManialinkScriptHandler.GameTime);
	else if (app.Network !is null && app.Network.PlaygroundClientScriptAPI !is null)
		netNow = int(app.Network.PlaygroundClientScriptAPI.GameTime);
	int rulesNow = -1;
	CSmArenaRulesMode@ rules = cast<CSmArenaRulesMode>(app.PlaygroundScript);
	if (rules !is null) rulesNow = int(rules.Now);

	int start = int(player.StartTime);
	int apiRaceMs = int(api.CurrentRaceTime);
	int raceMs;
	if (!PickRaceTime(apiRaceMs, start, netNow, rulesNow, raceMs))
		flags &= ~FLAG_PLAYING;
	int cpCount, lastCp;
	ReadWaypoints(playground, player, api, rules, apiRaceMs, raceMs, (flags & FLAG_PLAYING) != 0,
		(flags & FLAG_FINISHED) != 0, cpCount, lastCp);
	if ((flags & FLAG_FINISHED) != 0 && lastCp > 0) flags |= FLAG_FINISH_TIME;

	buf.Seek(0, 0);
	buf.Write(MAGIC);
	buf.Write(VERSION);
	buf.Write(flags);
	buf.Write(g_seq++);
	buf.Write(apiRaceMs);
	buf.Write(start);
	buf.Write(netNow);
	buf.Write(rulesNow);
	buf.Write(cpCount);
	buf.Write(lastCp);
	WriteVec3(buf, api.Position);
	buf.Write(api.Speed);
	buf.Write(float(api.EngineCurGear));
	buf.Write(api.EngineRpm);
	buf.Write(api.InputSteer);
	buf.Write(api.InputGasPedal);

	if (vis !is null)
	{
		WriteVec3(buf, vis.WorldVel);
		WriteVec3(buf, vis.Dir);
		WriteVec3(buf, vis.Up);
		buf.Write(vis.FrontSpeed);
		buf.Write(VehicleState::GetSideSpeed(vis));
		buf.Write(vis.InputBrakePedal);
		buf.Write(uint8(vis.FLGroundContactMaterial));
		buf.Write(uint8(vis.FRGroundContactMaterial));
		buf.Write(uint8(vis.RLGroundContactMaterial));
		buf.Write(uint8(vis.RRGroundContactMaterial));
		buf.Write(vis.FLSlipCoef);
		buf.Write(vis.FRSlipCoef);
		buf.Write(vis.RLSlipCoef);
		buf.Write(vis.RRSlipCoef);
	}
	else
	{
		WriteZeros(buf, 12);
		for (int i = 0; i < 4; i++) buf.Write(uint8(0));
		WriteZeros(buf, 4);
	}
	CGameCtnChallenge@ map = app.RootMap;
	if (map is null) @map = playground.Map;
	WriteFixedString(buf, MapUid(map), MAP_UID_SIZE);
	buf.Seek(0, 0);
	return true;
}

void Main()
{
	while (true)
	{
		auto server = Net::Socket();
		if (!server.Listen(HOST, PORT))
		{
			error("TMAI: could not listen on " + HOST + ":" + PORT);
			return;
		}
		while (!server.IsReady()) yield();
		print("TMAI: waiting for tmai on port " + PORT);

		Net::Socket@ client = null;
		while (client is null)
		{
			yield();
			@client = server.Accept();
		}
		print("TMAI: connected");

		MemoryBuffer@ buf = MemoryBuffer(0);
		string lastWhy = "";
		while (true)
		{
			// A closed connection must be noticed even while nothing is sent,
			// or the next tmai would queue behind it forever.
			if (client.IsHungUp())
			{
				print("TMAI: disconnected");
				break;
			}
			string why;
			bool ready = BuildPacket(buf, why);
			if (why != lastWhy)
			{
				print(why.Length > 0 ? "TMAI: not sending: " + why : "TMAI: sending");
				lastWhy = why;
			}
			if (ready && !client.Write(buf))
			{
				print("TMAI: disconnected");
				break;
			}
			yield();
		}
		client.Close();
		server.Close();
	}
}
