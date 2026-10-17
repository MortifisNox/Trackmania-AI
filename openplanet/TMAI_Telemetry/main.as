// Streams the local car's state to the tmai Python process, once per frame.
// Localhost only, read-only: this plugin never changes game state.
// Wire format (little-endian) is mirrored in src/tmai/game/protocol.py.

const string HOST = "127.0.0.1";
const uint16 PORT = 9000;
const uint MAGIC = 0x49414D54;  // "TMAI"
const uint16 VERSION = 2;

const uint16 FLAG_PLAYER = 1;    // player API reachable, fields are valid
const uint16 FLAG_PLAYING = 2;   // UI sequence is Playing (driving)
const uint16 FLAG_FINISHED = 4;  // UI sequence is Finish
const uint16 FLAG_GROUND = 8;    // at least one wheel touches the ground

uint g_seq = 0;

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

// Fills buf with one packet. Returns false when there is nothing to send yet.
bool BuildPacket(MemoryBuffer@ buf)
{
	CTrackMania@ app = cast<CTrackMania>(GetApp());
	if (app is null) return false;
	CSmArenaClient@ playground = cast<CSmArenaClient>(app.CurrentPlayground);
	if (playground is null || playground.GameTerminals.Length == 0) return false;
	CSmArena@ arena = cast<CSmArena>(playground.Arena);
	if (arena is null || arena.Players.Length == 0) return false;
	auto player = arena.Players[0];
	if (player is null) return false;
	CSmScriptPlayer@ api = cast<CSmScriptPlayer>(player.ScriptAPI);
	if (api is null) return false;

	CSceneVehicleVisState@ vis = VehicleState::ViewingPlayerState();

	uint16 flags = FLAG_PLAYER;
	auto seq = playground.GameTerminals[0].UISequence_Current;
	if (seq == SGamePlaygroundUIConfig::EUISequence::Playing) flags |= FLAG_PLAYING;
	if (seq == SGamePlaygroundUIConfig::EUISequence::Finish) flags |= FLAG_FINISHED;
	if (vis !is null && vis.IsGroundContact) flags |= FLAG_GROUND;

	// Race time sources differ by mode (CurrentRaceTime reads 0 in the editor's
	// test drive), so send the raw clocks too and let Python pick a valid one.
	int netNow = -1;
	if (app.Network !is null && app.Network.PlaygroundClientScriptAPI !is null)
		netNow = int(app.Network.PlaygroundClientScriptAPI.GameTime);
	int rulesNow = -1;
	CSmArenaRulesMode@ rules = cast<CSmArenaRulesMode>(app.PlaygroundScript);
	if (rules !is null) rulesNow = int(rules.Now);

	int cpCount = int(api.RaceWaypointTimes.Length);
	int lastCp = cpCount > 0 ? int(api.RaceWaypointTimes[cpCount - 1]) : 0;

	buf.Seek(0, 0);
	buf.Write(MAGIC);
	buf.Write(VERSION);
	buf.Write(flags);
	buf.Write(g_seq++);
	buf.Write(int(api.CurrentRaceTime));
	buf.Write(int(api.StartTime));
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
		while (true)
		{
			if (BuildPacket(buf) && !client.Write(buf))
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
