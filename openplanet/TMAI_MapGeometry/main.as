// Writes the loaded map's collision triangles to a file, once per map, so the
// tmai Python process can cast lidar rays against them (src/tmai/track/geometry.py
// reads this format; keep the two in sync).
//
// Read-only: it reads named Openplanet fields, plus two sanity-checked
// Dev::GetOffsetVec3 reads (free-block placement and item pivot) on classes
// Openplanet already gives us. It writes only to its own PluginStorage folder
// and never changes game state.
//
// The plugin does no placement maths. For every placed block and item it
// writes the raw placement (grid coordinate and direction, or position and
// angles) and, for every collision surface under it, the chain of local
// transforms it found on the way (block mobil, prefab entries). Python
// composes them and checks the result against recorded laps.

const uint MAGIC = 0x4F454754;   // "TGEO"
const uint16 VERSION = 2;

const uint8 SRC_BLOCKS = 0;
const uint8 SRC_BAKED = 1;
const uint8 SRC_ITEMS = 2;

const uint8 KIND_GRID_BLOCK = 0;
const uint8 KIND_FREE_BLOCK = 1;
const uint8 KIND_ITEM = 2;

const uint8 STEP_TRANS_QUAT = 1;     // prefab entry: vec3 + quat fields x, y, z, w as named (hold w, x, y, z)
const uint8 STEP_MOBIL_GEOM = 2;     // block mobil: translation + rotation in degrees

const uint16 ID_UNKNOWN = 0xFFFF;
const uint MAX_DEPTH = 10;

[Setting name="Export automatically when a map loads"]
bool S_AutoExport = true;

class Step
{
	uint8 kind;
	array<float> v;
}

class Part
{
	uint shape;
	uint8 flags;          // 1 = from a moving (dyna) shape
	array<Step@> steps;
}

class Shape
{
	CPlugSurface@ surface;
	array<float> verts;   // x, y, z per vertex
	array<uint> tris;     // three vertex indices per triangle
	array<uint16> ids;    // physic id, gameplay id per triangle
	uint approx = 0;      // primitives written as their bounding box
}

// Per-export state
array<Shape@> g_shapes;
dictionary g_buckets;     // cheap key -> array<uint> of shape indices, to dedupe surfaces
dictionary g_unknown;     // model class name -> count, for the summary
dictionary g_counts;      // named problems -> count, for the summary
dictionary g_seen;        // placed blocks already written (Blocks and BakedBlocks overlap)
array<CPlugSurface@> g_emptySurfaces;   // surfaces that flatten to no triangles
uint g_itemVariant = 0;    // variant of the item being walked, for NPlugItem_SVariantList
array<Step@> g_stack;
array<Part@>@ g_parts;
uint8 g_partFlags = 0;

string g_lastUid = "";

// Openplanet stops a script that runs over a second without yielding, and a
// big map takes longer than that. The export yields every few milliseconds
// so the game keeps drawing frames, and gives up if the map changes meanwhile.
const uint SLICE_MS = 15;
uint64 g_sliceStart = 0;
string g_exportUid = "";
bool g_abort = false;

void Breathe()
{
	if (g_abort || Time::Now - g_sliceStart < SLICE_MS) return;
	yield();
	g_sliceStart = Time::Now;
	auto app = cast<CTrackMania>(GetApp());
	if (app is null || app.RootMap is null || MapUid(app.RootMap) != g_exportUid) g_abort = true;
}
bool g_requestExport = false;
string g_status = "idle";

void RenderMenu()
{
	if (UI::MenuItem("TMAI: export map geometry", g_status))
		g_requestExport = true;
}

// ---------------------------------------------------------------- shapes

void PushVert(Shape@ s, const mat4 &in m, const vec3 &in p)
{
	vec4 w = m * p;
	s.verts.InsertLast(w.x);
	s.verts.InsertLast(w.y);
	s.verts.InsertLast(w.z);
}

// Two triangles per box face; corners are numbered by their (x, y, z) bits.
const array<uint> BOX_TRIS = {0,1,3, 0,3,2,  4,6,7, 4,7,5,  0,4,5, 0,5,1,  2,3,7, 2,7,6,  0,2,6, 0,6,4,  1,5,7, 1,7,3};

// Primitives keep their surface ids, so NotCollidable ones are still dropped.
void AddBox(Shape@ s, const mat4 &in m, const vec3 &in c, const vec3 &in h, GmSurf@ surf)
{
	auto prim = cast<GmSurfPrimitive>(surf);
	uint16 pid = prim is null ? ID_UNKNOWN : uint16(int(prim.SurfaceIds_PhysicId));
	uint16 gid = prim is null ? ID_UNKNOWN : uint16(int(prim.SurfaceIds_GameplayId));
	uint base = s.verts.Length / 3;
	for (int i = 0; i < 8; i++)
	{
		vec3 p = vec3((i & 1) != 0 ? h.x : -h.x, (i & 2) != 0 ? h.y : -h.y, (i & 4) != 0 ? h.z : -h.z);
		PushVert(s, m, c + p);
	}
	for (uint i = 0; i < BOX_TRIS.Length; i += 3)
	{
		s.tris.InsertLast(base + BOX_TRIS[i]);
		s.tris.InsertLast(base + BOX_TRIS[i + 1]);
		s.tris.InsertLast(base + BOX_TRIS[i + 2]);
		s.ids.InsertLast(pid);
		s.ids.InsertLast(gid);
	}
	s.approx++;
}

void FlattenSurf(Shape@ s, GmSurf@ surf, const mat4 &in m, uint depth)
{
	if (surf is null || depth > MAX_DEPTH) return;

	auto mesh = cast<GmSurfMesh>(surf);
	if (mesh !is null)
	{
		uint base = s.verts.Length / 3;
		uint nv = mesh.m_Verts.Length;
		for (uint i = 0; i < nv; i++)
		{
			PushVert(s, m, mesh.m_Verts[i]);
			if (i % 1024 == 1023) Breathe();
		}
		for (uint i = 0; i < mesh.m_Tris.Length; i++)
		{
			auto tri = mesh.m_Tris[i];
			if (tri.VertIndices_0_ >= nv || tri.VertIndices_1_ >= nv || tri.VertIndices_2_ >= nv) continue;
			s.tris.InsertLast(base + tri.VertIndices_0_);
			s.tris.InsertLast(base + tri.VertIndices_1_);
			s.tris.InsertLast(base + tri.VertIndices_2_);
			s.ids.InsertLast(uint16(int(tri.SurfaceIds_PhysicId)));
			s.ids.InsertLast(uint16(int(tri.SurfaceIds_GameplayId)));
			if (i % 1024 == 1023) Breathe();
		}
		return;
	}

	auto box = cast<GmSurfBox>(surf);
	if (box !is null)
	{
		AddBox(s, m, box.AABB.m_Center, box.AABB.m_HalfDiag, surf);
		return;
	}

	// Convex hulls are written as their bounding box for now; the summary counts them.
	auto convex = cast<GmSurfConvexPolyhedron>(surf);
	if (convex !is null)
	{
		AddBox(s, m, convex.AABB.m_Center, convex.AABB.m_HalfDiag, surf);
		return;
	}

	// Rounded primitives are written as their bounding box, centred on their origin.
	auto sl = cast<GmSurfSphereLocated>(surf);
	if (sl !is null) { AddBox(s, m, sl.Center, vec3(sl.Radius, sl.Radius, sl.Radius), surf); return; }
	auto sp = cast<GmSurfSphere>(surf);
	if (sp !is null) { AddBox(s, m, vec3(), vec3(sp.Radius, sp.Radius, sp.Radius), surf); return; }
	auto vc = cast<GmSurfVCylinder>(surf);
	if (vc !is null) { AddBox(s, m, vec3(), vec3(vc.Radius, vc.Height * 0.5f, vc.Radius), surf); return; }
	auto cy = cast<GmSurfCylinder>(surf);
	if (cy !is null) { AddBox(s, m, vec3(), vec3(cy.RadiusXZ, cy.RadiusY, cy.RadiusXZ), surf); return; }
	auto ca = cast<GmSurfCapsule>(surf);
	if (ca !is null)
	{
		vec3 half = ca.Dir * (ca.Length * 0.5f);
		vec3 ext = vec3(Math::Abs(half.x), Math::Abs(half.y), Math::Abs(half.z)) + vec3(ca.Radius, ca.Radius, ca.Radius);
		AddBox(s, m, ca.SphereCenter + half, ext, surf);
		return;
	}

	auto compound = cast<GmSurfCompound>(surf);
	if (compound !is null)
	{
		for (uint i = 0; i < compound.Surfs.Length; i++)
		{
			mat4 loc = i < compound.SurfLocs.Length ? mat4(compound.SurfLocs[i]) : mat4::Identity();
			FlattenSurf(s, compound.Surfs[i], m * loc, depth + 1);
		}
		return;
	}

	auto inst = cast<GmSurfCompoundInstance>(surf);
	if (inst !is null && inst.Compound !is null)
	{
		auto c = inst.Compound;
		for (uint i = 0; i < c.Surfs.Length; i++)
		{
			mat4 loc = mat4::Identity();
			if (i < inst.SurfLocs.Length) loc = mat4(inst.SurfLocs[i]);
			else if (i < c.SurfLocs.Length) loc = mat4(c.SurfLocs[i]);
			FlattenSurf(s, c.Surfs[i], m * loc, depth + 1);
		}
		return;
	}

	Count("unsupported surface type " + tostring(int(surf.GmSurfType)));
}

// Index of the shape for this surface, flattening it the first time it is seen.
int ShapeIndex(CPlugSurface@ surface)
{
	if (surface is null) return -1;
	if (surface.m_GmSurf is null) { Count("surface without GmSurf"); return -1; }
	string key = surface.IdName + "|" + tostring(int(surface.m_GmSurf.GmSurfType));
	auto km = cast<GmSurfMesh>(surface.m_GmSurf);
	if (km !is null) key += "|" + tostring(km.m_Verts.Length) + "|" + tostring(km.m_Tris.Length);
	array<uint>@ bucket;
	if (!g_buckets.Get(key, @bucket))
	{
		@bucket = array<uint>();
		g_buckets.Set(key, @bucket);
	}
	for (uint i = 0; i < bucket.Length; i++)
		if (g_shapes[bucket[i]].surface is surface) return int(bucket[i]);

	for (uint i = 0; i < g_emptySurfaces.Length; i++)
		if (g_emptySurfaces[i] is surface) return -1;

	Shape@ s = Shape();
	@s.surface = surface;
	FlattenSurf(s, surface.m_GmSurf, mat4::Identity(), 0);
	if (s.tris.Length == 0)
	{
		g_emptySurfaces.InsertLast(surface);
		return -1;
	}
	bucket.InsertLast(g_shapes.Length);
	g_shapes.InsertLast(s);
	return int(g_shapes.Length - 1);
}

// ---------------------------------------------------------------- model walk

void AddSurface(CPlugSurface@ surface, const string &in owner)
{
	if (surface is null) { Count(owner + " without shape"); return; }
	int idx = ShapeIndex(surface);
	if (idx < 0) return;
	Part@ p = Part();
	p.shape = uint(idx);
	p.flags = g_partFlags;
	for (uint i = 0; i < g_stack.Length; i++) p.steps.InsertLast(g_stack[i]);
	g_parts.InsertLast(p);
}

void Bump(dictionary@ d, const string &in key)
{
	int64 n = 0;
	d.Get(key, n);
	d.Set(key, n + 1);
}

void Count(const string &in what) { Bump(g_counts, what); }

void CountUnknown(CMwNod@ nod)
{
	auto t = Reflection::TypeOf(nod);
	Bump(g_unknown, t is null ? "?" : t.Name);
}

string Summary(dictionary@ d)
{
	string res = "";
	auto keys = d.GetKeys();
	for (uint i = 0; i < keys.Length; i++)
	{
		int64 n = 0;
		d.Get(keys[i], n);
		res += " " + keys[i] + "=" + tostring(n);
	}
	return res;
}

void WalkModel(CMwNod@ model, uint depth)
{
	if (model is null) { Count("prefab entry or item without model"); return; }
	if (depth > MAX_DEPTH) return;

	// Prefab entries and other references often point at a file; its loaded
	// model is the file's Nod. Unloaded files are counted, never loaded.
	auto fid = cast<CSystemFidFile>(model);
	if (fid !is null)
	{
		if (fid.Nod is null) { Count("file not loaded"); return; }
		WalkModel(fid.Nod, depth + 1);
		return;
	}

	auto variants = cast<NPlugItem_SVariantList>(model);
	if (variants !is null)
	{
		uint n = variants.Variants.Length;
		if (n == 0) { Count("empty item variant list"); return; }
		uint v = g_itemVariant < n ? g_itemVariant : 0;
		WalkModel(variants.Variants[v].EntityModel, depth + 1);
		return;
	}

	auto prefab = cast<CPlugPrefab>(model);
	if (prefab !is null)
	{
		for (uint i = 0; i < prefab.Ents.Length; i++)
		{
			auto ent = prefab.Ents[i];
			CMwNod@ entModel = ent.Model;
			if (entModel is null && ent.ModelFid !is null) @entModel = ent.ModelFid;
			Step@ st = Step();
			st.kind = STEP_TRANS_QUAT;
			vec3 t = ent.Location.Trans;
			quat q = ent.Location.Quat;
			array<float> vals = {t.x, t.y, t.z, q.x, q.y, q.z, q.w};
			st.v = vals;
			g_stack.InsertLast(st);
			WalkModel(entModel, depth + 1);
			g_stack.RemoveLast();
		}
		return;
	}

	auto so = cast<CPlugStaticObjectModel>(model);
	if (so !is null) { AddSurface(so.Shape, "static object"); return; }

	auto dyna = cast<CPlugDynaObjectModel>(model);
	if (dyna !is null)
	{
		uint8 saved = g_partFlags;
		if (!dyna.IsStatic) g_partFlags |= 1;
		AddSurface(dyna.StaticShape !is null ? dyna.StaticShape : dyna.DynaShape, "dyna object");
		g_partFlags = saved;
		return;
	}

	auto common = cast<CGameCommonItemEntityModel>(model);
	if (common !is null)
	{
		if (common.StaticObject !is null) WalkModel(common.StaticObject, depth + 1);
		if (common.PhyModel !is null) WalkModel(common.PhyModel, depth + 1);
		return;
	}

	auto phy = cast<CGameObjectPhyModel>(model);
	if (phy !is null) { AddSurface(phy.MoveShapeFid, "phy model"); return; }

	auto surface = cast<CPlugSurface>(model);
	if (surface !is null) { AddSurface(surface, "surface"); return; }

	// Lights, effects, sounds and other models without collision end here.
	CountUnknown(model);
}

// ---------------------------------------------------------------- blocks and items

CGameCtnBlockInfoVariant@ BlockVariant(CGameCtnBlock@ block)
{
	auto info = block.BlockInfo;
	if (info is null) return null;
	uint idx = block.BlockInfoVariantIndex;
	if (block.IsGround)
	{
		if (idx > 0 && idx - 1 < info.AdditionalVariantsGround.Length) return info.AdditionalVariantsGround[idx - 1];
		if (info.VariantGround !is null) return info.VariantGround;
		return info.VariantBaseGround;
	}
	if (idx > 0 && idx - 1 < info.AdditionalVariantsAir.Length) return info.AdditionalVariantsAir[idx - 1];
	if (info.VariantAir !is null) return info.VariantAir;
	return info.VariantBaseAir;
}

uint MobilSlotLength(CGameCtnBlockInfoVariant@ v, uint slot)
{
	switch (slot)
	{
		case 0: return v.Mobils00.Length;
		case 1: return v.Mobils01.Length;
		case 2: return v.Mobils02.Length;
		case 3: return v.Mobils03.Length;
		case 4: return v.Mobils04.Length;
		case 5: return v.Mobils05.Length;
		case 6: return v.Mobils06.Length;
		case 7: return v.Mobils07.Length;
		case 8: return v.Mobils08.Length;
		case 9: return v.Mobils09.Length;
		case 10: return v.Mobils10.Length;
		case 11: return v.Mobils11.Length;
		case 12: return v.Mobils12.Length;
		case 13: return v.Mobils13.Length;
		case 14: return v.Mobils14.Length;
		case 15: return v.Mobils15.Length;
	}
	return 0;
}

CMwNod@ MobilAt(CGameCtnBlockInfoVariant@ v, uint slot, uint i)
{
	if (i >= MobilSlotLength(v, slot)) return null;
	switch (slot)
	{
		case 0: return v.Mobils00[i];
		case 1: return v.Mobils01[i];
		case 2: return v.Mobils02[i];
		case 3: return v.Mobils03[i];
		case 4: return v.Mobils04[i];
		case 5: return v.Mobils05[i];
		case 6: return v.Mobils06[i];
		case 7: return v.Mobils07[i];
		case 8: return v.Mobils08[i];
		case 9: return v.Mobils09[i];
		case 10: return v.Mobils10[i];
		case 11: return v.Mobils11[i];
		case 12: return v.Mobils12[i];
		case 13: return v.Mobils13[i];
		case 14: return v.Mobils14[i];
		case 15: return v.Mobils15[i];
	}
	return null;
}

// The block's selected mobil; slot 0 entry 0 when the selection is out of range.
CGameCtnBlockInfoMobil@ BlockMobil(CGameCtnBlock@ block, CGameCtnBlockInfoVariant@ v)
{
	auto m = cast<CGameCtnBlockInfoMobil>(MobilAt(v, block.MobilVariantIndex, block.MobilIndex));
	if (m is null)
	{
		Count("block mobil fallback to slot 0");
		@m = cast<CGameCtnBlockInfoMobil>(MobilAt(v, 0, 0));
	}
	return m;
}

void CollectMobil(CGameCtnBlockInfoMobil@ mobil, const string &in blockName)
{
	g_stack.RemoveRange(0, g_stack.Length);
	Step@ geom = Step();
	geom.kind = STEP_MOBIL_GEOM;
	vec3 t = mobil.GeomTranslation;
	vec3 r = mobil.GeomRotation;
	array<float> vals = {t.x, t.y, t.z, r.x, r.y, r.z};
	geom.v = vals;
	g_stack.InsertLast(geom);
	// Custom blocks carry their surface directly; it can come with a prefab too.
	if (mobil.PrefabFid !is null) WalkModel(mobil.PrefabFid, 0);
	if (mobil.SurfaceFromBlockItem !is null) AddSurface(mobil.SurfaceFromBlockItem, "block item");
	if (mobil.PrefabFid is null && mobil.SurfaceFromBlockItem is null) Bump(g_unknown, "noprefab:" + blockName);
	g_stack.RemoveRange(0, g_stack.Length);
}

uint16 MemberOffset(const string &in cls, const string &in member)
{
	auto t = Reflection::GetType(cls);
	if (t is null) return 0;
	auto m = t.GetMember(member);
	// 0xFFFF marks a member with no fixed offset; never read through it.
	return (m is null || m.Offset == 0xFFFF || m.Offset >= t.Size) ? 0 : m.Offset;
}

bool Finite(const vec3 &in v)
{
	return !Math::IsNaN(v.x) && !Math::IsNaN(v.y) && !Math::IsNaN(v.z)
		&& !Math::IsInf(v.x) && !Math::IsInf(v.y) && !Math::IsInf(v.z);
}

void WriteStr(MemoryBuffer@ buf, const string &in s)
{
	buf.Write(uint16(s.Length));
	buf.Write(s);
}

void WriteVec3(MemoryBuffer@ buf, const vec3 &in v)
{
	buf.Write(v.x);
	buf.Write(v.y);
	buf.Write(v.z);
}

void WriteIso4(MemoryBuffer@ buf, const iso4 &in m)
{
	buf.Write(m.xx); buf.Write(m.xy); buf.Write(m.xz);
	buf.Write(m.yx); buf.Write(m.yy); buf.Write(m.yz);
	buf.Write(m.zx); buf.Write(m.zy); buf.Write(m.zz);
	buf.Write(m.tx); buf.Write(m.ty); buf.Write(m.tz);
}

void WriteParts(MemoryBuffer@ buf)
{
	buf.Write(uint(g_parts.Length));
	for (uint i = 0; i < g_parts.Length; i++)
	{
		auto p = g_parts[i];
		if (i % 256 == 255) Breathe();
		buf.Write(p.shape);
		buf.Write(p.flags);
		buf.Write(uint8(p.steps.Length));
		for (uint k = 0; k < p.steps.Length; k++)
		{
			buf.Write(p.steps[k].kind);
			buf.Write(uint8(p.steps[k].v.Length));
			for (uint j = 0; j < p.steps[k].v.Length; j++) buf.Write(p.steps[k].v[j]);
		}
	}
}

// Returns the number of instances written.
uint WriteBlocks(MemoryBuffer@ buf, const MwFastBuffer<CGameCtnBlock@> &in blocks, uint8 source)
{
	uint16 dirOff = MemberOffset("CGameCtnBlock", "Dir");
	uint written = 0;
	for (uint i = 0; i < blocks.Length && !g_abort; i++)
	{
		Breathe();
		auto block = blocks[i];
		if (block is null) continue;
		auto variant = BlockVariant(block);
		if (variant is null) { Count("block without variant"); continue; }
		auto mobil = BlockMobil(block, variant);
		if (mobil is null) { Count("block without mobil"); continue; }
		string name = string(block.BlockInfo.Name);

		// Free blocks have no grid coordinate (CoordX reads negative); their
		// position and rotation sit right after Dir. A zero position is a
		// placeholder entry, not a placed block.
		bool free = int(block.CoordX) < 0;
		vec3 pos, ypr;
		if (free)
		{
			if (dirOff == 0) { Count("free block offset unknown"); continue; }
			pos = Dev::GetOffsetVec3(block, dirOff + 0x8);
			ypr = Dev::GetOffsetVec3(block, dirOff + 0x14);
			if (!Finite(pos) || !Finite(ypr) || Math::Abs(pos.x) > 1e5 || Math::Abs(pos.y) > 1e5 || Math::Abs(pos.z) > 1e5)
			{
				Count("free block read failed");
				continue;
			}
			if (pos.LengthSquared() < 1e-6) { Count("placeholder block"); continue; }
		}

		// Blocks and BakedBlocks overlap; write each placed block once.
		// Include ground state, variant and mobil indices so distinct blocks at
		// the same pose survive, and mark seen only after collision parts exist.
		string key = name + "|" + int(block.CoordX) + "|" + int(block.CoordY) + "|" + int(block.CoordZ) + "|" + int(block.BlockDir)
			+ "|" + (block.IsGround ? "1" : "0")
			+ "|" + uint(block.BlockInfoVariantIndex)
			+ "|" + uint(block.MobilVariantIndex)
			+ "|" + uint(block.MobilIndex);
		if (free) key += "|" + Text::Format("%.2f", pos.x) + "|" + Text::Format("%.2f", pos.y) + "|" + Text::Format("%.2f", pos.z)
			+ "|" + Text::Format("%.3f", ypr.x) + "|" + Text::Format("%.3f", ypr.y) + "|" + Text::Format("%.3f", ypr.z);
		if (g_seen.Exists(key)) { Count("duplicate block"); continue; }

		@g_parts = array<Part@>();
		g_partFlags = 0;
		CollectMobil(mobil, name);
		if (g_parts.Length == 0) continue;
		g_seen.Set(key, true);

		buf.Write(source);
		buf.Write(free ? KIND_FREE_BLOCK : KIND_GRID_BLOCK);
		WriteStr(buf, name);
		buf.Write(int(block.CoordX));
		buf.Write(int(block.CoordY));
		buf.Write(int(block.CoordZ));
		buf.Write(uint8(int(block.BlockDir)));
		buf.Write(uint8(block.IsGround ? 1 : 0));
		buf.Write(variant.Size.x);
		buf.Write(variant.Size.y);
		buf.Write(variant.Size.z);
		WriteVec3(buf, pos);
		WriteVec3(buf, ypr);
		buf.Write(uint(block.BlockInfoVariantIndex));
		buf.Write(uint(block.MobilVariantIndex));
		buf.Write(uint(block.MobilIndex));
		WriteParts(buf);
		written++;
	}
	return written;
}

uint WriteItems(MemoryBuffer@ buf, CGameCtnChallenge@ map)
{
	uint16 posOff = MemberOffset("CGameCtnAnchoredObject", "AbsolutePositionInMap");
	uint written = 0;
	for (uint i = 0; i < map.AnchoredObjects.Length && !g_abort; i++)
	{
		Breathe();
		auto item = map.AnchoredObjects[i];
		if (item is null || item.ItemModel is null) continue;

		@g_parts = array<Part@>();
		g_partFlags = 0;
		g_itemVariant = item.IVariant;
		g_stack.RemoveRange(0, g_stack.Length);
		WalkModel(item.ItemModel.EntityModel, 0);
		g_itemVariant = 0;
		if (g_parts.Length == 0) continue;

		vec3 pivot = vec3();
		uint8 pivotOk = 0;
		if (posOff != 0)
		{
			pivot = Dev::GetOffsetVec3(item, posOff + 0xC + 0x24);
			if (Finite(pivot) && Math::Abs(pivot.x) < 1000 && Math::Abs(pivot.y) < 1000 && Math::Abs(pivot.z) < 1000) pivotOk = 1;
			else { pivot = vec3(); Count("item pivot read failed"); }
		}

		buf.Write(SRC_ITEMS);
		buf.Write(KIND_ITEM);
		WriteStr(buf, item.ItemModel.IdName);
		WriteVec3(buf, item.AbsolutePositionInMap);
		buf.Write(item.Yaw);
		buf.Write(item.Pitch);
		buf.Write(item.Roll);
		buf.Write(item.Scale);
		buf.Write(uint8(item.IsLocationInitialised ? 1 : 0));
		WriteIso4(buf, item.BlockLocation);
		WriteIso4(buf, item.LocationInBlock);
		WriteVec3(buf, pivot);
		buf.Write(pivotOk);
		WriteParts(buf);
		written++;
	}
	return written;
}

void WriteShapes(MemoryBuffer@ buf)
{
	buf.Write(uint(g_shapes.Length));
	for (uint i = 0; i < g_shapes.Length; i++)
	{
		auto s = g_shapes[i];
		WriteStr(buf, s.surface.IdName);
		buf.Write(s.approx);
		buf.Write(uint(s.verts.Length / 3));
		for (uint k = 0; k < s.verts.Length; k++)
		{
			buf.Write(s.verts[k]);
			if (k % 4096 == 4095) Breathe();
		}
		buf.Write(uint(s.tris.Length / 3));
		for (uint k = 0; k < s.tris.Length; k++)
		{
			buf.Write(s.tris[k]);
			if (k % 4096 == 4095) Breathe();
		}
		for (uint k = 0; k < s.ids.Length; k++)
		{
			buf.Write(s.ids[k]);
			if (k % 4096 == 4095) Breathe();
		}
	}
}

string MapUid(CGameCtnChallenge@ map)
{
	if (map.MapInfo !is null && map.MapInfo.MapUid != "") return map.MapInfo.MapUid;
	return map.EdChallengeId;
}

void Export(CGameCtnChallenge@ map)
{
	uint64 t0 = Time::Now;
	g_shapes.RemoveRange(0, g_shapes.Length);
	g_buckets.DeleteAll();
	g_unknown.DeleteAll();
	g_counts.DeleteAll();
	g_seen.DeleteAll();
	g_emptySurfaces.RemoveRange(0, g_emptySurfaces.Length);

	string uid = MapUid(map);
	g_exportUid = uid;
	g_abort = false;
	g_sliceStart = Time::Now;
	auto inst = MemoryBuffer(0);
	uint nBlocks = WriteBlocks(inst, map.Blocks, SRC_BLOCKS);
	uint nBaked = WriteBlocks(inst, map.BakedBlocks, SRC_BAKED);
	uint nItems = WriteItems(inst, map);
	if (g_abort)
	{
		print("TMAI geometry: the map changed during the export; it starts again when a map is played");
		g_status = "export interrupted";
		g_lastUid = "";
		g_shapes.RemoveRange(0, g_shapes.Length);
		g_buckets.DeleteAll();
		return;
	}

	auto buf = MemoryBuffer(0);
	buf.Write(MAGIC);
	buf.Write(VERSION);
	WriteStr(buf, uid);
	WriteStr(buf, Text::StripFormatCodes(string(map.MapName)));
	WriteStr(buf, map.DecorationName);
	buf.Write(map.Size.x);
	buf.Write(map.Size.y);
	buf.Write(map.Size.z);
	buf.Write(map.DecoBaseHeightOffset);
	WriteShapes(buf);
	if (g_abort)
	{
		print("TMAI geometry: the map changed during the export; it starts again when a map is played");
		g_status = "export interrupted";
		g_lastUid = "";
		g_shapes.RemoveRange(0, g_shapes.Length);
		g_buckets.DeleteAll();
		return;
	}
	buf.Write(uint(nBlocks + nBaked + nItems));
	inst.Seek(0);
	buf.WriteFromBuffer(inst, inst.GetSize());

	string path = IO::FromStorageFolder(uid + ".tmgeo");
	string tmp = path + ".tmp";
	IO::File f(tmp, IO::FileMode::Write);
	buf.Seek(0);
	f.Write(buf);
	f.Close();
	if (IO::FileExists(path)) IO::Delete(path);
	IO::Move(tmp, path);

	uint tris = 0;
	uint approx = 0;
	for (uint i = 0; i < g_shapes.Length; i++)
	{
		tris += g_shapes[i].tris.Length / 3;
		approx += g_shapes[i].approx;
	}
	string unknown = Summary(g_unknown);
	string problems = Summary(g_counts);
	g_status = tostring(nBlocks + nBaked + nItems) + " placed, " + tostring(tris) + " triangles";
	print("TMAI geometry: wrote " + path);
	print("TMAI geometry: blocks " + nBlocks + ", baked " + nBaked + ", items " + nItems
		+ ", shapes " + g_shapes.Length + ", triangles " + tris
		+ ", boxes " + approx + ", " + (Time::Now - t0) + " ms (spread over frames)");
	if (problems.Length > 0) print("TMAI geometry: skipped:" + problems);
	if (unknown.Length > 0) print("TMAI geometry: models without collision:" + unknown);
	g_shapes.RemoveRange(0, g_shapes.Length);
	g_buckets.DeleteAll();
	g_seen.DeleteAll();
	g_emptySurfaces.RemoveRange(0, g_emptySurfaces.Length);
}

void Main()
{
	while (true)
	{
		yield();
		auto app = cast<CTrackMania>(GetApp());
		if (app is null) continue;
		auto map = app.RootMap;
		// Only while a map is being played, not in menus or the editor.
		if (map is null || app.CurrentPlayground is null || app.Editor !is null)
		{
			if (map is null) g_lastUid = "";
			continue;
		}
		string uid = MapUid(map);
		bool fresh = uid != g_lastUid;
		if (!(g_requestExport || (S_AutoExport && fresh))) continue;
		g_requestExport = false;
		g_lastUid = uid;
		g_status = "exporting...";
		Export(map);
	}
}
