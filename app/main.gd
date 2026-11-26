extends Node3D
## Loads the exported station (station/), gives every surface its baked lightmap material,
## adds collision, reflection probes, the walker and the touch UI.

const SHADER := preload("res://station.gdshader")
const FLOORS := ["b2s", "b2bands", "b1_floor2"]

var meta := {}
var mats := {}
var tex_cache := {}
var wear: NoiseTexture2D
var player: CharacterBody3D
var status: Label
var hud: Control
var spawn_buttons: VBoxContainer
var fps_label: Label
var env: Environment
## exposure underground and out in the street (the eye adapts on the way up the exit stairs)
const EXPOSURE_IN := 0.55
const EXPOSURE_OUT := 0.32
const Z_STREET := 5.6
var probe_data := []                 # every probe of the station: [position, size, origin offset]
var probe_nodes := {}                # index -> ReflectionProbe, only for the nearest ones
var probe_t := 0.0
## The reflection atlas holds this many probes (project setting reflection_count). A probe keeps its
## slot while it exists, hidden or not, and running out of slots crashes the renderer, so only the
## nearest probes exist at any time (made when they come near, freed when they fall behind).
const PROBE_SLOTS := 32
const PROBES_ON := 24
const PROBES_KEEP := 30
## Lightmap streaming. Every chunk always has its 1/8 size stand-in; the full lightmap is in memory
## only where it can be on screen: underground all of the station (B1, B2, stairs) and the street
## chunks within NEAR_LOAD m (the exit stairs and what is seen up them); up the exit stairs and in
## the street all of the street and the station chunks within NEAR_LOAD m (seen down the stairs),
## never the platform level. Full lightmaps load on a thread before they come into view.
var chunk_info := {}                 # chunk -> {mats, aabb, level, hi, lo, state 0 stand-in / 1 loading / 2 full}
var stream_t := 0.0
var street_mode := false
const NEAR_LOAD := 70.0
const NEAR_DROP := 85.0
const STREET_IN := 2.5               # feet height (m) above which the walker counts as in the street
const STREET_OUT := 1.5


func _ready() -> void:
	_setup_env()
	_setup_ui()
	await get_tree().process_frame
	await get_tree().process_frame
	await _load_station()


func _setup_env() -> void:
	var we := WorldEnvironment.new()
	var e := Environment.new()
	e.background_mode = Environment.BG_COLOR
	e.background_color = Color.BLACK
	e.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	e.ambient_light_color = Color.BLACK
	e.ambient_light_energy = 0.0
	e.reflected_light_source = Environment.REFLECTION_SOURCE_DISABLED
	e.tonemap_mode = Environment.TONE_MAPPER_AGX
	e.tonemap_exposure = 0.55
	e.glow_enabled = true
	e.glow_intensity = 0.5
	e.glow_strength = 0.9
	e.glow_bloom = 0.02
	e.glow_hdr_threshold = 1.1
	e.glow_blend_mode = Environment.GLOW_BLEND_MODE_ADDITIVE
	we.environment = e
	env = e
	add_child(we)


func _setup_ui() -> void:
	var layer := CanvasLayer.new()
	add_child(layer)
	var font := load("res://ui/NotoSansJP_500Medium.ttf")
	var theme := Theme.new()
	theme.default_font = font
	theme.default_font_size = 26
	hud = preload("res://hud.gd").new()
	hud.set_anchors_preset(Control.PRESET_FULL_RECT)
	hud.mouse_filter = Control.MOUSE_FILTER_IGNORE
	hud.theme = theme
	layer.add_child(hud)
	status = Label.new()
	status.text = "読み込み中…"
	status.set_anchors_preset(Control.PRESET_CENTER)
	status.add_theme_font_size_override("font_size", 40)
	hud.add_child(status)
	fps_label = Label.new()
	fps_label.position = Vector2(16, 10)
	fps_label.add_theme_font_size_override("font_size", 18)
	fps_label.modulate = Color(1, 1, 1, 0.6)
	hud.add_child(fps_label)
	spawn_buttons = VBoxContainer.new()
	spawn_buttons.set_anchors_preset(Control.PRESET_TOP_RIGHT)
	spawn_buttons.position = Vector2(-300, 16)
	spawn_buttons.custom_minimum_size = Vector2(280, 0)
	spawn_buttons.add_theme_constant_override("separation", 8)
	hud.add_child(spawn_buttons)


func _tex(path: String) -> Texture2D:
	if not tex_cache.has(path):
		tex_cache[path] = load("res://station/" + path)
	return tex_cache[path]


func _lin2srgb(c: Array) -> Color:
	return Color(c[0], c[1], c[2]).linear_to_srgb()


func _material(name: String, lm: Texture2D) -> ShaderMaterial:
	var d: Dictionary = mats.get(name, {})
	var m := ShaderMaterial.new()
	m.shader = SHADER
	if d.has("albedo"):
		m.set_shader_parameter("albedo_tex", _tex(d["albedo"]))
		m.set_shader_parameter("use_albedo_tex", true)
	elif d.has("color"):
		m.set_shader_parameter("albedo_color", _lin2srgb(d["color"]))
	if d.has("uv_scale"):
		m.set_shader_parameter("uv_scale", Vector2(d["uv_scale"][0], d["uv_scale"][1]))
	if d.has("rough"):
		m.set_shader_parameter("rough_tex", _tex(d["rough"]))
		m.set_shader_parameter("use_rough_tex", true)
	m.set_shader_parameter("roughness_val", float(d.get("roughness", 0.5)))
	if d.has("normal"):
		m.set_shader_parameter("normal_tex", _tex(d["normal"]))
		m.set_shader_parameter("normal_mode", 2 if d.get("normal_space", "TANGENT") == "OBJECT" else 1)
	m.set_shader_parameter("metallic_val", float(d.get("metallic", 0.0)))
	m.set_shader_parameter("coat", float(d.get("coat", 0.0)))
	var es := float(d.get("emission_strength", 0.0))
	if es > 0.0:
		m.set_shader_parameter("emission_strength", es)
		if d.has("emission_tex"):
			m.set_shader_parameter("emission_tex", _tex(d["emission_tex"]))
			m.set_shader_parameter("use_emission_tex", true)
		else:
			m.set_shader_parameter("emission_color", _lin2srgb(d.get("emission", [1, 1, 1])))
	m.set_shader_parameter("lightmap", lm)
	if name in FLOORS:
		m.set_shader_parameter("wear_tex", wear)
		m.set_shader_parameter("wear_rough", 0.10)
		m.set_shader_parameter("wear_albedo", 0.10)
	return m


func _find_meshes(n: Node, out: Array) -> void:
	if n is MeshInstance3D:
		out.append(n)
	for c in n.get_children():
		_find_meshes(c, out)


func _v(a: Array) -> Vector3:
	return Vector3(a[0], a[1], a[2])


func _load_station() -> void:
	meta = JSON.parse_string(FileAccess.get_file_as_string("res://station/app.json"))
	mats = JSON.parse_string(FileAccess.get_file_as_string("res://station/materials.json"))
	wear = NoiseTexture2D.new()
	wear.width = 512
	wear.height = 512
	wear.seamless = true
	wear.generate_mipmaps = true
	var fn := FastNoiseLite.new()
	fn.frequency = 0.012
	fn.fractal_octaves = 5
	wear.noise = fn
	await wear.changed
	status.text = "読み込み中… 駅の形"
	await get_tree().process_frame
	var st: Node = load("res://station/station.glb").instantiate()
	add_child(st)
	var meshes := []
	_find_meshes(st, meshes)
	var n := 0
	for mi in meshes:
		var chunk: String = mi.name
		var lmi: Dictionary = meta["lightmaps"].get(chunk, {})
		if lmi.is_empty():
			push_warning("no lightmap for " + chunk)
			continue
		var streamed: bool = lmi.has("lo") and not OS.get_cmdline_user_args().has("--chunkcol")
		var lm: Texture2D = load("res://station/" + lmi["lo"]) if streamed else _tex(lmi["file"])
		if OS.get_cmdline_user_args().has("--chunkcol"):
			# desktop check: a flat colour per chunk instead of its lightmap (finds which chunk a
			# surface belongs to); the colours are printed as CHUNKCOL lines
			var h := chunk.hash()
			var img := Image.create(4, 4, false, Image.FORMAT_RGB8)
			img.fill(Color(float(h & 255) / 255.0, float((h >> 8) & 255) / 255.0, float((h >> 16) & 255) / 255.0))
			lm = ImageTexture.create_from_image(img)
			print("CHUNKCOL ", chunk, " ", img.get_pixel(0, 0).to_html(false))
		var cm := []
		for s in mi.mesh.get_surface_count():
			var sm: Material = mi.mesh.surface_get_material(s)
			var mname := sm.resource_name if sm != null else ""
			var mat := _material(mname, lm)
			mi.set_surface_override_material(s, mat)
			cm.append(mat)
		mi.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
		if streamed:
			chunk_info[chunk] = {"mats": cm, "aabb": mi.global_transform * mi.get_aabb(),
				"level": chunk.get_slice("_", 0), "hi": "res://station/" + lmi["file"], "lo": lm, "state": 0}
		n += 1
		if n % 8 == 0:
			status.text = "読み込み中… %d / %d" % [n, meshes.size()]
			await get_tree().process_frame
	# the avenue beyond the walkable stretch (not baked): lit like the baked frontages near by
	if ResourceLoader.exists("res://station/far.glb"):
		var far: Node = load("res://station/far.glb").instantiate()
		add_child(far)
		var fe: float = float(meta.get("far_irradiance", 3.0))
		var img := Image.create(1, 1, false, Image.FORMAT_RGB8)
		var enc := sqrt(fe / (fe + 1.0))
		img.fill(Color(enc, enc, enc))
		var flm := ImageTexture.create_from_image(img)
		var fms := []
		_find_meshes(far, fms)
		for mi in fms:
			for s_ in mi.mesh.get_surface_count():
				var sm: Material = mi.mesh.surface_get_material(s_)
				mi.set_surface_override_material(s_, _material(sm.resource_name if sm != null else "", flm))
			mi.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	if ResourceLoader.exists("res://station/sky.png"):
		var sky := Sky.new()
		var pm := PanoramaSkyMaterial.new()
		pm.panorama = load("res://station/sky.png")
		sky.sky_material = pm
		env.sky = sky
		env.background_mode = Environment.BG_SKY
	# occlusion culling: chunks hidden behind big walls, floors and ceilings are not drawn
	if FileAccess.file_exists("res://station/occluders.bin"):
		var raw := FileAccess.get_file_as_bytes("res://station/occluders.bin").to_float32_array()
		var pts := PackedVector3Array()
		pts.resize(raw.size() / 3)
		for i in pts.size():
			pts[i] = Vector3(raw[3 * i], raw[3 * i + 1], raw[3 * i + 2])
		var idx := PackedInt32Array()
		idx.resize(pts.size())
		for i in idx.size():
			idx[i] = i
		var occ := ArrayOccluder3D.new()
		occ.set_arrays(pts, idx)
		var oi := OccluderInstance3D.new()
		oi.occluder = occ
		add_child(oi)
	# collision: walls, floors, ramps over stairs and escalators
	var cs: Node = load("res://station/collision.glb").instantiate()
	var cms := []
	_find_meshes(cs, cms)
	var body := StaticBody3D.new()
	for cm in cms:
		var shape := CollisionShape3D.new()
		var tri: ConcavePolygonShape3D = cm.mesh.create_trimesh_shape()
		tri.backface_collision = true
		shape.shape = tri
		body.add_child(shape)
	add_child(body)
	cs.free()
	status.text = "読み込み中… 反射"
	await get_tree().process_frame
	for p in ([] if OS.get_cmdline_user_args().has("--noprobes") else meta["probes"]):
		probe_data.append([_v(p["pos"]), _v(p["size"]), _v(p["origin"])])
	player = preload("res://player.gd").new()
	add_child(player)
	hud.player = player
	for i in meta["spawns"].size():
		var b := Button.new()
		b.text = meta["spawns"][i]["name"]
		b.custom_minimum_size = Vector2(280, 60)
		b.pressed.connect(_go.bind(i))
		spawn_buttons.add_child(b)
	var run_b := Button.new()
	run_b.text = "歩く / 走る"
	run_b.toggle_mode = true
	run_b.custom_minimum_size = Vector2(280, 60)
	run_b.toggled.connect(func(on): player.run = on)
	spawn_buttons.add_child(run_b)
	player.ui_box = spawn_buttons
	_go(0)
	status.text = ""
	_maybe_screenshot()


func _maybe_screenshot() -> void:
	# desktop check: godot -- --shot=/path/out.png --spawn=0 [--yawoff=deg] [--pitch=deg]
	var shot := ""
	var spawn := 0
	for a in OS.get_cmdline_user_args():
		if a.begins_with("--shot="):
			shot = a.substr(7)
		elif a.begins_with("--spawn="):
			spawn = int(a.substr(8))
	var walk := -1.0
	for a in OS.get_cmdline_user_args():
		if a.begins_with("--walk="):
			walk = float(a.substr(7))
	if walk > 0.0:
		# physics check: walk straight ahead from the spawn point and log the path
		_go(spawn)
		player.move_vec = Vector2(0, -1)
		var t := 0.0
		while t < walk:
			await get_tree().physics_frame
			t += get_physics_process_delta_time()
			if int(t * 60) % 60 == 0:
				print("walk t=%.1f pos=%s floor=%s" % [t, player.global_position, player.is_on_floor()])
		get_tree().quit()
		return
	if OS.get_cmdline_user_args().has("--stats"):
		# desktop check: what each spawn view costs (objects, draw calls, memory)
		for i in meta["spawns"].size():
			_go(i)
			for k in 60:
				await get_tree().process_frame
			var P := Performance
			print("STATS spawn %d objects %d draws %d prims %d | vmem %.0f MB tex %.0f MB buf %.0f MB | static %.0f MB" % [i,
				P.get_monitor(P.RENDER_TOTAL_OBJECTS_IN_FRAME), P.get_monitor(P.RENDER_TOTAL_DRAW_CALLS_IN_FRAME),
				P.get_monitor(P.RENDER_TOTAL_PRIMITIVES_IN_FRAME), P.get_monitor(P.RENDER_VIDEO_MEM_USED) / 1048576.0,
				P.get_monitor(P.RENDER_TEXTURE_MEM_USED) / 1048576.0, P.get_monitor(P.RENDER_BUFFER_MEM_USED) / 1048576.0,
				P.get_monitor(P.MEMORY_STATIC) / 1048576.0])
		get_tree().quit()
		return
	if OS.get_cmdline_user_args().has("--bench"):
		# desktop check: average frame time at each spawn (relative numbers only)
		for i in meta["spawns"].size():
			_go(i)
			for k in 30:
				await get_tree().process_frame
			var t0 := Time.get_ticks_usec()
			for k in 40:
				await get_tree().process_frame
			print("BENCH spawn %d %.1f ms/frame" % [i, (Time.get_ticks_usec() - t0) / 40000.0])
		get_tree().quit()
		return
	for a in OS.get_cmdline_user_args():
		if a.begins_with("--shots="):
			# desktop check: many views in one run; each line "x,y,z,heading,pitch,out.png" (Blender coords)
			spawn_buttons.visible = false
			_go(0)
			for line in FileAccess.get_file_as_string(a.substr(8)).split("\n"):
				var f := line.split(",")
				if f.size() < 6:
					continue
				player.global_position = Vector3(float(f[0]), float(f[2]), -float(f[1]))
				player.yaw = deg_to_rad(float(f[3]) - 90.0)
				player.pitch = deg_to_rad(float(f[4]))
				player.rotation = Vector3(0, player.yaw, 0)
				player.cam.rotation = Vector3(player.pitch, 0, 0)
				player.velocity = Vector3.ZERO
				_stream(true)
				_pick_probes()
				for k in 6:
					await get_tree().process_frame
					player.global_position = Vector3(float(f[0]), float(f[2]), -float(f[1]))
				get_viewport().get_texture().get_image().save_png(f[5].strip_edges())
			get_tree().quit()
			return
	if shot == "":
		return
	_go(spawn)
	for a in OS.get_cmdline_user_args():
		if a.begins_with("--at="):
			# --at=x,y,z,heading in Blender coordinates (metres, heading in degrees from +x)
			var v := a.substr(5).split_floats(",")
			player.global_position = Vector3(v[0], v[2], -v[1])
			player.yaw = deg_to_rad(v[3] - 90.0)
	for a in OS.get_cmdline_user_args():
		if a.begins_with("--yawoff="):
			player.yaw += deg_to_rad(float(a.substr(9)))
		elif a.begins_with("--pitch="):
			player.pitch = deg_to_rad(float(a.substr(8)))
	player.rotation = Vector3(0, player.yaw, 0)
	player.cam.rotation = Vector3(player.pitch, 0, 0)
	spawn_buttons.visible = false
	for i in 90:
		await get_tree().process_frame
	get_viewport().get_texture().get_image().save_png(shot)
	print("saved ", shot)
	get_tree().quit()


func _go(i: int) -> void:
	var s: Dictionary = meta["spawns"][i]
	player.place(_v(s["pos"]), _v(s["fwd"]))
	_stream(true)
	_pick_probes()


func _hdist(b: AABB, p: Vector3) -> float:
	var q := p.clamp(b.position, b.end)
	return Vector2(q.x - p.x, q.z - p.z).length()


func _want_full(c: Dictionary, eye: Vector3) -> bool:
	var near := _hdist(c["aabb"], eye) < (NEAR_LOAD if c["state"] == 0 else NEAR_DROP)
	var lvl: String = c["level"]
	if street_mode:
		if lvl == "UP":
			return true
		return near and lvl != "B2"
	return near if lvl == "UP" else true


func _set_lm(c: Dictionary, t: Texture2D) -> void:
	for m in c["mats"]:
		m.set_shader_parameter("lightmap", t)


func _stream(now := false) -> void:
	# now: everything wanted is loaded before returning (start, teleports, desktop shots)
	if player == null or chunk_info.is_empty():
		return
	var eye := player.global_position
	if street_mode and eye.y < STREET_OUT:
		street_mode = false
	elif not street_mode and eye.y > STREET_IN:
		street_mode = true
	for name in chunk_info:
		var c: Dictionary = chunk_info[name]
		var want := _want_full(c, eye)
		if c["state"] == 1:
			var st := ResourceLoader.load_threaded_get_status(c["hi"])
			if st == ResourceLoader.THREAD_LOAD_LOADED or (now and st == ResourceLoader.THREAD_LOAD_IN_PROGRESS):
				var t: Texture2D = ResourceLoader.load_threaded_get(c["hi"])
				if want and t != null:
					_set_lm(c, t)
					c["state"] = 2
				else:
					c["state"] = 0
			elif st != ResourceLoader.THREAD_LOAD_IN_PROGRESS:
				c["state"] = 0
		elif want and c["state"] == 0:
			if now:
				var t2: Texture2D = load(c["hi"])
				if t2 != null:
					_set_lm(c, t2)
					c["state"] = 2
			elif ResourceLoader.load_threaded_request(c["hi"]) == OK:
				c["state"] = 1
		elif not want and c["state"] == 2:
			_set_lm(c, c["lo"])
			c["state"] = 0


func respawn() -> void:
	_go(0)


func _process(d: float) -> void:
	if fps_label:
		fps_label.text = "%d fps" % Engine.get_frames_per_second()
	if player and env:
		# eye adaptation: from the concourse exposure at the foot of an exit stair to the street one
		var t := clampf((player.global_position.y - 1.5) / (Z_STREET - 1.5), 0.0, 1.0)
		var target := lerpf(EXPOSURE_IN, EXPOSURE_OUT, t * t * (3.0 - 2.0 * t))
		env.tonemap_exposure = lerpf(env.tonemap_exposure, target, clampf(d * 1.5, 0.0, 1.0))
	stream_t -= d
	if stream_t <= 0.0:
		stream_t = 0.3
		_stream()
	probe_t -= d
	if probe_t <= 0.0 and player and not probe_data.is_empty():
		probe_t = 0.25
		_pick_probes()


func _pick_probes() -> void:
	# the probes whose boxes are nearest to the walker (0 inside the box) exist, the rest do not
	if probe_data.is_empty():
		return
	var eye := player.global_position
	var ranked := []
	for i in probe_data.size():
		var c: Vector3 = probe_data[i][0]
		var hs: Vector3 = probe_data[i][1] * 0.5
		var q := eye.clamp(c - hs, c + hs)
		ranked.append([q.distance_squared_to(eye), i])
	ranked.sort_custom(func(a, b): return a[0] < b[0])
	# a probe that exists stays while it is among the nearest PROBES_KEEP (each new probe renders
	# the station six times, so walking along does not keep making and dropping the same ones);
	# new ones are made for the nearest PROBES_ON. At most PROBES_KEEP exist (< PROBE_SLOTS).
	var keep := {}
	for j in mini(PROBES_KEEP, ranked.size()):
		keep[ranked[j][1]] = true
	var want := {}
	for j in mini(PROBES_ON, ranked.size()):
		want[ranked[j][1]] = true
	for i in probe_nodes.keys():
		if not keep.has(i):
			probe_nodes[i].queue_free()
			probe_nodes.erase(i)
	for i in want.keys():
		if probe_nodes.has(i):
			continue
		var rp := ReflectionProbe.new()
		rp.position = probe_data[i][0]
		rp.size = probe_data[i][1]
		rp.origin_offset = probe_data[i][2]
		rp.box_projection = true
		rp.interior = true
		rp.update_mode = ReflectionProbe.UPDATE_ONCE
		rp.ambient_mode = ReflectionProbe.AMBIENT_DISABLED
		rp.enable_shadows = false
		rp.blend_distance = 2.0
		add_child(rp)
		probe_nodes[i] = rp
