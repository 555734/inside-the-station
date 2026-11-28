extends Node3D
## Underground maze: places every module of the layout (data/app.json) from the shared module meshes
## (data/modules.glb), each with the lightmaps baked for that module, its collision and its
## reflection probes; then the walker and the touch UI (as in the Umeda walkthrough).

const SHADER := preload("res://station.gdshader")
const GLASS := preload("res://station_glass.gdshader")
const FLOORS := ["b1_floor2", "b2s"]
const DRAW_DIST := 90.0             # modules further than this are not drawn (the maze hides them anyway)

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
var probe_data := []
var probe_nodes := {}
var probe_t := 0.0
const PROBE_SLOTS := 32
const PROBES_ON := 24
const PROBES_KEEP := 30


func _ready() -> void:
	_setup_env()
	_setup_ui()
	await get_tree().process_frame
	await get_tree().process_frame
	await _load_maze()


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
		tex_cache[path] = load("res://data/" + path)
	return tex_cache[path]


func _lin2srgb(c: Array) -> Color:
	return Color(c[0], c[1], c[2]).linear_to_srgb()


func _material(name: String, lm: Texture2D) -> ShaderMaterial:
	var d: Dictionary = mats.get(name, {})
	var m := ShaderMaterial.new()
	if d.has("alpha"):
		m.shader = GLASS
		if d.has("color"):
			m.set_shader_parameter("albedo_color", _lin2srgb(d["color"]))
		m.set_shader_parameter("alpha", float(d["alpha"]))
		m.set_shader_parameter("roughness_val", float(d.get("roughness", 0.03)))
		m.set_shader_parameter("lightmap", lm)
		return m
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


func _xform(inst: Dictionary) -> Transform3D:
	# Blender's anticlockwise quarter turns about z are the same turns about Godot's y
	return Transform3D(Basis(Vector3.UP, PI / 2.0 * float(inst["rot"])), _v(inst["pos"]))


func _load_maze() -> void:
	meta = JSON.parse_string(FileAccess.get_file_as_string("res://data/app.json"))
	mats = JSON.parse_string(FileAccess.get_file_as_string("res://data/materials.json"))
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
	status.text = "読み込み中… 部品"
	await get_tree().process_frame
	# module chunks: mesh + its materials (with the chunk's lightmap), shared by every placement
	var lib: Node = load("res://data/modules.glb").instantiate()
	var meshes := []
	_find_meshes(lib, meshes)
	var chunk := {}            # name -> [mesh, [materials]]
	for mi in meshes:
		var cname: String = mi.name
		var lmi: Dictionary = meta["lightmaps"].get(cname, {})
		var lm: Texture2D
		if lmi.is_empty() or not ResourceLoader.exists("res://data/" + lmi["file"]) or OS.get_cmdline_user_args().has("--flatlm"):
			var img := Image.create(4, 4, false, Image.FORMAT_RGB8)
			img.fill(Color(0.6, 0.6, 0.6))
			lm = ImageTexture.create_from_image(img)
		else:
			lm = _tex(lmi["file"])
		var ms := []
		for s in mi.mesh.get_surface_count():
			var sm: Material = mi.mesh.surface_get_material(s)
			ms.append(_material(sm.resource_name if sm != null else "", lm))
		chunk[cname] = [mi.mesh, ms]
	lib.free()
	# goods lit per vertex (one mesh per module that has shops)
	var goods := {}
	if ResourceLoader.exists("res://data/goods.glb"):
		var gl: Node = load("res://data/goods.glb").instantiate()
		var gms := []
		_find_meshes(gl, gms)
		for mi in gms:
			var ms := []
			for s in mi.mesh.get_surface_count():
				var sm: Material = mi.mesh.surface_get_material(s)
				var mat := _material(sm.resource_name if sm != null else "", null)
				mat.set_shader_parameter("vertex_lm", true)
				ms.append(mat)
			goods[String(mi.name)] = [mi.mesh, ms]
		gl.free()
	# collision shapes per module
	var cs: Node = load("res://data/collision.glb").instantiate()
	var cms := []
	_find_meshes(cs, cms)
	var shapes := {}
	for cm in cms:
		var tri: ConcavePolygonShape3D = cm.mesh.create_trimesh_shape()
		tri.backface_collision = true
		shapes[String(cm.name).trim_prefix("COL_")] = tri
	cs.free()
	status.text = "読み込み中… 配置"
	await get_tree().process_frame
	var body := StaticBody3D.new()
	add_child(body)
	var n := 0
	for inst in meta["instances"]:
		var mod: String = inst["module"]
		var xf := _xform(inst)
		for cname in meta["modules"][mod]["chunks"]:
			if not chunk.has(cname):
				continue
			var mi := MeshInstance3D.new()
			mi.mesh = chunk[cname][0]
			for s in chunk[cname][1].size():
				mi.set_surface_override_material(s, chunk[cname][1][s])
			mi.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
			mi.visibility_range_end = DRAW_DIST + 40.0 if mod == "STATION" else DRAW_DIST
			mi.transform = xf
			add_child(mi)
		var gname: String = meta.get("goods", {}).get(mod, "")
		if goods.has(gname):
			var gi := MeshInstance3D.new()
			gi.mesh = goods[gname][0]
			for s in goods[gname][1].size():
				gi.set_surface_override_material(s, goods[gname][1][s])
			gi.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
			gi.visibility_range_end = 40.0
			gi.transform = xf
			add_child(gi)
		if shapes.has(mod):
			var sh := CollisionShape3D.new()
			sh.shape = shapes[mod]
			sh.transform = xf
			body.add_child(sh)
		for p in meta["modules"][mod]["probes"]:
			var c: Vector3 = xf * _v(p["pos"])
			var sz := _v(p["size"])
			if int(inst["rot"]) % 2 == 1:
				sz = Vector3(sz.z, sz.y, sz.x)
			probe_data.append([c, sz, Vector3.ZERO])
		n += 1
		if n % 16 == 0:
			status.text = "読み込み中… %d / %d" % [n, meta["instances"].size()]
			await get_tree().process_frame
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
	var walk := -1.0
	var spawn := 0
	for a in OS.get_cmdline_user_args():
		if a.begins_with("--walk="):
			walk = float(a.substr(7))
		elif a.begins_with("--spawn="):
			spawn = int(a.substr(8))
	if walk > 0.0:
		# physics check: walk straight ahead from a spawn point and log the path
		_go(spawn)
		player.move_vec = Vector2(0, -1)
		var t := 0.0
		while t < walk:
			await get_tree().physics_frame
			t += get_physics_process_delta_time()
			if int(t * 60) % 60 == 0:
				var hit := ""
				for i in player.get_slide_collision_count():
					var c := player.get_slide_collision(i)
					if absf(c.get_normal().y) < 0.7:
						hit = " hit %s n %s" % [c.get_position(), c.get_normal()]
				print("walk t=%.1f pos=%s floor=%s%s" % [t, player.global_position, player.is_on_floor(), hit])
		get_tree().quit()
		return
	if OS.get_cmdline_user_args().has("--stats"):
		for i in meta["spawns"].size():
			_go(i)
			for k in 60:
				await get_tree().process_frame
			var P := Performance
			print("STATS spawn %d objects %d draws %d prims %d | vmem %.0f MB tex %.0f MB" % [i,
				P.get_monitor(P.RENDER_TOTAL_OBJECTS_IN_FRAME), P.get_monitor(P.RENDER_TOTAL_DRAW_CALLS_IN_FRAME),
				P.get_monitor(P.RENDER_TOTAL_PRIMITIVES_IN_FRAME), P.get_monitor(P.RENDER_VIDEO_MEM_USED) / 1048576.0,
				P.get_monitor(P.RENDER_TEXTURE_MEM_USED) / 1048576.0])
		get_tree().quit()
		return
	for a in OS.get_cmdline_user_args():
		if a.begins_with("--shots="):
			# desktop check: "x,y,z,heading,pitch,out.png" per line (Blender coordinates, z = eye)
			spawn_buttons.visible = false
			_go(0)
			for line in FileAccess.get_file_as_string(a.substr(8)).split("\n"):
				var f := line.split(",")
				if f.size() < 6:
					continue
				var pos := Vector3(float(f[0]), float(f[2]) - player.EYE, -float(f[1]))
				player.global_position = pos
				player.yaw = deg_to_rad(float(f[3]) - 90.0)
				player.pitch = deg_to_rad(float(f[4]))
				player.rotation = Vector3(0, player.yaw, 0)
				player.cam.rotation = Vector3(player.pitch, 0, 0)
				player.velocity = Vector3.ZERO
				_pick_probes()
				for k in 6:
					await get_tree().process_frame
					player.global_position = pos
				get_viewport().get_texture().get_image().save_png(f[5].strip_edges())
			get_tree().quit()
			return


func _go(i: int) -> void:
	var s: Dictionary = meta["spawns"][i]
	player.place(_v(s["pos"]), _v(s["fwd"]))
	_pick_probes()


func respawn() -> void:
	_go(0)


func _process(d: float) -> void:
	if fps_label:
		fps_label.text = "%d fps" % Engine.get_frames_per_second()
	probe_t -= d
	if probe_t <= 0.0 and player and not probe_data.is_empty():
		probe_t = 0.25
		_pick_probes()


func _pick_probes() -> void:
	if probe_data.is_empty() or player == null:
		return
	var eye := player.global_position
	var ranked := []
	for i in probe_data.size():
		var c: Vector3 = probe_data[i][0]
		var hs: Vector3 = probe_data[i][1] * 0.5
		var q := eye.clamp(c - hs, c + hs)
		ranked.append([q.distance_squared_to(eye), i])
	ranked.sort_custom(func(a, b): return a[0] < b[0])
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
		rp.box_projection = true
		rp.interior = true
		rp.update_mode = ReflectionProbe.UPDATE_ONCE
		rp.ambient_mode = ReflectionProbe.AMBIENT_DISABLED
		rp.enable_shadows = false
		rp.blend_distance = 1.5
		add_child(rp)
		probe_nodes[i] = rp
