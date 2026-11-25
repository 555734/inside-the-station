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
		var lm := _tex(lmi["file"])
		if OS.get_cmdline_user_args().has("--chunkcol"):
			# desktop check: a flat colour per chunk instead of its lightmap (finds which chunk a
			# surface belongs to); the colours are printed as CHUNKCOL lines
			var h := chunk.hash()
			var img := Image.create(4, 4, false, Image.FORMAT_RGB8)
			img.fill(Color(float(h & 255) / 255.0, float((h >> 8) & 255) / 255.0, float((h >> 16) & 255) / 255.0))
			lm = ImageTexture.create_from_image(img)
			print("CHUNKCOL ", chunk, " ", img.get_pixel(0, 0).to_html(false))
		for s in mi.mesh.get_surface_count():
			var sm: Material = mi.mesh.surface_get_material(s)
			var mname := sm.resource_name if sm != null else ""
			mi.set_surface_override_material(s, _material(mname, lm))
		mi.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
		n += 1
		if n % 8 == 0:
			status.text = "読み込み中… %d / %d" % [n, meshes.size()]
			await get_tree().process_frame
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
	for p in meta["probes"]:
		var rp := ReflectionProbe.new()
		rp.position = _v(p["pos"])
		rp.size = _v(p["size"])
		rp.origin_offset = _v(p["origin"])
		rp.box_projection = true
		rp.interior = true
		rp.update_mode = ReflectionProbe.UPDATE_ONCE
		rp.ambient_mode = ReflectionProbe.AMBIENT_DISABLED
		rp.enable_shadows = false
		rp.blend_distance = 2.0
		add_child(rp)
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
	if shot == "":
		return
	_go(spawn)
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


func respawn() -> void:
	_go(0)


func _process(_d: float) -> void:
	if fps_label:
		fps_label.text = "%d fps" % Engine.get_frames_per_second()
