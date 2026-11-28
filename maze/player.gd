extends CharacterBody3D
## First-person walker. Touch: left half of the screen is a floating joystick, right half drags
## the view. Desktop: WASD / arrows to walk, drag with the mouse (or capture with click) to look.

const WALK := 1.5
const RUN := 3.2
const EYE := 1.55

var cam: Camera3D
var yaw := 0.0
var pitch := 0.0
var move_vec := Vector2.ZERO          # from the joystick, -1..1
var stick_id := -1
var stick_origin := Vector2.ZERO
var stick_pos := Vector2.ZERO
var look_id := -1
var look_last := Vector2.ZERO
var run := false
var gravity := 9.8
var ui_box: Control                   # touches on the buttons are not used for walking/looking


func _ready() -> void:
	var col := CollisionShape3D.new()
	var cap := CapsuleShape3D.new()
	cap.radius = 0.25
	cap.height = 1.7
	col.shape = cap
	col.position.y = 0.85
	add_child(col)
	cam = Camera3D.new()
	cam.position.y = EYE
	cam.fov = 72.0
	cam.near = 0.05
	cam.far = 400.0
	add_child(cam)
	cam.current = true
	floor_snap_length = 0.45
	floor_max_angle = deg_to_rad(46.0)
	floor_stop_on_slope = true
	safe_margin = 0.02


func place(pos: Vector3, fwd: Vector3) -> void:
	global_position = pos + Vector3(0, 0.05, 0)
	yaw = atan2(-fwd.x, -fwd.z)
	pitch = 0.0
	velocity = Vector3.ZERO
	_apply_rot()


func _apply_rot() -> void:
	rotation = Vector3(0, yaw, 0)
	cam.rotation = Vector3(pitch, 0, 0)


func _input(event: InputEvent) -> void:
	var half := get_viewport().get_visible_rect().size.x * 0.5
	if event is InputEventScreenTouch:
		if event.pressed:
			if ui_box != null and ui_box.get_global_rect().grow(12).has_point(event.position):
				return
			if event.position.x < half and stick_id == -1:
				stick_id = event.index
				stick_origin = event.position
				stick_pos = event.position
			elif event.position.x >= half and look_id == -1:
				look_id = event.index
				look_last = event.position
		else:
			if event.index == stick_id:
				stick_id = -1
				move_vec = Vector2.ZERO
			elif event.index == look_id:
				look_id = -1
	elif event is InputEventScreenDrag:
		if event.index == stick_id:
			stick_pos = event.position
			var r := 90.0 * get_viewport().get_visible_rect().size.y / 720.0
			var d: Vector2 = (stick_pos - stick_origin) / r
			if d.length() > 1.0:
				d = d.normalized()
			move_vec = d
		elif event.index == look_id:
			var dd: Vector2 = event.position - look_last
			look_last = event.position
			_look(dd * 0.22)
	elif event is InputEventMouseMotion and Input.get_mouse_mode() == Input.MOUSE_MODE_CAPTURED:
		_look(event.relative * 0.15)
	elif event is InputEventMouseButton and event.pressed and event.button_index == MOUSE_BUTTON_LEFT and not DisplayServer.is_touchscreen_available():
		Input.set_mouse_mode(Input.MOUSE_MODE_CAPTURED)
	elif event is InputEventKey and event.pressed and event.keycode == KEY_ESCAPE:
		Input.set_mouse_mode(Input.MOUSE_MODE_VISIBLE)


func _look(d: Vector2) -> void:
	yaw -= deg_to_rad(d.x)
	pitch = clamp(pitch - deg_to_rad(d.y), deg_to_rad(-80), deg_to_rad(80))
	_apply_rot()


func _physics_process(delta: float) -> void:
	var v := move_vec
	var k := Input.get_vector("ui_left", "ui_right", "ui_up", "ui_down")
	if Input.is_key_pressed(KEY_W): k.y -= 1
	if Input.is_key_pressed(KEY_S): k.y += 1
	if Input.is_key_pressed(KEY_A): k.x -= 1
	if Input.is_key_pressed(KEY_D): k.x += 1
	if k.length() > 0.0:
		v = k.limit_length(1.0)
	var speed := RUN if (run or Input.is_key_pressed(KEY_SHIFT)) else WALK
	var basis_y := Basis(Vector3.UP, yaw)
	var dir := basis_y * Vector3(v.x, 0, v.y)
	velocity.x = dir.x * speed
	velocity.z = dir.z * speed
	if not is_on_floor():
		velocity.y -= gravity * delta
	else:
		velocity.y = min(velocity.y, 0.0)
	move_and_slide()
	if global_position.y < -30.0:
		get_parent().respawn()
