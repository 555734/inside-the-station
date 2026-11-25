extends Control
## Draws the floating joystick and a short hint.

var player: CharacterBody3D


func _process(_d: float) -> void:
	queue_redraw()


func _draw() -> void:
	if player == null:
		return
	var r := 90.0 * get_viewport_rect().size.y / 720.0
	if player.stick_id != -1:
		draw_circle(player.stick_origin, r, Color(1, 1, 1, 0.12))
		draw_arc(player.stick_origin, r, 0, TAU, 48, Color(1, 1, 1, 0.35), 2.0)
		var k: Vector2 = player.stick_origin + player.move_vec * r
		draw_circle(k, r * 0.38, Color(1, 1, 1, 0.45))
	else:
		var f := get_theme_default_font()
		var y := get_viewport_rect().size.y - 24
		draw_string(f, Vector2(24, y), "左側：ドラッグで移動　　右側：ドラッグで見回す", HORIZONTAL_ALIGNMENT_LEFT, -1, 22, Color(1, 1, 1, 0.55))
