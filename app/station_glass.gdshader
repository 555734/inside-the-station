// Clear glass (balustrades): see-through, a faint tint, reflections from the probes and a little of
// the light around it (from its lightmap) so it reads as a pane, both faces drawn.
shader_type spatial;
render_mode blend_mix, depth_draw_opaque, cull_disabled, diffuse_burley, specular_schlick_ggx;

uniform vec3 albedo_color : source_color = vec3(0.8, 0.88, 0.86);
uniform float alpha = 0.14;
uniform float roughness_val : hint_range(0.0, 1.0) = 0.03;
uniform sampler2D lightmap : filter_linear_mipmap, repeat_disable;
uniform float lm_gain = 1.0;

void fragment() {
	vec3 l = texture(lightmap, UV2).rgb;
	l *= l;
	vec3 irradiance = l / max(vec3(1.0) - l, vec3(0.002));
	// more opaque at grazing angles (Fresnel), as real glass
	float f = pow(1.0 - clamp(abs(dot(NORMAL, VIEW)), 0.0, 1.0), 5.0);
	ALBEDO = albedo_color;
	ROUGHNESS = roughness_val;
	METALLIC = 0.0;
	SPECULAR = 0.5;
	EMISSION = albedo_color * irradiance * lm_gain * 0.25;
	ALPHA = clamp(alpha + f * 0.6, 0.0, 0.9);
}
