// Station surfaces: diffuse light comes from the lightmap baked in Cycles (emitted, so it needs no
// realtime lights); reflections come from the reflection probes through the normal specular path.
shader_type spatial;
render_mode diffuse_burley, specular_schlick_ggx, cull_back;

uniform vec3 albedo_color : source_color = vec3(1.0);
uniform sampler2D albedo_tex : source_color, filter_linear_mipmap_anisotropic, repeat_enable;
uniform bool use_albedo_tex = false;
uniform vec2 uv_scale = vec2(1.0);

uniform sampler2D rough_tex : hint_default_white, filter_linear_mipmap_anisotropic, repeat_enable;
uniform bool use_rough_tex = false;
uniform float roughness_val : hint_range(0.0, 1.0) = 0.5;

uniform sampler2D normal_tex : hint_normal, filter_linear_mipmap_anisotropic, repeat_enable;
uniform int normal_mode = 0;          // 0 none, 1 tangent space, 2 object space (Blender axes)

uniform float metallic_val : hint_range(0.0, 1.0) = 0.0;
uniform float coat = 0.0;

uniform sampler2D emission_tex : source_color, filter_linear_mipmap_anisotropic, repeat_enable;
uniform bool use_emission_tex = false;
uniform vec3 emission_color : source_color = vec3(0.0);
uniform float emission_strength = 0.0;

uniform sampler2D lightmap : filter_linear_mipmap, repeat_disable;
uniform float lm_gain = 1.0;
// a metal shows its lit surroundings: the lightmap holds the light arriving at it (baked as if
// it were not metal), and this share of it comes back off the rough metal; the reflection probes
// add the sharp part on top (where none reaches, stainless no longer goes black)
uniform float metal_env = 0.55;

// large-scale wear on floors: gloss and tone vary across the station like a real worn floor
uniform sampler2D wear_tex : filter_linear_mipmap, repeat_enable;
uniform float wear_rough = 0.0;
uniform float wear_albedo = 0.0;

varying vec3 world_pos;

void vertex() {
	world_pos = (MODEL_MATRIX * vec4(VERTEX, 1.0)).xyz;
}

void fragment() {
	vec2 uv = UV * uv_scale;
	vec3 alb = albedo_color;
	if (use_albedo_tex) {
		alb *= texture(albedo_tex, uv).rgb;
	}
	float r = use_rough_tex ? texture(rough_tex, uv).r : roughness_val;
	if (wear_rough > 0.0 || wear_albedo > 0.0) {
		float w = texture(wear_tex, world_pos.xz * 0.11).r;
		float w2 = texture(wear_tex, world_pos.xz * 0.73 + vec2(0.37, 0.11)).r;
		r += wear_rough * (w - 0.5) + 0.5 * wear_rough * (w2 - 0.5);
		alb *= 1.0 + wear_albedo * (w2 - 0.5);
	}
	vec3 l = texture(lightmap, UV2).rgb;
	l *= l;
	vec3 irradiance = l / max(vec3(1.0) - l, vec3(0.002));
	vec3 em = vec3(0.0);
	if (emission_strength > 0.0) {
		em = (use_emission_tex ? texture(emission_tex, uv).rgb : emission_color) * emission_strength;
	}
	ALBEDO = alb;
	METALLIC = metallic_val;
	ROUGHNESS = clamp(r, 0.04, 1.0);
	SPECULAR = 0.5;
	EMISSION = alb * ((1.0 - metallic_val) + metallic_val * metal_env) * irradiance * lm_gain + em;
	if (coat > 0.0) {
		CLEARCOAT = coat;
		CLEARCOAT_ROUGHNESS = 0.06;
	}
	if (normal_mode == 1) {
		NORMAL_MAP = texture(normal_tex, uv).rgb;
	} else if (normal_mode == 2) {
		vec3 n = texture(normal_tex, uv).rgb * 2.0 - 1.0;
		vec3 nw = normalize(vec3(n.x, n.z, -n.y));
		NORMAL = normalize((VIEW_MATRIX * vec4(nw, 0.0)).xyz);
	}
}
