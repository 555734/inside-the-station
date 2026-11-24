"""Close-up realism for the Umeda materials: reflections, wear and grime, built as shader nodes.

Every preset works in world space (Geometry > Position) so wear is continuous across objects,
and uses the height above the nearest floor (B1 at z=0, B2 at z=-5.8) for dirt that gathers
near the floor and smudges at hand height. Presets only modify an existing Principled BSDF:
whatever already feeds Base Color, Roughness and Normal is kept and modulated.
"""
import bpy

Z_FLOORS = (0.0, -5.8)


class _NT:
    def __init__(self, mat):
        self.m = mat
        self.nt = mat.node_tree
        self.b = self.nt.nodes["Principled BSDF"]
        g = self.nt.nodes.new("ShaderNodeNewGeometry")
        self.pos = g.outputs["Position"]
        sep = self.n("ShaderNodeSeparateXYZ")
        self.l(self.pos, sep.inputs[0])
        self.z = sep.outputs["Z"]
        # height above the floor of the level the point is on
        upper = self.math("GREATER_THAN", self.z, Z_FLOORS[1] + 4.5)
        flo = self.math("MULTIPLY_ADD", upper, Z_FLOORS[0] - Z_FLOORS[1], Z_FLOORS[1])
        self.hz = self.math("SUBTRACT", self.z, flo)

    def n(self, t, **kw):
        node = self.nt.nodes.new(t)
        for k, v in kw.items():
            setattr(node, k, v)
        return node

    def l(self, a, b):
        self.nt.links.new(a, b)

    def math(self, op, a, b=None, c=None, clamp=False):
        m = self.n("ShaderNodeMath", operation=op, use_clamp=clamp)
        for i, v in enumerate((a, b, c)):
            if v is None:
                continue
            if isinstance(v, (int, float)):
                m.inputs[i].default_value = v
            else:
                self.l(v, m.inputs[i])
        return m.outputs[0]

    def noise(self, scale, detail=4.0, rough=0.55, stretch=(1, 1, 1), dist=0.0, seed=0.0, local=False):
        vec = self.pos
        if local:
            tc = self.n("ShaderNodeTexCoord")
            vec = tc.outputs["Object"]
        if stretch != (1, 1, 1) or seed:
            mp = self.n("ShaderNodeMapping")
            mp.inputs["Scale"].default_value = stretch
            mp.inputs["Location"].default_value = (seed * 13.1, seed * 7.7, seed * 3.3)
            self.l(vec, mp.inputs["Vector"])
            vec = mp.outputs["Vector"]
        nz = self.n("ShaderNodeTexNoise")
        nz.inputs["Scale"].default_value = scale
        nz.inputs["Detail"].default_value = detail
        nz.inputs["Roughness"].default_value = rough
        nz.inputs["Distortion"].default_value = dist
        self.l(vec, nz.inputs["Vector"])
        return nz.outputs["Fac"]

    def ramp(self, x, a, b, lo=0.0, hi=1.0):
        """Linear map of x from [a, b] to [lo, hi], clamped."""
        mr = self.n("ShaderNodeMapRange", clamp=True)
        mr.inputs["From Min"].default_value = a
        mr.inputs["From Max"].default_value = b
        mr.inputs["To Min"].default_value = lo
        mr.inputs["To Max"].default_value = hi
        self.l(x, mr.inputs["Value"])
        return mr.outputs["Result"]

    # --- the three BSDF inputs, each with "current value" access
    def _src(self, name):
        inp = self.b.inputs[name]
        if inp.links:
            return inp.links[0].from_socket
        v = inp.default_value
        if name == "Base Color":
            rgb = self.n("ShaderNodeRGB")
            rgb.outputs[0].default_value = tuple(v)
            return rgb.outputs[0]
        val = self.n("ShaderNodeValue")
        val.outputs[0].default_value = float(v)
        return val.outputs[0]

    def mul_color(self, factor):
        src = self._src("Base Color")
        vm = self.n("ShaderNodeVectorMath", operation="SCALE")
        self.l(src, vm.inputs[0])
        self.l(factor, vm.inputs["Scale"])
        self.l(vm.outputs["Vector"], self.b.inputs["Base Color"])

    def add_rough(self, delta, lo=0.02, hi=1.0):
        src = self._src("Roughness")
        s = self.math("ADD", src, delta)
        s = self.math("MAXIMUM", s, lo)
        s = self.math("MINIMUM", s, hi)
        self.l(s, self.b.inputs["Roughness"])

    def bump(self, height, strength, distance=0.001):
        bp = self.n("ShaderNodeBump")
        bp.inputs["Strength"].default_value = strength
        bp.inputs["Distance"].default_value = distance
        self.l(height, bp.inputs["Height"])
        inp = self.b.inputs["Normal"]
        if inp.links:
            self.l(inp.links[0].from_socket, bp.inputs["Normal"])
        self.l(bp.outputs["Normal"], inp)

    # --- common wear layers
    def floor_dirt(self, amount=0.25, height=0.35, rough=0.15, seed=0.0):
        """Darker, duller band just above the floor, broken up by noise."""
        near = self.ramp(self.hz, height, 0.0, 0.0, 1.0)
        brk = self.ramp(self.noise(3.0, 6.0, 0.6, seed=seed), 0.35, 0.7, 0.3, 1.0)
        d = self.math("MULTIPLY", near, brk)
        self.mul_color(self.math("MULTIPLY_ADD", d, -amount, 1.0))
        self.add_rough(self.math("MULTIPLY", d, rough))

    def hand_smudge(self, rough=0.12, seed=1.0):
        """Greasy patches where people touch (0.8-1.6 m)."""
        band = self.math("MULTIPLY", self.ramp(self.hz, 0.7, 0.95), self.ramp(self.hz, 1.7, 1.4))
        sm = self.ramp(self.noise(9.0, 3.0, 0.5, seed=seed), 0.5, 0.75)
        self.add_rough(self.math("MULTIPLY", self.math("MULTIPLY", band, sm), rough))

    def blotches(self, scale=2.5, rough=0.08, color=0.04, seed=2.0):
        nz = self.noise(scale, 5.0, 0.6, seed=seed)
        self.add_rough(self.ramp(nz, 0.3, 0.7, -rough, rough))
        if color:
            self.mul_color(self.ramp(nz, 0.3, 0.7, 1.0 - color, 1.0 + color * 0.3))


# ------------------------------------------------------------------ presets

def floor(mat, seed=0.0):
    """Polished stone / ceramic pavers: sharp-ish reflections that vary lane by lane, scuffs, grime."""
    s = _NT(mat)
    # traffic lanes: worn (duller) vs. polished patches at metre scale
    lanes = s.noise(0.45, 3.0, 0.5, seed=seed)
    s.add_rough(s.ramp(lanes, 0.35, 0.65, -0.10, 0.10))
    # fine smudges and wet-mop streaks
    streak = s.noise(6.0, 4.0, 0.6, stretch=(1.0, 6.0, 1.0), seed=seed + 1)
    s.add_rough(s.ramp(streak, 0.45, 0.7, 0.0, 0.08))
    # scuff marks: short dark smears
    sc = s.noise(14.0, 2.0, 0.5, stretch=(4.0, 1.0, 1.0), seed=seed + 2)
    scuff = s.ramp(sc, 0.70, 0.78)
    s.mul_color(s.math("MULTIPLY_ADD", scuff, -0.18, 1.0))
    # sparse dark spots (old gum, stains)
    sp = s.ramp(s.noise(40.0, 1.0, 0.4, seed=seed + 3), 0.80, 0.84)
    s.mul_color(s.math("MULTIPLY_ADD", sp, -0.35, 1.0))
    # large-scale tone variation
    s.mul_color(s.ramp(s.noise(0.25, 3.0, 0.5, seed=seed + 4), 0.3, 0.7, 0.90, 1.04))


def brushed(mat, axis="Z", seed=0.0, smudge=True):
    """Brushed stainless: anisotropic highlights, brushing streaks, fingerprints."""
    s = _NT(mat)
    s.b.inputs["Anisotropic"].default_value = 0.55
    tg = s.n("ShaderNodeTangent")
    tg.direction_type = "RADIAL"
    tg.axis = "Z" if axis == "Z" else "X"
    s.l(tg.outputs["Tangent"], s.b.inputs["Tangent"])
    st = (60.0, 60.0, 0.4) if axis == "Z" else (0.4, 60.0, 60.0)
    # object space, so the grain follows each part (gates and frames are rotated boxes)
    streak = s.noise(8.0, 8.0, 0.7, stretch=st, seed=seed, local=True)
    s.add_rough(s.ramp(streak, 0.3, 0.7, -0.07, 0.09))
    s.mul_color(s.ramp(streak, 0.3, 0.7, 0.93, 1.04))
    s.bump(streak, 0.03, 0.0005)
    if smudge:
        s.hand_smudge(0.18, seed + 1)
        s.floor_dirt(0.18, 0.15, 0.2, seed + 2)


def paint(mat, seed=0.0, dirt=0.22):
    """Powder-coated / enamelled panels: orange-peel, faint smudges, grime at the base."""
    s = _NT(mat)
    peel = s.noise(350.0, 2.0, 0.5, seed=seed)
    s.bump(peel, 0.05, 0.0003)
    s.blotches(2.0, 0.07, 0.03, seed + 1)
    s.hand_smudge(0.12, seed + 2)
    s.floor_dirt(dirt, 0.3, 0.15, seed + 3)


def tile(mat, seed=0.0):
    """Glazed wall tiles: smears in the glaze, grime toward the floor."""
    s = _NT(mat)
    s.blotches(3.0, 0.05, 0.02, seed)
    s.hand_smudge(0.08, seed + 1)
    s.floor_dirt(0.25, 0.5, 0.2, seed + 2)


def glass_cover(mat, seed=0.0):
    """Screens and backlit signs: a glossy cover sheet with faint smudges on it."""
    s = _NT(mat)
    b = s.b
    b.inputs["Coat Weight"].default_value = 1.0
    b.inputs["Coat IOR"].default_value = 1.5
    sm = s.noise(5.0, 4.0, 0.6, seed=seed)
    cr = s.ramp(sm, 0.4, 0.75, 0.015, 0.12)
    s.l(cr, b.inputs["Coat Roughness"])


def floor_decal(mat, seed=0.0):
    """Printed floor stickers: matte, scuffed and greyed by foot traffic."""
    s = _NT(mat)
    s.b.inputs["Roughness"].default_value = 0.45
    wear = s.noise(9.0, 6.0, 0.65, seed=seed)
    s.mul_color(s.ramp(wear, 0.35, 0.75, 0.80, 1.0))
    sc = s.ramp(s.noise(18.0, 2.0, 0.5, stretch=(4.0, 1.0, 1.0), seed=seed + 1), 0.66, 0.74)
    s.mul_color(s.math("MULTIPLY_ADD", sc, -0.25, 1.0))
    s.add_rough(s.ramp(wear, 0.3, 0.7, -0.1, 0.15))


def ceiling(mat, seed=0.0):
    s = _NT(mat)
    s.blotches(1.2, 0.05, 0.05, seed)


def apply_all(M):
    """Apply the presets to the Umeda material set (by key, where present)."""
    plan = {
        floor: ["b2_floor", "b2_floor_band", "b1_floor"],
        tile: ["track_tile", "wall_white", "wall_tile", "b1_wall"],
        paint: ["kiosk_black", "kiosk_white", "navy", "clad_dark", "psd_panel", "panel_white",
                "parapet", "esc_clad", "beige", "column", "portal", "panel_grey"],
        glass_cover: ["ad0", "ad1", "ad2", "adp0", "adp1", "adp2", "band0", "band1", "band2",
                      "name_board_w", "name_board_e", "dep1", "dep2", "clock", "portal_sign",
                      "exit_sign", "exit_b1", "track1", "track2", "gate_fascia_n", "b1_sign_down",
                      "led_vision"],
        ceiling: ["ceil_b2", "ceil"],
        floor_decal: ["queue0", "queue1", "decal_up", "decal_down", "ic_only"],
    }
    seed = 0.0
    for fn, keys in plan.items():
        for k in keys:
            if k in M:
                fn(M[k], seed=seed)
                seed += 1.7
    for k, ax in (("col_steel", "Z"), ("steel_brushed", "X"), ("gate_side", "X"), ("steel", "X")):
        if k in M:
            brushed(M[k], ax, seed=seed, smudge=k != "steel")
            seed += 1.3
