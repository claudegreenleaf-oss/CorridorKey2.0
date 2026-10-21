"""Render one randomized keying clip with exact ground truth (Blender / Cycles).

Run headless:
  blender -b --factory-startup -P render/gen_scene.py -- --seed 7 --out /data/clip_000007

Per clip it writes
  fg/0001.exr ...  subject only, transparent film: premultiplied scene-linear
                   RGBA. Alpha and color are exact, with real motion blur,
                   depth of field, hair, volumes and glass.
  bg/0001.exr ...  the backing as the camera sees it with the subject hidden,
                   including the subject's real shadow on the screen.
  meta.json        everything that was randomized.

The green or blue screen is a real lit surface behind the subject. It is
hidden from the camera in the foreground pass, but it still bounces light
onto the subject, so spill and screen reflections are physically correct.
The plate is then  plate = fg.rgb + (1 - fg.a) * bg,  exactly.

Subjects: procedural hair grooms (straight, wavy, curly, frizz, flyaways) on
a head or on random shapes as fur, CC0 Poly Haven models (plants, dandelion,
furniture), drifting smoke and fire volumes, glass with liquid, sheer
patterned cloth, and fast particle clouds. A clip mixes one primary subject
with optional secondary layers.
"""

import json
import math
import os
import random
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def arg(name, default, cast=str):
    if f"--{name}" in ARGS:
        return cast(ARGS[ARGS.index(f"--{name}") + 1])
    return default


SEED = arg("seed", 0, int)
OUT = Path(arg("out", "/tmp/ck_render"))
FRAMES = arg("frames", 8, int)
RES = (arg("width", 1920, int), arg("height", 1080, int))
SAMPLES = arg("samples", 128, int)
BG_SAMPLES = arg("bg_samples", 32, int)
ASSETS = Path(os.path.expanduser(arg("assets", "~/babel/assets/ph")))
THREADS = arg("threads", 32, int)
DEVICE = arg("device", "CPU")
FORCE_PRIMARY = arg("primary", -1, int)   # index into PRIMARY, for previews

R = random.Random(SEED)
NP = np.random.default_rng(SEED)
META = {"seed": SEED, "layers": []}


# ---------------------------------------------------------------------------
# scene
# ---------------------------------------------------------------------------

def setup_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = DEVICE
    if DEVICE == "GPU":
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "OPTIX"
        prefs.get_devices()
        for d in prefs.devices:
            d.use = d.type == "OPTIX"
    sc.render.threads_mode = "FIXED"
    sc.render.threads = THREADS
    sc.render.resolution_x, sc.render.resolution_y = RES
    sc.render.resolution_percentage = 100
    sc.render.film_transparent = True
    sc.cycles.film_transparent_glass = True
    sc.cycles.film_transparent_roughness = 0.1
    sc.cycles.samples = SAMPLES
    sc.cycles.use_adaptive_sampling = True
    sc.cycles.adaptive_threshold = 0.01
    sc.cycles.use_denoising = False
    sc.cycles.max_bounces = 8
    sc.cycles.transparent_max_bounces = 64
    sc.cycles.volume_bounces = 2
    sc.cycles.caustics_reflective = False
    sc.cycles.caustics_refractive = False
    sc.render.use_motion_blur = True
    shutter = R.choice([0.25, 0.5, 0.5, 0.5, 0.75, 1.0])
    sc.render.motion_blur_shutter = shutter
    sc.view_settings.view_transform = "Standard"
    sc.view_settings.look = "None"
    sc.render.image_settings.file_format = "OPEN_EXR"
    sc.render.image_settings.exr_codec = "ZIP"
    sc.render.image_settings.color_depth = "16"
    sc.render.image_settings.color_mode = "RGBA"
    sc.frame_start, sc.frame_end = 1, FRAMES
    META["shutter"] = shutter
    return sc


def world_light():
    w = bpy.data.worlds.new("w")
    bpy.context.scene.world = w
    w.use_nodes = True
    nt = w.node_tree
    bg = nt.nodes["Background"]
    hdris = sorted((ASSETS / "hdri").glob("*.hdr"))
    strength = R.uniform(0.05, 0.6)
    if hdris and R.random() < 0.8:
        env = nt.nodes.new("ShaderNodeTexEnvironment")
        env.image = bpy.data.images.load(str(R.choice(hdris)))
        mapping = nt.nodes.new("ShaderNodeMapping")
        coord = nt.nodes.new("ShaderNodeTexCoord")
        mapping.inputs["Rotation"].default_value[2] = R.uniform(0, 2 * math.pi)
        nt.links.new(coord.outputs["Generated"], mapping.inputs["Vector"])
        nt.links.new(mapping.outputs["Vector"], env.inputs["Vector"])
        nt.links.new(env.outputs["Color"], bg.inputs["Color"])
    else:
        bg.inputs["Color"].default_value = (*[R.uniform(0.3, 1.0)] * 3, 1)
    bg.inputs["Strength"].default_value = strength


def area_light(name, loc, target, power, size, color):
    d = bpy.data.lights.new(name, "AREA")
    d.energy = power
    d.size = size
    d.color = color
    o = bpy.data.objects.new(name, d)
    bpy.context.collection.objects.link(o)
    o.location = loc
    o.rotation_euler = (Vector(target) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
    return o


def lighting(center, radius):
    def warm():
        t = R.uniform(-0.15, 0.15)
        return (1 + t, 1.0, 1 - t)
    key_az = R.uniform(-1.2, 1.2)
    lights = []
    d = radius * R.uniform(3, 6)
    for name, az, el, pw, sz in (
        ("key", key_az, R.uniform(0.2, 0.8), R.uniform(300, 1500), R.uniform(0.5, 2.5)),
        ("fill", -key_az + R.uniform(-0.4, 0.4), R.uniform(0.0, 0.4), R.uniform(50, 500), R.uniform(1, 4)),
        ("rim", math.pi + R.uniform(-0.8, 0.8), R.uniform(0.3, 1.0), R.uniform(200, 2000), R.uniform(0.3, 1.5)),
    ):
        if name == "fill" and R.random() < 0.3:
            continue
        loc = center + Vector((math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), math.sin(el))) * d
        lights.append(area_light(name, loc, center, pw * (d / 4) ** 2, sz, warm()))
    META["lights"] = len(lights)


def make_screen(center, cam_loc):
    """A big lit backing behind the subject, facing the camera."""
    color = "blue" if R.random() < 0.2 else "green"
    base = {"green": (0.06, R.uniform(0.35, 0.6), 0.08), "blue": (0.04, 0.09, R.uniform(0.35, 0.6))}[color]
    mat = bpy.data.materials.new("screen")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*base, 1)
    bsdf.inputs["Roughness"].default_value = R.uniform(0.6, 0.95)
    # fabric wrinkles: low-frequency bump
    if R.random() < 0.7:
        noise = nt.nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = R.uniform(0.3, 2.0)
        noise.inputs["Detail"].default_value = 3
        bump = nt.nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = R.uniform(0.05, 0.4)
        nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
        nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    # emissive boost: some stages light the screen hard and evenly
    bsdf.inputs["Emission Color"].default_value = (*base, 1)
    bsdf.inputs["Emission Strength"].default_value = R.choice([0.0, 0.0, R.uniform(0.2, 1.5)])
    away = (center - cam_loc).normalized()
    dist = R.uniform(1.0, 4.0) * max(1.0, (center - cam_loc).length * 0.4)
    bpy.ops.mesh.primitive_plane_add(size=1)
    plane = bpy.context.active_object
    plane.name = "screen"
    plane.scale = (60, 60, 1)
    plane.location = center + away * dist
    plane.rotation_euler = away.to_track_quat("Z", "Y").to_euler()
    plane.data.materials.append(mat)
    # optional floor screen (cyc) catching shadows
    META["screen"] = {"color": color, "base": base, "distance": dist}
    return plane


# ---------------------------------------------------------------------------
# subjects
# ---------------------------------------------------------------------------

def hair_material():
    m = bpy.data.materials.new("hair")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.remove(nt.nodes["Principled BSDF"])
    h = nt.nodes.new("ShaderNodeBsdfHairPrincipled")
    h.parametrization = "MELANIN"
    h.inputs["Melanin"].default_value = R.choice([R.uniform(0.05, 0.3), R.uniform(0.3, 0.8), R.uniform(0.8, 1.0)])
    h.inputs["Melanin Redness"].default_value = R.uniform(0.0, 0.9)
    h.inputs["Roughness"].default_value = R.uniform(0.2, 0.5)
    h.inputs["Radial Roughness"].default_value = R.uniform(0.2, 0.5)
    h.inputs["Random Roughness"].default_value = R.uniform(0.0, 0.3)
    if R.random() < 0.15:   # dyed
        h.parametrization = "COLOR"
        h.inputs["Color"].default_value = (R.random(), R.random(), R.random(), 1)
    nt.links.new(h.outputs[0], nt.nodes["Material Output"].inputs["Surface"])
    return m


def sample_surface(obj, n, region):
    """Area-weighted random points + normals on a mesh object (world space)."""
    me = obj.data
    me.calc_loop_triangles()
    tris = np.zeros(len(me.loop_triangles) * 3, np.int32)
    me.loop_triangles.foreach_get("vertices", tris)
    tris = tris.reshape(-1, 3)
    co = np.zeros(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get("co", co)
    mw = np.array(obj.matrix_world)
    co = co.reshape(-1, 3) @ mw[:3, :3].T + mw[:3, 3]
    a, b, c = co[tris[:, 0]], co[tris[:, 1]], co[tris[:, 2]]
    nrm = np.cross(b - a, c - a)
    area = np.linalg.norm(nrm, axis=1)
    nrm = nrm / np.maximum(area[:, None], 1e-12)
    cen = (a + b + c) / 3
    w = area * region(cen, nrm)
    if w.sum() <= 0:
        w = area
    idx = NP.choice(len(w), size=n, p=w / w.sum())
    u, v = NP.random((2, n))
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    pts = a[idx] + (b[idx] - a[idx]) * u[:, None] + (c[idx] - a[idx]) * v[:, None]
    return pts.astype(np.float32), nrm[idx].astype(np.float32)


def groom(roots, normals, scale, style, length, n_pts=14):
    """Strand polylines (n, n_pts, 3) from roots and normals."""
    n = len(roots)
    up = np.array([0, 0, 1], np.float32)
    grav = R.uniform(0.6, 0.95) if style != "fur" else 0.05
    L = length * scale * NP.uniform(0.7, 1.15, n)
    t = np.linspace(0, 1, n_pts, dtype=np.float32)
    # clumps: strands follow a nearby guide's direction noise
    n_cl = max(8, n // R.randint(60, 400))
    cl = NP.integers(0, n_cl, n)
    cl_phase = NP.uniform(0, 2 * np.pi, (n_cl, 2)).astype(np.float32)
    # Hair leaves the scalp at a shallow angle and flows along the head
    # toward the ground: blend the normal with the downhill surface tangent.
    down = -up[None, :] - (normals @ -up)[:, None] * normals
    down /= np.maximum(np.linalg.norm(down, axis=1, keepdims=True), 1e-6)
    lift = 0.9 if style == "fur" else R.uniform(0.2, 0.45)
    d0 = lift * normals + (1 - lift) * down + NP.normal(0, 0.08, normals.shape).astype(np.float32)
    d0 /= np.linalg.norm(d0, axis=1, keepdims=True)
    pts = np.zeros((n, n_pts, 3), np.float32)
    side = np.cross(d0, up)
    side /= np.maximum(np.linalg.norm(side, axis=1, keepdims=True), 1e-6)
    other = np.cross(side, d0)
    for i, ti in enumerate(t):
        bend = (1 - ti) * d0 + ti * ((1 - grav) * d0 - grav * up)
        bend /= np.maximum(np.linalg.norm(bend, axis=1, keepdims=True), 1e-6)
        p = roots + bend * (L * ti)[:, None]
        amp = L * {"straight": 0.01, "wavy": 0.05, "curly": 0.09, "frizz": 0.03, "fur": 0.02}[style]
        freq = {"straight": 1, "wavy": 3, "curly": 9, "frizz": 2, "fur": 1}[style]
        ph = cl_phase[cl]
        p += side * (amp * np.sin(2 * np.pi * freq * ti + ph[:, 0]))[:, None]
        p += other * (amp * np.cos(2 * np.pi * freq * ti + ph[:, 1]))[:, None] * (style == "curly")
        if style == "frizz":
            p += NP.normal(0, 1, p.shape).astype(np.float32) * (L * 0.02 * ti)[:, None]
        pts[:, i] = p
    # flyaways: a few long stray strands
    fly = NP.random(n) < (0.01 if style != "fur" else 0.0)
    pts[fly] += NP.normal(0, 1, (fly.sum(), 1, 3)).astype(np.float32) * (L[fly] * 0.3)[:, None, None] * t[None, :, None]
    return pts


def add_hair_object(name, pts, root_radius, mat):
    n, k, _ = pts.shape
    c = bpy.data.hair_curves.new(name)
    c.add_curves([k] * n)
    c.position_data.foreach_set("vector", pts.reshape(-1))
    if "radius" not in c.attributes:
        c.attributes.new("radius", "FLOAT", "POINT")
    taper = np.linspace(1.0, 0.15, k, dtype=np.float32)
    r = (root_radius * NP.uniform(0.7, 1.3, n).astype(np.float32))[:, None] * taper[None]
    c.attributes["radius"].data.foreach_set("value", r.reshape(-1))
    c.materials.append(mat)
    o = bpy.data.objects.new(name, c)
    bpy.context.collection.objects.link(o)
    return o


def import_gltf(name):
    folder = ASSETS / "models" / name
    files = list(folder.glob("*.gltf")) + list(folder.glob("*.glb"))
    if not files:
        return None
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(files[0]))
    new = [o for o in bpy.data.objects if o not in before]
    root = bpy.data.objects.new(name, None)
    bpy.context.collection.objects.link(root)
    for o in new:
        if o.parent is None:
            o.parent = root
    bpy.context.view_layer.update()
    lo, hi = bbox(new)
    size = max(hi - lo)
    s = 1.0 / max(size, 1e-6)
    root.scale = (s, s, s)
    root.location = -((lo + hi) / 2) * s
    bpy.context.view_layer.update()
    wrap = bpy.data.objects.new(name + "_wrap", None)
    bpy.context.collection.objects.link(wrap)
    root.parent = wrap
    return wrap


def bbox(objs):
    pts = []
    for o in objs:
        if o.type in ("MESH", "CURVES", "VOLUME") or o.type == "EMPTY" and False:
            pts += [o.matrix_world @ Vector(c) for c in o.bound_box]
    if not pts:
        return np.zeros(3), np.ones(3)
    a = np.array([[p.x, p.y, p.z] for p in pts])
    return a.min(0), a.max(0)


def subject_hair_head():
    head = import_gltf("marble_bust_01")
    if head is None:
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.5, segments=64, ring_count=32)
        head = bpy.context.active_object
    meshes = [o for o in head.children_recursive if o.type == "MESH"] or [head]
    bpy.context.view_layer.update()
    lo, hi = bbox(meshes)
    top = lo[2] + (hi[2] - lo[2]) * R.uniform(0.62, 0.72)
    mesh = max(meshes, key=lambda o: len(o.data.vertices))
    style = R.choice(["straight", "straight", "wavy", "wavy", "curly", "frizz"])
    n = R.randint(20000, 70000)
    region = lambda cen, nrm: ((cen[:, 2] > top) & (nrm[:, 1] > -0.6)).astype(np.float32)
    roots, nrm = sample_surface(mesh, n, region)
    pts = groom(roots, nrm, 1.0, style, R.uniform(0.15, 0.9))
    hair = add_hair_object("hair", pts, R.uniform(0.0006, 0.0015), hair_material())
    # skin-ish material on the bust
    skin = bpy.data.materials.new("skin")
    skin.use_nodes = True
    b = skin.node_tree.nodes["Principled BSDF"]
    tone = R.uniform(0.15, 0.85)
    b.inputs["Base Color"].default_value = (tone, tone * R.uniform(0.6, 0.8), tone * R.uniform(0.45, 0.65), 1)
    b.inputs["Subsurface Weight"].default_value = 0.3
    b.inputs["Roughness"].default_value = 0.5
    for m in meshes:
        m.data.materials.clear()
        m.data.materials.append(skin)
    hair.parent = head
    hair.matrix_parent_inverse = head.matrix_world.inverted()
    META["layers"].append({"type": "hair_head", "style": style, "strands": n})
    return head


def subject_fur_shape():
    shape = R.choice(["sphere", "torus", "monkey", "cube"])
    if shape == "sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.4, segments=48, ring_count=24)
    elif shape == "torus":
        bpy.ops.mesh.primitive_torus_add(major_radius=0.35, minor_radius=0.15)
    elif shape == "monkey":
        bpy.ops.mesh.primitive_monkey_add(size=0.7)
    else:
        bpy.ops.mesh.primitive_cube_add(size=0.6)
        bpy.ops.object.modifier_add(type="SUBSURF")
        bpy.context.active_object.modifiers[0].levels = 2
        bpy.ops.object.modifier_apply(modifier=bpy.context.active_object.modifiers[0].name)
    base = bpy.context.active_object
    n = R.randint(30000, 90000)
    roots, nrm = sample_surface(base, n, lambda c, nr: np.ones(len(c), np.float32))
    style = R.choice(["fur", "fur", "frizz", "wavy"])
    pts = groom(roots, nrm, 1.0, style, R.uniform(0.04, 0.25), n_pts=8)
    m = hair_material()
    fur = add_hair_object("fur", pts, R.uniform(0.0004, 0.0012), m)
    base.data.materials.append(m)
    fur.parent = base
    fur.matrix_parent_inverse = base.matrix_world.inverted()
    META["layers"].append({"type": "fur", "shape": shape, "strands": n, "style": style})
    return base


def subject_model():
    models = [p.name for p in (ASSETS / "models").iterdir() if p.is_dir()]
    fine = [m for m in models if any(k in m for k in ("plant", "fern", "dandelion", "grass", "nettle",
                                                       "weed", "shrub", "moss", "root", "anthurium",
                                                       "calathea", "pachira", "periwinkle"))]
    pool = fine if fine and R.random() < 0.75 else models
    name = R.choice(pool)
    o = import_gltf(name)
    META["layers"].append({"type": "model", "name": name})
    return o


def subject_smoke(fire=False):
    bpy.ops.mesh.primitive_cube_add(size=1.2)
    cube = bpy.context.active_object
    m = bpy.data.materials.new("smoke")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.remove(nt.nodes["Principled BSDF"])
    vol = nt.nodes.new("ShaderNodeVolumePrincipled")
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.noise_dimensions = "4D"
    noise.inputs["Scale"].default_value = R.uniform(1.5, 4.0)
    noise.inputs["Detail"].default_value = R.uniform(3, 8)
    noise.inputs["Roughness"].default_value = R.uniform(0.5, 0.7)
    # soft spherical falloff so the volume has no box edges
    geo = nt.nodes.new("ShaderNodeTexCoord")
    sph = nt.nodes.new("ShaderNodeTexGradient")
    sph.gradient_type = "QUADRATIC_SPHERE"
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (1.1, 1.1, 1.1)
    nt.links.new(geo.outputs["Object"], mp.inputs["Vector"])
    nt.links.new(mp.outputs["Vector"], sph.inputs["Vector"])
    ramp = nt.nodes.new("ShaderNodeMapRange")
    ramp.inputs["From Min"].default_value = R.uniform(0.45, 0.6)
    ramp.inputs["From Max"].default_value = 0.8
    ramp.inputs["To Max"].default_value = R.uniform(3, 25)
    mul = nt.nodes.new("ShaderNodeMath")
    mul.operation = "MULTIPLY"
    nt.links.new(noise.outputs["Fac"], ramp.inputs["Value"])
    nt.links.new(ramp.outputs["Result"], mul.inputs[0])
    nt.links.new(sph.outputs["Fac"], mul.inputs[1])
    nt.links.new(mul.outputs[0], vol.inputs["Density"])
    g = R.uniform(0.4, 0.95)
    vol.inputs["Color"].default_value = (g, g, g * R.uniform(0.9, 1.0), 1)
    if fire:
        vol.inputs["Blackbody Intensity"].default_value = R.uniform(1, 6)
        vol.inputs["Temperature"].default_value = R.uniform(1200, 2200)
        vol.inputs["Density"].default_value = 0
        nt.links.new(mul.outputs[0], vol.inputs["Blackbody Intensity"])
        vol.inputs["Color"].default_value = (0.1, 0.1, 0.1, 1)
    nt.links.new(vol.outputs[0], nt.nodes["Material Output"].inputs["Volume"])
    # animate the 4D noise so the smoke billows
    w0 = R.uniform(0, 10)
    noise.inputs["W"].default_value = w0
    noise.inputs["W"].keyframe_insert("default_value", frame=1)
    noise.inputs["W"].default_value = w0 + R.uniform(0.1, 0.8)
    noise.inputs["W"].keyframe_insert("default_value", frame=FRAMES)
    cube.data.materials.append(m)
    META["layers"].append({"type": "fire" if fire else "smoke"})
    return cube


def subject_glass():
    bpy.ops.mesh.primitive_cylinder_add(radius=0.22, depth=0.6, vertices=96)
    glass = bpy.context.active_object
    bpy.ops.object.modifier_add(type="SOLIDIFY")
    glass.modifiers[-1].thickness = R.uniform(0.005, 0.02)
    bpy.ops.object.modifier_add(type="BEVEL")
    m = bpy.data.materials.new("glass")
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Transmission Weight"].default_value = 1.0
    b.inputs["Roughness"].default_value = R.choice([0.0, 0.0, R.uniform(0.02, 0.2)])
    b.inputs["IOR"].default_value = R.uniform(1.44, 1.55)
    tint = R.uniform(0.85, 1.0)
    b.inputs["Base Color"].default_value = (tint, 1.0, tint * R.uniform(0.95, 1.0), 1)
    glass.data.materials.append(m)
    if R.random() < 0.6:
        bpy.ops.mesh.primitive_cylinder_add(radius=0.2, depth=R.uniform(0.15, 0.45), vertices=96,
                                            location=(0, 0, -0.1))
        liquid = bpy.context.active_object
        lm = bpy.data.materials.new("liquid")
        lm.use_nodes = True
        lb = lm.node_tree.nodes["Principled BSDF"]
        lb.inputs["Transmission Weight"].default_value = 1.0
        lb.inputs["IOR"].default_value = 1.33
        lb.inputs["Base Color"].default_value = (R.random() * 0.6 + 0.4, R.random() * 0.6 + 0.4,
                                                 R.random() * 0.6 + 0.4, 1)
        liquid.data.materials.append(lm)
        liquid.parent = glass
    META["layers"].append({"type": "glass"})
    return glass


def subject_cloth():
    bpy.ops.mesh.primitive_plane_add(size=1.2)
    cloth = bpy.context.active_object
    bpy.ops.object.modifier_add(type="SUBSURF")
    cloth.modifiers[-1].levels = 5
    cloth.modifiers[-1].subdivision_type = "SIMPLE"
    wave = cloth.modifiers.new("wave", "WAVE")
    wave.height = R.uniform(0.02, 0.08)
    wave.width = R.uniform(0.2, 0.6)
    wave.speed = R.uniform(0.02, 0.1)
    cloth.rotation_euler = (math.pi / 2, 0, R.uniform(-0.3, 0.3))
    m = bpy.data.materials.new("sheer")
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    c = (R.random(), R.random(), R.random())
    b.inputs["Base Color"].default_value = (*c, 1)
    b.inputs["Roughness"].default_value = 0.7
    weave = nt.nodes.new("ShaderNodeTexWave" if R.random() < 0.5 else "ShaderNodeTexVoronoi")
    weave.inputs["Scale"].default_value = R.uniform(20, 120)
    ramp = nt.nodes.new("ShaderNodeMapRange")
    ramp.inputs["To Min"].default_value = R.uniform(0.05, 0.4)
    ramp.inputs["To Max"].default_value = R.uniform(0.5, 0.95)
    nt.links.new(weave.outputs[0 if weave.bl_idname.endswith("Wave") else "Distance"], ramp.inputs["Value"])
    nt.links.new(ramp.outputs["Result"], b.inputs["Alpha"])
    cloth.data.materials.append(m)
    META["layers"].append({"type": "sheer_cloth"})
    return cloth


def subject_particles():
    root = bpy.data.objects.new("particles", None)
    bpy.context.collection.objects.link(root)
    n = R.randint(40, 200)
    kind = R.choice(["confetti", "drops", "sparks"])
    m = bpy.data.materials.new(kind)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    if kind == "drops":
        b.inputs["Transmission Weight"].default_value = 1.0
        b.inputs["Roughness"].default_value = 0.0
    elif kind == "sparks":
        b.inputs["Emission Color"].default_value = (1.0, R.uniform(0.4, 0.8), 0.2, 1)
        b.inputs["Emission Strength"].default_value = R.uniform(5, 40)
    for i in range(n):
        if kind == "confetti":
            bpy.ops.mesh.primitive_plane_add(size=R.uniform(0.01, 0.04))
            col = bpy.data.materials.new(f"c{i}")
            col.use_nodes = True
            col.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (
                R.random(), R.random(), R.random(), 1)
            mat = col
        else:
            bpy.ops.mesh.primitive_ico_sphere_add(radius=R.uniform(0.003, 0.012), subdivisions=2)
            mat = m
        o = bpy.context.active_object
        o.data.materials.append(mat)
        o.location = (R.uniform(-0.6, 0.6), R.uniform(-0.6, 0.6), R.uniform(-0.6, 0.6))
        o.rotation_euler = (R.uniform(0, 6.3), R.uniform(0, 6.3), R.uniform(0, 6.3))
        v = Vector((R.gauss(0, 0.06), R.gauss(0, 0.06), R.gauss(-0.05, 0.06)))
        o.keyframe_insert("location", frame=1)
        o.location += v * FRAMES
        o.rotation_euler.x += R.uniform(-3, 3)
        o.keyframe_insert("location", frame=FRAMES)
        o.keyframe_insert("rotation_euler", frame=FRAMES)
        o.parent = root
    META["layers"].append({"type": "particles", "kind": kind, "count": n})
    return root


PRIMARY = [(subject_hair_head, 0.35), (subject_fur_shape, 0.15), (subject_model, 0.25),
           (subject_glass, 0.08), (subject_cloth, 0.07), (lambda: subject_smoke(False), 0.06),
           (lambda: subject_smoke(True), 0.04)]
SECONDARY = [(lambda: subject_smoke(False), 0.4), (subject_particles, 0.35), (subject_cloth, 0.25)]


def pick(table):
    x = R.random() * sum(w for _, w in table)
    for f, w in table:
        x -= w
        if x <= 0:
            return f
    return table[-1][0]


def animate(obj, speed):
    """Rigid motion with keyframes, so Cycles motion blur is exact."""
    p0 = Vector(obj.location)
    r0 = obj.rotation_euler.copy()
    v = Vector((R.gauss(0, 1), R.gauss(0, 0.3), R.gauss(0, 0.6))) * speed
    w = Vector((R.gauss(0, 1), R.gauss(0, 1), R.gauss(0, 1))) * speed * 1.5
    for f in range(1, FRAMES + 1):
        t = (f - 1)
        obj.location = p0 + v * t
        obj.rotation_euler = (r0.x + w.x * t, r0.y + w.y * t, r0.z + w.z * t)
        obj.keyframe_insert("location", frame=f)
        obj.keyframe_insert("rotation_euler", frame=f)


# ---------------------------------------------------------------------------


def main():
    sc = setup_scene()
    world_light()

    prim = (PRIMARY[FORCE_PRIMARY][0] if FORCE_PRIMARY >= 0 else pick(PRIMARY))()
    prim.rotation_euler = (R.uniform(-0.3, 0.3), R.uniform(-0.3, 0.3), R.uniform(-math.pi, math.pi))
    speed = R.choice([0.0, 0.01, 0.02, 0.05, 0.12])   # scene units per frame; 0.12 is a whip
    animate(prim, speed)
    layers = [prim]
    for _ in range(R.choice([0, 0, 1, 1, 2])):
        o = pick(SECONDARY)()
        side = R.choice([-1, 1])
        o.location = (side * R.uniform(0.15, 0.6), R.uniform(-0.4, 0.3), R.uniform(-0.3, 0.3))
        o.scale = [R.uniform(0.4, 1.0)] * 3
        animate(o, R.choice([0.0, 0.01, 0.03]))
        layers.append(o)
    META["speed"] = speed
    bpy.context.view_layer.update()

    # camera framing the primary
    cam_d = bpy.data.cameras.new("cam")
    cam_d.lens = R.choice([24, 35, 50, 50, 85, 85, 135])
    cam = bpy.data.objects.new("cam", cam_d)
    bpy.context.collection.objects.link(cam)
    sc.camera = cam
    center = Vector((0, 0, 0))
    radius = 0.5
    fov = 2 * math.atan(cam_d.sensor_width / 2 / cam_d.lens) * RES[1] / RES[0]
    fill = R.uniform(0.5, 1.3)         # >1 crops the subject at the frame edge
    dist = radius / math.tan(fov / 2) / fill
    az = R.uniform(-0.5, 0.5)
    el = R.uniform(-0.2, 0.3)
    cam.location = center + Vector((math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), math.sin(el))) * dist
    cam.rotation_euler = (center - cam.location).to_track_quat("-Z", "Y").to_euler()
    # handheld shake
    if R.random() < 0.4:
        base = cam.rotation_euler.copy()
        for f in range(1, FRAMES + 1):
            cam.rotation_euler = (base.x + R.gauss(0, 0.004), base.y, base.z + R.gauss(0, 0.004))
            cam.keyframe_insert("rotation_euler", frame=f)
    if R.random() < 0.6:
        cam_d.dof.use_dof = True
        cam_d.dof.focus_distance = dist * R.uniform(0.85, 1.15)
        cam_d.dof.aperture_fstop = R.choice([1.4, 2.0, 2.8, 4.0, 5.6, 8.0])
    META["camera"] = {"lens": cam_d.lens, "distance": dist, "dof": cam_d.dof.use_dof,
                      "fstop": cam_d.dof.aperture_fstop if cam_d.dof.use_dof else None}

    lighting(center, radius)
    screen = make_screen(center, cam.location)

    subjects = [o for o in bpy.data.objects
                if o.type in ("MESH", "CURVES", "EMPTY") and o.name != "screen" and o.name != "cam"]

    # --- pass 1: foreground (screen hidden from camera, still lights subject)
    screen.visible_camera = False
    screen.visible_transmission = False
    sc.cycles.samples = SAMPLES
    sc.render.filepath = str(OUT / "fg" / "####")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "meta.json").write_text(json.dumps(META, indent=1))
    bpy.ops.render.render(animation=True)

    # --- pass 2: backing with the subject hidden but still casting shadows
    screen.visible_camera = True
    for o in subjects:
        o.visible_camera = False
        o.visible_transmission = False
    sc.cycles.samples = BG_SAMPLES
    sc.cycles.use_denoising = True
    sc.render.film_transparent = False
    sc.render.image_settings.color_mode = "RGB"
    sc.render.filepath = str(OUT / "bg" / "####")
    bpy.ops.render.render(animation=True)
    (OUT / "done").write_text("ok")


main()
