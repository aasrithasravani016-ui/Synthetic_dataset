import blenderproc as bproc
import bpy
import os
import random
import math
import cv2
import numpy as np
import mathutils
import json
from bpy_extras.object_utils import world_to_camera_view

# ==============================
# INIT
# ==============================
bproc.init()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Go up one level from scripts/
DEVICES_DIR = os.path.join(BASE_DIR, "Assets", "devices_front_only")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
LABELS_DIR = os.path.join(OUTPUT_DIR, "labels")
PORT_JSON = os.path.join(BASE_DIR, "config", "port_locations.json")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LABELS_DIR, exist_ok=True)

# ==============================
# CLASS DEFINITIONS FOR OBJECT DETECTION
# ==============================
CLASSES = {
    'patch_panel': 0,
    'pdu': 1,
    'switch': 2,
    'server': 3,
}
CLASS_NAMES = ['patch_panel', 'pdu', 'switch', 'server']

# Save classes.txt for YOLO
with open(os.path.join(OUTPUT_DIR, 'classes.txt'), 'w') as f:
    for cls_name in CLASS_NAMES:
        f.write(cls_name + '\n')

# ==============================
# LOAD DEVICE TEXTURES BY CATEGORY
# ==============================
def load_category(subfolder):
    folder = os.path.join(DEVICES_DIR, subfolder)
    if not os.path.isdir(folder):
        return []
    return [
        os.path.abspath(os.path.join(folder, f))
        for f in os.listdir(folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

switch_files = load_category("switches")
patch_panel_files = load_category("patch_panel")
pdu_files = load_category("pdu")
server_files = load_category("Server")

if not (switch_files or patch_panel_files):
    raise RuntimeError("Need at least switch or patch panel images")

print(f"Loaded: {len(switch_files)} switches, {len(patch_panel_files)} patch panels, "
      f"{len(pdu_files)} PDUs, {len(server_files)} servers")

port_locations = {}
if os.path.exists(PORT_JSON):
    with open(PORT_JSON, "r") as f:
        port_locations = json.load(f)
    print(f"Loaded port locations for {len(port_locations)} devices")

# ==============================
# CONSTANTS
# ==============================
RACK_WIDTH = 0.6
RACK_HEIGHT = 2.0
RACK_DEPTH = 1.0
U_HEIGHT = 0.0445
START_Z = 0.1
TOTAL_SLOTS = 42
Y_FRONT = -RACK_DEPTH / 2 + 0.05
NUM_IMAGES = 1000

CABLE_COLORS = [
    (0.55, 0.55, 0.55, 1),
    (0.55, 0.55, 0.55, 1),
    (0.55, 0.55, 0.55, 1),
    (0.30, 0.30, 0.30, 1),
    (0.10, 0.18, 0.38, 1),
    (0.38, 0.35, 0.10, 1),
    (0.38, 0.08, 0.08, 1),
    (0.08, 0.30, 0.10, 1),
]

# ==============================
# MATERIALS
# ==============================
def make_material(name, color, roughness=0.45, metallic=0.0):
    if name in bpy.data.materials:
        return bpy.data.materials[name]
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    return mat


def make_emissive_material(name, color, strength=3.0):
    if name in bpy.data.materials:
        return bpy.data.materials[name]
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for n in list(nodes):
        nodes.remove(n)
    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = color
    emission.inputs["Strength"].default_value = strength
    links.new(emission.outputs[0], output.inputs[0])
    return mat


def make_cable_material(name, color):
    if name in bpy.data.materials:
        return bpy.data.materials[name]
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = 0.85
    bsdf.inputs["Metallic"].default_value = 0.0
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.05
    elif "Specular" in bsdf.inputs:
        bsdf.inputs["Specular"].default_value = 0.05
    return mat


def create_device_material(img_path):
    mat_name = "tex_" + os.path.basename(img_path)
    if mat_name in bpy.data.materials:
        return bpy.data.materials[mat_name]
    mat = bpy.data.materials.new(mat_name)
    mat.use_nodes = True
    mat.blend_method = "BLEND"
    mat.use_screen_refraction = True
    nodes = mat.node_tree.nodes
    bsdf = nodes["Principled BSDF"]
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(img_path, check_existing=True)
    mat.node_tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    if "Alpha" in tex.outputs:
        mat.node_tree.links.new(tex.outputs["Alpha"], bsdf.inputs["Alpha"])
    bsdf.inputs["Roughness"].default_value = 0.5
    bsdf.inputs["Metallic"].default_value = 0.35
    return mat


rack_mat = make_material("Black_metal_rack", (0.01, 0.01, 0.01, 1), 0.35, 0.85)
dark_mat = make_material("Dark_inner_panel", (0.005, 0.005, 0.005, 1), 0.7)
ear_mat = make_material("Silver_Ear", (0.55, 0.55, 0.55, 1), 0.3, 0.9)
screw_mat = make_material("Screw", (0.45, 0.45, 0.45, 1), 0.25, 0.9)
blanking_mat = make_material("Blanking_Panel", (0.015, 0.015, 0.015, 1), 0.4, 0.3)
cable_mgmt_mat = make_material("Cable_Mgmt", (0.02, 0.02, 0.02, 1), 0.5, 0.3)
def make_rj45_material():
    """Translucent clear plastic for RJ45 housing."""
    name = "RJ45_Clear_Plastic"
    if name in bpy.data.materials:
        return bpy.data.materials[name]
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.blend_method = "BLEND"
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.82, 0.82, 0.78, 1)
    bsdf.inputs["Roughness"].default_value = 0.15
    bsdf.inputs["Metallic"].default_value = 0.0
    if "Transmission Weight" in bsdf.inputs:
        bsdf.inputs["Transmission Weight"].default_value = 0.4
    elif "Transmission" in bsdf.inputs:
        bsdf.inputs["Transmission"].default_value = 0.4
    bsdf.inputs["Alpha"].default_value = 0.85
    if "IOR" in bsdf.inputs:
        bsdf.inputs["IOR"].default_value = 1.45
    return mat

rj45_clear_mat = make_rj45_material()
rj45_tab_mat = make_material("RJ45_Tab", (0.75, 0.75, 0.72, 1), 0.2, 0.0)
rj45_pin_mat = make_material("RJ45_Pins", (0.85, 0.72, 0.2, 1), 0.3, 0.9)
led_green = make_emissive_material("LED_Green", (0.0, 0.8, 0.15, 1), 3.0)
led_amber = make_emissive_material("LED_Amber", (0.8, 0.5, 0.0, 1), 2.0)
floor_mat = make_material("DC_Floor", (0.08, 0.08, 0.08, 1), 0.7, 0.0)
ceiling_mat = make_material("DC_Ceiling", (0.7, 0.7, 0.7, 1), 0.6, 0.0)
wall_mat = make_material("DC_Wall", (0.15, 0.15, 0.16, 1), 0.65, 0.0)

# ==============================
# AUGMENTATION FUNCTIONS
# ==============================
def _adjust_labels_for_horizontal_flip(labels):
    updated = []
    for label in labels:
        parts = label.strip().split()
        if len(parts) != 5:
            updated.append(label)
            continue
        class_id, cx, cy, bw, bh = parts
        cx_new = 1.0 - float(cx)
        updated.append(f"{class_id} {cx_new:.6f} {cy} {bw} {bh}")
    return updated


def augment_image(image, labels_list):
    """
    Apply realistic, label-safe augmentations and return the adjusted image and labels.
    Only photometric transforms and horizontal flips are used so bounding boxes remain valid.
    """
    h, w = image.shape[:2]
    aug_labels = labels_list.copy()

    # Photometric adjustments that mimic camera exposure and indoor lighting.
    if random.random() < 0.75:
        brightness = random.uniform(-20, 20)
        contrast = random.uniform(0.9, 1.2)
        image = np.clip(image.astype(np.float32) * contrast + brightness, 0, 255).astype(np.uint8)

    if random.random() < 0.6:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.85, 1.15), 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * random.uniform(0.9, 1.1), 0, 255)
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    if random.random() < 0.5:
        temp_shift = random.uniform(-10, 10)
        image_f = image.astype(np.float32)
        image_f[:, :, 2] = np.clip(image_f[:, :, 2] + temp_shift, 0, 255)
        image_f[:, :, 0] = np.clip(image_f[:, :, 0] - temp_shift * 0.4, 0, 255)
        image = image_f.astype(np.uint8)

    # Natural sensor noise and lens effects.
    if random.random() < 0.5:
        noise_std = random.uniform(1.0, 4.0)
        noise = np.random.normal(0, noise_std, image.shape).astype(np.float32)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if random.random() < 0.4:
        kernel_size = random.choice([3, 5])
        image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    if random.random() < 0.35:
        cy, cx = h / 2, w / 2
        Y_grid, X_grid = np.ogrid[:h, :w]
        dist = np.sqrt((X_grid - cx) ** 2 + (Y_grid - cy) ** 2)
        max_dist = np.sqrt(cx ** 2 + cy ** 2)
        vignette = 1.0 - random.uniform(0.08, 0.22) * (dist / max_dist) ** 2
        image = np.clip(image.astype(np.float32) * vignette[:, :, np.newaxis], 0, 255).astype(np.uint8)

    if random.random() < 0.3:
        image = cv2.flip(image, 1)
        aug_labels = _adjust_labels_for_horizontal_flip(aug_labels)

    return image, aug_labels

# ==============================
# BOUNDING BOX PROJECTION
# ==============================
def project_3d_to_2d(pos_3d, scene, camera, image_width, image_height):
    """Project 3D world position to 2D image coordinates using Blender's camera projection."""
    if not isinstance(pos_3d, mathutils.Vector):
        try:
            pos_3d = mathutils.Vector((float(pos_3d[0]), float(pos_3d[1]), float(pos_3d[2])))
        except Exception as e:
            raise RuntimeError(f"Invalid 3D coordinate for projection: {pos_3d}") from e
    coord = world_to_camera_view(scene, camera, pos_3d)
    # coord.xyz are normalized camera-space coordinates
    if coord.z <= 0:
        return None
    if coord.x < 0 or coord.x > 1 or coord.y < 0 or coord.y > 1:
        return None
    x = coord.x * image_width
    y = (1.0 - coord.y) * image_height
    return (x, y)


def get_device_bbox_2d(device_info, scene, camera, img_width, img_height):
    """Calculate 2D bounding box for device by projecting 3D corners."""
    x, yf, z, w, h = device_info['x'], device_info['y_front'], device_info['z'], device_info['width'], device_info['height']
    corners_3d = [
        (x - w / 2, yf, z - h / 2),
        (x + w / 2, yf, z - h / 2),
        (x - w / 2, yf, z + h / 2),
        (x + w / 2, yf, z + h / 2),
    ]
    corners_2d = []
    for corner in corners_3d:
        proj = project_3d_to_2d(corner, scene, camera, img_width, img_height)
        if proj:
            corners_2d.append(proj)
    if not corners_2d:
        return None
    xs = [c[0] for c in corners_2d]
    ys = [c[1] for c in corners_2d]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    center_x = ((x_min + x_max) / 2) / img_width
    center_y = ((y_min + y_max) / 2) / img_height
    bbox_w = (x_max - x_min) / img_width
    bbox_h = (y_max - y_min) / img_height
    if bbox_w < 0.01 or bbox_h < 0.01:
        return None
    return (center_x, center_y, bbox_w, bbox_h)

def save_yolo_labels(devices, scene, camera, img_width, img_height, label_path):
    """Save YOLO format labels for all devices in image. Returns labels list."""
    labels = []
    for device in devices:
        if device['class'] is None:
            continue
        bbox = get_device_bbox_2d(device, scene, camera, img_width, img_height)
        if bbox:
            class_id = CLASSES[device['class']]
            center_x, center_y, bbox_w, bbox_h = bbox
            center_x = max(0, min(1, center_x))
            center_y = max(0, min(1, center_y))
            bbox_w = max(0.001, min(1, bbox_w))
            bbox_h = max(0.001, min(1, bbox_h))
            labels.append(f"{class_id} {center_x:.6f} {center_y:.6f} {bbox_w:.6f} {bbox_h:.6f}")
    with open(label_path, 'w') as f:
        for label in labels:
            f.write(label + '\n')
    return labels

# ==============================
# HELPERS
# ==============================
def cube(name, location, scale, material):
    bpy.ops.mesh.primitive_cube_add(size=1, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    if material:
        obj.data.materials.append(material)
    return obj


# ==============================
# DATACENTER ROOM ENVIRONMENT
# ==============================
def build_datacenter_room():
    room_w, room_d, room_h = 4.0, 6.0, 3.5
    cube("floor", (0, 0, -0.025), (room_w, room_d, 0.05), floor_mat)
    cube("ceiling", (0, 0, room_h), (room_w, room_d, 0.05), ceiling_mat)
    cube("wall_back", (0, room_d / 2, room_h / 2), (room_w, 0.05, room_h), wall_mat)
    cube("wall_left", (-room_w / 2, 0, room_h / 2), (0.05, room_d, room_h), wall_mat)
    cube("wall_right", (room_w / 2, 0, room_h / 2), (0.05, room_d, room_h), wall_mat)

build_datacenter_room()


# ==============================
# RACK FRAME
# ==============================
cube("front_left_post", (-0.27, -RACK_DEPTH / 2, RACK_HEIGHT / 2), (0.035, 0.035, RACK_HEIGHT), rack_mat)
cube("front_right_post", (0.27, -RACK_DEPTH / 2, RACK_HEIGHT / 2), (0.035, 0.035, RACK_HEIGHT), rack_mat)
cube("back_left_post", (-0.27, RACK_DEPTH / 2, RACK_HEIGHT / 2), (0.035, 0.035, RACK_HEIGHT), rack_mat)
cube("back_right_post", (0.27, RACK_DEPTH / 2, RACK_HEIGHT / 2), (0.035, 0.035, RACK_HEIGHT), rack_mat)
cube("mount_left_rail", (-0.24, -RACK_DEPTH / 2 + 0.05, RACK_HEIGHT / 2), (0.025, 0.025, RACK_HEIGHT), rack_mat)
cube("mount_right_rail", (0.24, -RACK_DEPTH / 2 + 0.05, RACK_HEIGHT / 2), (0.025, 0.025, RACK_HEIGHT), rack_mat)
cube("top_panel", (0, 0, RACK_HEIGHT), (RACK_WIDTH, RACK_DEPTH, 0.02), rack_mat)
cube("bottom_panel", (0, 0, 0), (RACK_WIDTH, RACK_DEPTH, 0.05), rack_mat)
cube("left_panel", (-RACK_WIDTH / 2 + 0.01, 0, RACK_HEIGHT / 2), (0.015, RACK_DEPTH, RACK_HEIGHT), rack_mat)
cube("right_panel", (RACK_WIDTH / 2 - 0.01, 0, RACK_HEIGHT / 2), (0.015, RACK_DEPTH, RACK_HEIGHT), rack_mat)
cube("back_panel", (0, RACK_DEPTH / 2 - 0.01, RACK_HEIGHT / 2), (RACK_WIDTH, 0.01, RACK_HEIGHT), dark_mat)

# Rail markings
for u in range(1, TOTAL_SLOTS + 1):
    z = START_Z + (u - 1) * U_HEIGHT + U_HEIGHT / 2
    if z > RACK_HEIGHT - 0.05:
        break
    for side in [-1, 1]:
        x = side * 0.245
        cube(f"rail_mark_{u}_{side}", (x, -RACK_DEPTH / 2 + 0.04, z), (0.008, 0.003, 0.002), ear_mat)
        if u % 3 == 0:
            cube(f"rail_screw_{u}_{side}", (x, -RACK_DEPTH / 2 + 0.035, z), (0.005, 0.005, 0.005), screw_mat)

# Dark world background (for areas not covered by room)
world = bpy.context.scene.world
world.use_nodes = True
bg_node = world.node_tree.nodes.get("Background")
bg_node.inputs["Color"].default_value = (0.002, 0.002, 0.003, 1)
bg_node.inputs["Strength"].default_value = 0.1

# ==============================
# RENDER SETTINGS — high quality
# ==============================
bpy.context.scene.render.engine = "BLENDER_EEVEE_NEXT"
bpy.context.scene.eevee.taa_render_samples = 64
bproc.camera.set_resolution(1024, 1400)

# Compositor passthrough
bpy.context.scene.use_nodes = True
comp_tree = bpy.context.scene.node_tree
for n in list(comp_tree.nodes):
    comp_tree.nodes.remove(n)
rl_node = comp_tree.nodes.new("CompositorNodeRLayers")
comp_out = comp_tree.nodes.new("CompositorNodeComposite")
comp_tree.links.new(rl_node.outputs[0], comp_out.inputs[0])


# ==============================
# PLACE DEVICE
# ==============================
def place_device(name, img_path, slot, device_u, used_slots, device_class=None):
    z = START_Z + slot * U_HEIGHT + (device_u * U_HEIGHT) / 2
    x, y = 0, Y_FRONT
    width = 0.44
    height = device_u * U_HEIGHT - 0.003
    depth = random.uniform(0.2, 0.55)

    for s in range(slot, slot + device_u):
        used_slots.add(s)

    cube(f"device_body_{name}", (x, y + depth / 2, z), (width, depth, height),
         make_material("dev_dark", (0.04, 0.04, 0.04, 1), 0.5, 0.3))

    bpy.ops.mesh.primitive_plane_add(size=1, location=(x, y - 0.001, z))
    face = bpy.context.object
    face.name = f"device_front_{name}"
    face.rotation_euler[0] = math.radians(90)
    face.dimensions = (width, height, 1)
    face.data.materials.append(create_device_material(img_path))

    ear_w, ear_h = 0.02, height - 0.003
    cube(f"ear_l_{name}", (-width / 2 - ear_w / 2, y, z), (ear_w, 0.002, ear_h), ear_mat)
    cube(f"ear_r_{name}", (width / 2 + ear_w / 2, y, z), (ear_w, 0.002, ear_h), ear_mat)
    cube(f"screw_l_{name}", (-width / 2 - ear_w / 2, y - 0.002, z), (0.006, 0.004, 0.006), screw_mat)
    cube(f"screw_r_{name}", (width / 2 + ear_w / 2, y - 0.002, z), (0.006, 0.004, 0.006), screw_mat)

    return {'x': x, 'y_front': y, 'z': z, 'width': width, 'height': height, 'texture_file': img_path, 'class': device_class}


# ==============================
# CABLES + RJ45 CONNECTORS
# ==============================
def create_cable(name, start, end, thickness=0.003, color=(0.55, 0.55, 0.55, 1)):
    """
    Cable exits port FORWARD, hangs DOWN vertically, enters destination port.
    Like real rack cables: out → down → in. Not random arcs.
    """
    curve_data = bpy.data.curves.new(name, type='CURVE')
    curve_data.dimensions = '3D'
    curve_data.bevel_depth = thickness
    curve_data.bevel_resolution = 1
    curve_data.fill_mode = 'FULL'

    spline = curve_data.splines.new('BEZIER')
    spline.bezier_points.add(3)  # 4 points: exit → hang forward → hang at dest X → enter

    forward_dist = random.uniform(0.012, 0.020)

    pts = [
        start,
        (start[0], start[1] - forward_dist, start[2]),
        (end[0], end[1] - forward_dist, end[2]),
        end,
    ]

    for i, co in enumerate(pts):
        bp = spline.bezier_points[i]
        bp.co = co
        bp.handle_left_type = 'AUTO'
        bp.handle_right_type = 'AUTO'

    obj = bpy.data.objects.new(name, curve_data)
    bpy.context.collection.objects.link(obj)
    mat = make_cable_material(f"cmat_{name}", color)
    obj.data.materials.append(mat)
    return obj


def place_rj45(name, port_pos, cable_color=(0.55, 0.55, 0.55, 1)):
    """
    Realistic RJ45 plugged into port.
    Most of the connector is INSIDE the port — only the boot sticks out.
    Returns cable attachment point.
    """
    x, y_face, z = port_pos
    # Body mostly inside port — only ~3mm sticks out
    cube(f"{name}_body", (x, y_face + 0.003, z), (0.011, 0.014, 0.008), rj45_clear_mat)
    # Tiny tab visible at top
    cube(f"{name}_tab", (x, y_face - 0.001, z + 0.005), (0.004, 0.006, 0.0015), rj45_tab_mat)
    # Boot / strain relief — the prominent visible part, colored like the cable
    boot_y = y_face - 0.008
    boot_mat = make_cable_material(f"boot_{name}", cable_color)
    cube(f"{name}_boot", (x, boot_y, z), (0.007, 0.010, 0.007), boot_mat)
    return (x, boot_y - 0.005, z)


def get_port_positions_3d(dev_info):
    """Returns port positions at the device face plane."""
    fname = dev_info.get('texture_file', '')
    x, yf, z, w, h = dev_info['x'], dev_info['y_front'], dev_info['z'], dev_info['width'], dev_info['height']
    basename = os.path.basename(fname) if fname else ''
    ports = port_locations.get(basename, [])
    y_face = yf - 0.001

    if ports:
        return [(x - w / 2 + p['cx'] * w, y_face, z + h / 2 - p['cy'] * h) for p in ports]
    else:
        num = random.randint(6, 16)
        return [(x - w * 0.38 + (i + 0.5) / num * w * 0.76,
                 y_face,
                 z + random.uniform(-h * 0.1, h * 0.1)) for i in range(num)]


def generate_cables_between(dev_a, dev_b, cable_idx_start):
    """Dense cables with mixed colors — fill most ports like real racks."""
    ports_a = get_port_positions_3d(dev_a)
    ports_b = get_port_positions_3d(dev_b)

    # Light: 10-25% of ports get cables — no overlapping
    occupancy = random.uniform(0.1, 0.25)
    num = max(2, int(len(ports_a) * occupancy))
    selected = random.sample(ports_a, min(num, len(ports_a)))

    # Mixed colors: pick 1-3 dominant colors, randomly assign
    num_colors = random.choices([1, 2, 3], weights=[0.3, 0.4, 0.3])[0]
    color_palette = random.sample(CABLE_COLORS, min(num_colors, len(CABLE_COLORS)))

    cables = []
    for c, port_a in enumerate(selected):
        cable_color = random.choice(color_palette)

        if ports_b:
            ep = random.choice(ports_b)
            port_b = (ep[0] + random.uniform(-0.003, 0.003), ep[1], ep[2])
        else:
            port_b = (port_a[0], dev_b['y_front'] - 0.001, dev_b['z'])

        idx = cable_idx_start + c
        cable_start = place_rj45(f"cable_rj45_s_{idx}", port_a, cable_color)
        cable_end = place_rj45(f"cable_rj45_e_{idx}", port_b, cable_color)
        thickness = random.uniform(0.0025, 0.0035)
        cables.append(create_cable(f"cable_{idx}", cable_start, cable_end, thickness, cable_color))

    return cables


# ==============================
# CABLE MANAGEMENT PANEL
# ==============================
def create_cable_mgmt(name, z, width=0.48, y=Y_FRONT):
    cube(f"{name}_bar", (0, y + 0.012, z), (width, 0.025, 0.035), cable_mgmt_mat)
    nf = random.randint(10, 18)
    fw = width / (nf * 2)
    for f in range(nf):
        fx = -width / 2 + (f * 2 + 1) * fw + fw / 2
        cube(f"{name}_f{f}", (fx, y - 0.004, z), (fw * 0.3, 0.007, 0.025), cable_mgmt_mat)


# ==============================
# LED INDICATORS
# ==============================
def add_leds(dev, idx):
    x, yf, z, w, h = dev['x'], dev['y_front'], dev['z'], dev['width'], dev['height']
    mats = [led_green, led_green, led_amber]
    for l in range(random.randint(1, 3)):
        lx = x - w / 2 + random.uniform(0.01, 0.05)
        lz = z + random.uniform(-h * 0.15, h * 0.15)
        cube(f"led_{idx}_{l}", (lx, yf - 0.003, lz), (0.003, 0.003, 0.003), random.choice(mats))


# ==============================
# DYNAMIC PREFIXES
# ==============================
DYNAMIC_PREFIXES = (
    "device", "shelf", "ear_l_", "ear_r_", "screw_l_", "screw_r_",
    "blanking", "cable", "patch", "led", "cmgmt", "pp_", "pdu", "srv",
    "ceil_light",
)


# ==============================
# GENERATION LOOP
# ==============================
for img_idx in range(NUM_IMAGES):
    print(f"\n=== Generating image {img_idx + 1}/{NUM_IMAGES} ===")

    # Cleanup
    for obj in list(bpy.context.scene.objects):
        if any(obj.name.startswith(p) for p in DYNAMIC_PREFIXES) or obj.type == 'LIGHT':
            bpy.data.objects.remove(obj, do_unlink=True)
    for crv in list(bpy.data.curves):
        if crv.users == 0:
            bpy.data.curves.remove(crv)

    used_slots = set()
    cable_idx = 0
    cable_devices = []  # only switches + patch panels get cables
    all_devices = []

    # =========================================================
    # FIXED DISTRIBUTION PER IMAGE:
    #   - Many switches (8-14)
    #   - Few patch panels (2-3)
    #   - Exactly 2 servers (2U each)
    #   - Exactly 2 PDUs (1U each)
    # =========================================================
    num_switches = random.randint(8, 14)
    num_patch = random.randint(2, 3)
    num_servers = 2
    num_pdus = 2

    # Build a placement order: patch panels near top, then switches,
    # servers in middle, PDUs near bottom
    current_slot = random.randint(0, 2)

    # ---- Patch panels (top of rack, each followed by a switch) ----
    for pp in range(num_patch):
        if current_slot >= TOTAL_SLOTS - 1:
            break
        if patch_panel_files and current_slot not in used_slots:
            info = place_device(f"pp_{pp}", random.choice(patch_panel_files), current_slot, 1, used_slots, 'patch_panel')
            all_devices.append(info)
            cable_devices.append(info)
            current_slot += 1

        # Cable management after patch panel (common — like real racks)
        if random.random() < 0.8 and current_slot < TOTAL_SLOTS and current_slot not in used_slots:
            used_slots.add(current_slot)
            create_cable_mgmt(f"cmgmt_{pp}", START_Z + current_slot * U_HEIGHT + U_HEIGHT / 2)
            current_slot += 1

        # Switch paired with this patch panel
        if switch_files and current_slot < TOTAL_SLOTS and current_slot not in used_slots:
            info = place_device(f"sw_pp_{pp}", random.choice(switch_files), current_slot, 1, used_slots, 'switch')
            all_devices.append(info)
            cable_devices.append(info)
            if random.random() < 0.6:
                add_leds(info, pp)
            current_slot += 1
            num_switches -= 1  # count against total

        current_slot += random.choices([0, 1], weights=[0.7, 0.3])[0]

    # ---- Remaining switches (bulk of the rack) ----
    for sw in range(max(0, num_switches)):
        if current_slot >= TOTAL_SLOTS:
            break
        if switch_files and current_slot not in used_slots:
            info = place_device(f"sw_{sw}", random.choice(switch_files), current_slot, 1, used_slots, 'switch')
            all_devices.append(info)
            cable_devices.append(info)
            if random.random() < 0.5:
                add_leds(info, 50 + sw)
            current_slot += 1
        current_slot += random.choices([0, 1], weights=[0.8, 0.2])[0]

    # ---- 2 Servers (2U each, NO cables) ----
    for s in range(num_servers):
        if current_slot + 1 >= TOTAL_SLOTS:
            break
        if any(sl in used_slots for sl in range(current_slot, current_slot + 2)):
            current_slot += 1
            if current_slot + 1 >= TOTAL_SLOTS:
                break
            if any(sl in used_slots for sl in range(current_slot, current_slot + 2)):
                current_slot += 2
                continue
        if server_files:
            info = place_device(f"srv_{s}", random.choice(server_files), current_slot, 2, used_slots, 'server')
            all_devices.append(info)
            if random.random() < 0.5:
                add_leds(info, 100 + s)
            current_slot += 2
        current_slot += random.choices([0, 1], weights=[0.7, 0.3])[0]

    # ---- 2 PDUs (1U each, NO cables, near bottom) ----
    for p in range(num_pdus):
        if current_slot >= TOTAL_SLOTS:
            break
        if current_slot not in used_slots and pdu_files:
            info = place_device(f"pdu_{p}", random.choice(pdu_files), current_slot, 1, used_slots, 'pdu')
            all_devices.append(info)
            current_slot += 1
        else:
            current_slot += 1

    # ---- Fill remaining empty slots with extra switches to hit 85%+ ----
    fill_target = int(TOTAL_SLOTS * 0.85)
    fill_attempts = 0
    while len(used_slots) < fill_target and fill_attempts < 50:
        fill_attempts += 1
        slot = random.randint(0, TOTAL_SLOTS - 1)
        if slot in used_slots:
            continue
        if switch_files:
            info = place_device(f"sw_fill_{fill_attempts}", random.choice(switch_files), slot, 1, used_slots, 'switch')
            all_devices.append(info)
            cable_devices.append(info)

    # ---- Blanking panels for few remaining empty slots ----
    for slot in range(TOTAL_SLOTS):
        if slot not in used_slots and random.random() < 0.5:
            z = START_Z + slot * U_HEIGHT + U_HEIGHT / 2
            cube(f"blanking_{slot}", (0, Y_FRONT, z), (0.48, 0.002, U_HEIGHT - 0.002), blanking_mat)

    print(f"  Rack fill: {len(used_slots)}/{TOTAL_SLOTS} slots ({100*len(used_slots)//TOTAL_SLOTS}%)")

    # ---- Cables: ONLY between adjacent switches/patch panels (short runs) ----
    cable_devices_sorted = sorted(cable_devices, key=lambda d: d['z'])
    for i in range(len(cable_devices_sorted) - 1):
        dev_a = cable_devices_sorted[i]
        dev_b = cable_devices_sorted[i + 1]
        # Only connect devices within ~3U of each other (short runs)
        if abs(dev_a['z'] - dev_b['z']) < 0.15:
            cables = generate_cables_between(dev_a, dev_b, cable_idx)
            cable_idx += len(cables)

    # ---- LIGHTING: datacenter ceiling fluorescents ----
    # Main overhead strip lights
    for lx in [-0.3, 0.3]:
        light = bproc.types.Light()
        light.set_type("AREA")
        light.set_location([lx, random.uniform(-0.5, 0.0), 3.2])
        light.set_rotation_euler([0, 0, 0])
        light.set_energy(random.uniform(150, 300))
        light.set_color([0.95, 0.97, 1.0])

        # Visible fluorescent tube geometry
        tube = cube(f"ceil_light_tube_{lx}", (lx, -0.3, 3.15), (0.12, 1.2, 0.02),
                    make_emissive_material(f"fluor_{lx}", (0.95, 0.97, 1.0, 1), 2.0))

    # Fill light from behind camera
    fill = bproc.types.Light()
    fill.set_type("AREA")
    fill.set_location([0, random.uniform(-2.5, -1.8), random.uniform(1.5, 2.5)])
    fill.set_rotation_euler([math.radians(20), 0, 0])
    fill.set_energy(random.uniform(30, 70))
    fill.set_color([1.0, 0.98, 0.95])

    # Subtle ambient from side
    side = bproc.types.Light()
    side.set_type("POINT")
    side.set_location([random.choice([-1.5, 1.5]), random.uniform(-1, 0.5), random.uniform(0.5, 2.0)])
    side.set_energy(random.uniform(8, 25))
    side.set_color([0.9, 0.92, 1.0])

    # ---- CAMERA: realistic focal length + depth of field ----
    cam_x = random.uniform(-0.04, 0.04)
    cam_y = random.uniform(-1.2, -1.6)
    cam_z = random.uniform(0.6, 1.4)
    cam_loc = [cam_x, cam_y, cam_z]

    tgt_x = random.uniform(-0.02, 0.02)
    tgt_y = -RACK_DEPTH / 2
    tgt_z = cam_z + random.uniform(-0.05, 0.05)

    rot = bproc.camera.rotation_from_forward_vec(
        [tgt_x - cam_loc[0], tgt_y - cam_loc[1], tgt_z - cam_loc[2]]
    )
    bproc.utility.reset_keyframes()
    bproc.camera.add_camera_pose(bproc.math.build_transformation_mat(cam_loc, rot))

    # Depth of field
    cam = bpy.context.scene.camera.data
    cam.dof.use_dof = True
    cam.dof.focus_distance = abs(cam_y - tgt_y)
    cam.dof.aperture_fstop = random.uniform(2.8, 5.6)
    cam.lens = random.uniform(35, 50)

    # Subtle lens distortion per frame
    # ---- RENDER ----
    data = bproc.renderer.render()
    color = data["colors"][0]

    # Post-processing: camera noise
    noise_std = random.uniform(2, 5)
    noise = np.random.normal(0, noise_std, color.shape).astype(np.float32)
    color = np.clip(color.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Post-processing: vignette
    h, w = color.shape[:2]
    Y_grid, X_grid = np.ogrid[:h, :w]
    cy, cx = h / 2, w / 2
    dist = np.sqrt((X_grid - cx) ** 2 + (Y_grid - cy) ** 2)
    max_dist = np.sqrt(cx ** 2 + cy ** 2)
    vig = 1.0 - random.uniform(0.12, 0.28) * (dist / max_dist) ** 2
    color = np.clip(color * vig[:, :, np.newaxis], 0, 255).astype(np.uint8)

    # Post-processing: slight color temperature shift
    temp_shift = random.uniform(-8, 8)
    color_f = color.astype(np.float32)
    color_f[:, :, 2] = np.clip(color_f[:, :, 2] + temp_shift, 0, 255)  # R
    color_f[:, :, 0] = np.clip(color_f[:, :, 0] - temp_shift * 0.5, 0, 255)  # B
    color = color_f.astype(np.uint8)

    output_png = os.path.join(OUTPUT_DIR, f"synthetic_rack_{img_idx:03d}.png")
    cv2.imwrite(output_png, color[:, :, ::-1])
    print(f"  Saved {output_png}")

    # Generate YOLO format labels (before augmentation)
    scene = bpy.context.scene
    camera = scene.camera
    img_width = scene.render.resolution_x
    img_height = scene.render.resolution_y
    label_path = os.path.join(LABELS_DIR, f"synthetic_rack_{img_idx:03d}.txt")
    labels = save_yolo_labels(all_devices, scene, camera, img_width, img_height, label_path)
    print(f"  Saved {label_path} ({len(labels)} labels)")

    # Apply augmentation to generate additional variations
    if labels:
        num_augmentations = random.randint(2, 4)  # Generate 2-4 augmented versions per image
        for aug_idx in range(num_augmentations):
            aug_img, aug_labels = augment_image(color[:, :, ::-1], labels)
            
            # Save augmented image
            aug_img_path = os.path.join(OUTPUT_DIR, f"synthetic_rack_{img_idx:03d}_aug{aug_idx}.png")
            cv2.imwrite(aug_img_path, aug_img)
            
            # Save augmented labels
            aug_label_path = os.path.join(LABELS_DIR, f"synthetic_rack_{img_idx:03d}_aug{aug_idx}.txt")
            with open(aug_label_path, 'w') as f:
                for label in aug_labels:
                    f.write(label + '\n')
            print(f"  Saved augmented {img_idx:03d}_aug{aug_idx}")

print("\nDONE — all images generated.")
