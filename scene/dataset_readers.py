#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import math
import torch
import cv2
from utils import param_utils

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    fx: float
    fy: float
    cx: float
    cy: float
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image=Image.open(image_path)
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              fx=intr.params[0], fy=intr.params[1], cx=intr.params[2], cy=intr.params[3], 
                            image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8, testidx=[0], ply_name="points3D.ply"):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx not in testidx]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx in testidx]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/"+ply_name)
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            matrix = np.linalg.inv(np.array(frame["transform_matrix"]))
            R = -np.transpose(matrix[:3,:3])
            R[:,0] = -R[:,0]
            T = -matrix[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def sample_gaussians(path, sample_size):
    import trimesh
    mesh = trimesh.load(path, process=False, maintain_order=True)
    points = mesh.sample(sample_size)
    points_colors = torch.rand(points.shape)
    # return points, points_colors
    return BasicPointCloud(points=points, colors=points_colors, normals=None)

def get_cameras(params, images_folder, brics_type):
    new_params = []
    # skips the bottom side cameras due to reflections
    if brics_type == "sbc":
        skip_images = [
            "brics-sbc-003_cam0",
            "brics-sbc-003_cam1",
            "brics-sbc-004_cam0",
            "brics-sbc-008_cam0",
            "brics-sbc-008_cam1",
            "brics-sbc-009_cam0",
            "brics-sbc-013_cam0",
            "brics-sbc-013_cam1",
            "brics-sbc-014_cam0",
            "brics-sbc-018_cam0",
            "brics-sbc-018_cam1",
            "brics-sbc-019_cam0",
        ]
    else:
        skip_images = [
            "brics-odroid-003_cam0",
            "brics-odroid-003_cam1",
            "brics-odroid-004_cam0",
            "brics-odroid-008_cam0",
            "brics-odroid-008_cam1",
            "brics-odroid-009_cam0",
            "brics-odroid-013_cam0",
            "brics-odroid-013_cam1",
            "brics-odroid-014_cam0",
            "brics-odroid-018_cam0",
            "brics-odroid-018_cam1",
            "brics-odroid-019_cam0",
        ]

    for idx, param in enumerate(params):
        cam_name = param["cam_name"]

        if cam_name in skip_images:
            continue

        img_dir = os.path.join(images_folder, cam_name)
        if not os.path.exists(img_dir):
            continue
            
        new_params.append(param)
    return new_params

def readBricsCameras(params_path, images_folder, test_camera, frame_idx, brics_type="sbc"):
    params = param_utils.read_params(params_path)
    params = get_cameras(params, images_folder, brics_type)

    train_cam_infos = []
    test_cam_infos = []
    for idx, cam in enumerate(params):
        
        cam_name = cam["cam_name"]

        img_dir = os.path.join(images_folder, cam_name)
        imgs = os.listdir(img_dir)

        img = imgs[frame_idx]
        img_path = os.path.join(img_dir, img)
        img_name = os.path.basename(img_path).split(".")[0]
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        extr = param_utils.get_extr(cam)
        K, dist = param_utils.get_intr(cam)
        w, h = cam["width"], cam["height"]
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        fovx = 2 * math.atan(w / (2 * fx))
        fovy = 2 * math.atan(h / (2 * fy))
        
        R = np.transpose(extr[:, :3])
        T = extr[:, 3]

        # handles alpha channel if there's segmentation
        if img.shape[-1] == 4:
            b, g, r, alpha = cv2.split(img)

            rgb = np.stack([r, g, b, alpha], axis=-1)
            alpha = alpha[..., np.newaxis] / 255.0

            rgb = rgb / 255.0
            rgb = rgb * alpha
        else:
            b, g, r = cv2.split(img)
            rgb = np.stack([r, g, b], axis=-1)
            rgb = rgb / 255.0

        image = Image.fromarray(np.uint8(rgb*255))

        cam_info = CameraInfo(uid=cam["cam_id"], R=R, T=T, FovY=fovy, FovX=fovx, fx=fx, fy=fy, cx=cx, cy=cy, image=image,
                            image_path=img_path, image_name=img_name, width=int(w), height=int(h))
        if cam_name == test_camera:
            test_cam_infos.append(cam_info)
        else:
            train_cam_infos.append(cam_info)

    return train_cam_infos, test_cam_infos

def readBricsSceneInfo(path, test_camera, frame_idx, brics_type="sbc", ply_path="./mesh.ply"):
    params_path = os.path.join(path, "optim_params.txt")
    
    if brics_type == "sbc":
        images_folder = os.path.join(path, "data", "segmented_ngp")
    else:
        images_folder = os.path.join(path, "data", "mask")
    
    print(f"Loading frame {frame_idx} and test camera {test_camera}")
    
    train_cam_infos, test_cam_infos = readBricsCameras(params_path, images_folder, test_camera, frame_idx, brics_type=brics_type)

    nerf_normalization = getNerfppNorm(train_cam_infos)

    pcd = sample_gaussians(ply_path, 10000)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readBProcessedCamera(images_dir, idx, cam_params, test_camera, brics_type="sbc"):
    if brics_type == "odroid":
        cams_to_keep = ['brics-odroid-001_cam0', 'brics-odroid-001_cam1', 'brics-odroid-002_cam0', 'brics-odroid-006_cam0',
                        'brics-odroid-007_cam0', 'brics-odroid-007_cam1', 'brics-odroid-009_cam1', 'brics-odroid-010_cam0',
                        'brics-odroid-010_cam1', 'brics-odroid-011_cam0', 'brics-odroid-012_cam0', 'brics-odroid-012_cam1',
                        'brics-odroid-014_cam1', 'brics-odroid-015_cam0', 'brics-odroid-015_cam1', 'brics-odroid-016_cam0',
                        'brics-odroid-017_cam0', 'brics-odroid-017_cam1', 'brics-odroid-020_cam0', 'brics-odroid-020_cam1', 
                        'brics-odroid-021_cam0', 'brics-odroid-021_cam1', 'brics-odroid-022_cam0', 'brics-odroid-022_cam1', 
                        'brics-odroid-023_cam0', 'brics-odroid-024_cam0', 'brics-odroid-024_cam1', 'brics-odroid-025_cam0', 
                        'brics-odroid-025_cam1']
    else:
        cams_to_keep = ['brics-sbc-001_cam0', 'brics-sbc-001_cam1', 'brics-sbc-002_cam0', 'brics-sbc-006_cam0',
                        'brics-sbc-007_cam0', 'brics-sbc-007_cam1', 'brics-sbc-009_cam1', 'brics-sbc-010_cam0',
                        'brics-sbc-010_cam1', 'brics-sbc-011_cam0', 'brics-sbc-012_cam0', 'brics-sbc-012_cam1',
                        'brics-sbc-014_cam1', 'brics-sbc-015_cam0', 'brics-sbc-015_cam1', 'brics-sbc-016_cam0',
                        'brics-sbc-017_cam0', 'brics-sbc-017_cam1', 'brics-sbc-020_cam0', 'brics-sbc-020_cam1', 
                        'brics-sbc-021_cam0', 'brics-sbc-021_cam1', 'brics-sbc-022_cam0', 'brics-sbc-022_cam1', 
                        'brics-sbc-023_cam0', 'brics-sbc-024_cam0', 'brics-sbc-024_cam1', 'brics-sbc-025_cam0', 
                        'brics-sbc-025_cam1', 'brics-sbc-005_cam0']

    train_cam_infos = []
    test_cam_infos = []
    img_frames = os.listdir(images_dir)

    cam_keys = list(cam_params.keys())
    
    images_idx_dir = os.path.join(images_dir, img_frames[idx])
    cams_idx = cam_params[cam_keys[idx]]
    
    for cam_idx, cam_name_path in enumerate(os.listdir(images_idx_dir)):
        cam_name = cam_name_path.split('.')[0]
        if cam_name not in cams_to_keep:
            # print("Skipping camera: ", cam_name)
            continue

        img_path = os.path.join(images_idx_dir, cam_name_path)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        # handles alpha channel if there's segmentation
        if img.shape[-1] == 4:
            b, g, r, alpha = cv2.split(img)

            rgb = np.stack([r, g, b, alpha], axis=-1)
            # alpha = alpha[..., np.newaxis] / 255.0
            bg = np.array([1, 1, 1])

            rgb = rgb / 255.0
            rgb = rgb[:,:,:3] * rgb[:, :, 3:4] + bg * (1 - rgb[:, :, 3:4])
            # rgb = rgb * alpha
        else:
            # alpha, _, _ = cv2.split(mask)
            b, g, r = cv2.split(img)
            rgb = np.stack([r, g, b], axis=-1)
            rgb = rgb / 255.0

        image = Image.fromarray(np.uint8(rgb*255))
        width, height = image.size

        K = np.array(cams_idx[cam_name]['K'])
        R = np.array(cams_idx[cam_name]['R'])
        T = np.array(cams_idx[cam_name]['T'])
        fx, fy = K[1, 1], K[0, 0]
        cx, cy = K[0, 2], K[1, 2]

        fovy = focal2fov(fx, height)
        fovx = focal2fov(fy, width)

        cam_info = CameraInfo(uid=cam_idx, R=R, T=T, FovY=fovy, FovX=fovx, fx=fx, fy=fy, cx=cx, cy=cy, image=image,
                            image_path=img_path, image_name=cam_name_path, width=int(width), height=int(height))
        if cam_name == test_camera:
            test_cam_infos.append(cam_info)
        else:
            train_cam_infos.append(cam_info)
    return train_cam_infos, test_cam_infos

def readBProcessedSceneInfo(path, test_camera, frame_idx, brics_type="sbc", ply_path="./mesh.ply"):
    images_dir = os.path.join(path, "images")
    params_path = os.path.join(path, "cam_params.json")

    with open(params_path, 'r') as f:
        params = json.load(f)

    train_cam_infos, test_cam_infos = readBProcessedCamera(images_dir, int(frame_idx), params, test_camera, brics_type)

    nerf_normalization = getNerfppNorm(train_cam_infos)

    pcd = sample_gaussians(ply_path, 10000)


    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "brics": readBricsSceneInfo,
    "brics-processed": readBProcessedSceneInfo
}