import os, sys
# CAD recode imports
import pickle
import random
from collections import deque

import numpy as np
import torch
from torch.utils.data import Dataset

from metrics_async import transform_real_mesh

os.environ["PYGLET_HEADLESS"] = "True"

import trimesh
from PIL import Image, ImageOps
import skimage

sys.path.append("/home/jovyan/users/zhemchuzhnikov/miniconda3/envs/zhemchuzhnikov/lib/python3.10/site-packages")
import open3d


def render_mesh(mesh, camera_distance=-1.8, front=[1, 1, 1],
                width=500, height=500, img_size=128):
    vis = open3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)
    vis.add_geometry(mesh)

    lookat = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    front_array = np.array(front, dtype=np.float32)
    up = np.array([0, 1, 0], dtype=np.float32)

    eye = lookat + front_array * camera_distance
    right = np.cross(up, front_array)
    right /= np.linalg.norm(right)
    true_up = np.cross(front_array, right)
    rotation_matrix = np.column_stack((right, true_up, front_array)).T
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = rotation_matrix
    extrinsic[:3, 3] = -rotation_matrix @ eye

    view_control = vis.get_view_control()
    camera_params = view_control.convert_to_pinhole_camera_parameters()
    camera_params.extrinsic = extrinsic
    view_control.convert_from_pinhole_camera_parameters(camera_params)

    vis.poll_events()
    vis.update_renderer()
    image = vis.capture_screen_float_buffer(do_render=True)
    vis.destroy_window()

    image = np.asarray(image)
    image = (image * 255).astype(np.uint8)
    image = skimage.transform.resize(
        image,
        output_shape=(img_size, img_size),
        order=2,
        anti_aliasing=True,
        preserve_range=True).astype(np.uint8)

    return Image.fromarray(image)



class CadrilleSTLDataset(Dataset):
    def __init__(self, path, file_name, n_points=256, mode='pc',
                 img_size=128, noise_scale_pc=None, size=None):
        super().__init__()
        self.n_points = n_points
        self.path = path
        self.img_size = img_size
        self.noise_scale_pc = noise_scale_pc
        if mode != 'swap':
            self.mode = mode
            self.next_mode = mode
        else:
            self.mode = "pc"
            self.next_mode = "img"
        self.size = size

        with open(os.path.join(path, file_name), 'rb') as f:
            self.annotations = pickle.load(f)
        if self.size is None:
            self.size = len(self.annotations)

    def swap(self):
        self.mode, self.next_mode = self.next_mode, self.mode

    def __len__(self):
        return min(len(self.annotations), self.size)

    def __getitem__(self, idx):
        # try:
        mesh_path = os.path.join(self.path, self.annotations[idx]['mesh_path'])
        mesh = trimesh.load_mesh(mesh_path)
        mesh = transform_real_mesh(mesh)

        if self.mode == 'pc':
            input_item = self.get_point_cloud(mesh)
        elif self.mode == 'img':
            input_item = self.get_img(mesh)
        elif self.mode == 'pc_img':
            if np.random.rand() < 0.5:
                input_item = self.get_point_cloud(mesh)
            else:
                input_item = self.get_img(mesh)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        input_item['mesh_path'] = mesh_path
        input_item['mesh'] = mesh
        input_item['idx'] = idx

        return input_item

    # except:
    #     return self[(idx + 1) % len(self)]

    def get_img(self, mesh):
        mesh.apply_transform(trimesh.transformations.scale_matrix(1 / 2))
        mesh.apply_transform(trimesh.transformations.translation_matrix([0.5, 0.5, 0.5]))

        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.faces)
        mesh = open3d.geometry.TriangleMesh()
        mesh.vertices = open3d.utility.Vector3dVector(vertices)
        mesh.triangles = open3d.utility.Vector3iVector(faces)
        mesh.paint_uniform_color(np.array([255, 255, 136]) / 255.0)
        mesh.compute_vertex_normals()

        fronts = [[1, 1, 1], [-1, -1, -1], [-1, 1, -1], [1, -1, 1]]
        images = []
        for front in fronts:
            image = render_mesh(mesh, camera_distance=-0.9,
                                front=front, img_size=self.img_size)
            images.append(image)

        images = [ImageOps.expand(image, border=3, fill='black') for image in images]
        images = [Image.fromarray(np.vstack((np.hstack((np.array(images[0]), np.array(images[1]))),
                                             np.hstack((np.array(images[2]), np.array(images[3]))))))]
        # import time
        # os.makedirs("/home/jovyan/tarasov/imgs", exist_ok=True)
        # images[0].save(f"/home/jovyan/tarasov/imgs/{time.time()}.png")
        input_item = {
            'video': images,
            'description': 'Generate cadquery code',
        }
        return input_item