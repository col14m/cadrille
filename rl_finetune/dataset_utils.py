import os
# CAD recode imports
import pickle
import random
from collections import deque

import numpy as np
import torch
from torch.utils.data import Dataset

from utils import transform_real_mesh

os.environ["PYGLET_HEADLESS"] = "True"

import trimesh
from PIL import Image, ImageOps
import skimage
import open3d
from pytorch3d.ops import sample_farthest_points


def mesh_to_point_cloud(mesh, n_points=256, n_pre_points=8192):
    vertices, _ = trimesh.sample.sample_surface(mesh, n_pre_points)
    _, ids = sample_farthest_points(torch.tensor(vertices).unsqueeze(0), K=n_points)
    ids = ids[0].numpy()
    return np.asarray(vertices[ids])


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


class TrainDataset(Dataset):
    def __init__(self, path, file_name, tokenizer, n_points, normalize_std):
        super().__init__()
        self.path = path
        self.tokenizer = tokenizer
        self.n_points = n_points
        self.normalize_std = normalize_std
        with open(os.path.join(path, file_name), 'rb') as f:
            self.annotations = pickle.load(f)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # -> dict with input_ids of py_string, attention_mask, point_cloud
        item = self.annotations[index]
        mesh = trimesh.load(os.path.join(self.path, item['mesh_path']))
        with open(os.path.join(self.path, item['py_path']), 'r') as f:
            py_string = f.read()
        item = self.tokenizer('<|im_start|>' + py_string + '<|endoftext|>')
        item['input_ids'] = [self.tokenizer.pad_token_id] * self.n_points + item['input_ids']
        item['attention_mask'] = [-1] * self.n_points + item['attention_mask']
        point_cloud = mesh_to_point_cloud(mesh, self.n_points)
        point_cloud[:, :3] = point_cloud[:, :3] / self.normalize_std
        item['point_cloud'] = point_cloud.astype(np.float32)
        item['mesh'] = mesh
        return item


class TrainRLDataset(Dataset):
    def __init__(self, path, file_name, tokenizer, n_points, normalize_std):
        super().__init__()
        self.path = path
        self.tokenizer = tokenizer
        self.n_points = n_points
        self.normalize_std = normalize_std
        with open(os.path.join(path, file_name), 'rb') as f:
            self.annotations = pickle.load(f)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # -> dict with input_ids of py_string, attention_mask, point_cloud
        item = self.annotations[index]
        mesh = trimesh.load(os.path.join(self.path, item['mesh_path']))
        with open(os.path.join(self.path, item['py_path']), 'r') as f:
            py_string = f.read()
        item = self.tokenizer('<|im_start|>')
        item['input_ids'] = [self.tokenizer.pad_token_id] * self.n_points + item['input_ids']
        item['attention_mask'] = [-1] * self.n_points + item['attention_mask']
        item['py_string'] = py_string
        point_cloud = mesh_to_point_cloud(mesh, self.n_points)
        point_cloud[:, :3] = point_cloud[:, :3] / self.normalize_std
        item['point_cloud'] = point_cloud.astype(np.float32)
        item['mesh'] = mesh
        return item


class TrainDPODataset(Dataset):
    def __init__(self, path, file_name, tokenizer, n_points, normalize_std):
        super().__init__()
        self.path = path
        self.tokenizer = tokenizer
        self.n_points = n_points
        self.normalize_std = normalize_std
        with open(os.path.join(path, file_name), 'rb') as f:
            self.annotations = pickle.load(f)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # -> dict with input_ids of py_string, attention_mask, point_cloud
        orig_item = self.annotations[index]
        mesh = trimesh.load(os.path.join(self.path, orig_item['mesh_path']))
        with open(os.path.join(self.path, orig_item['py_path']), 'r') as f:
            py_string = f.read()
        with open(os.path.join(self.path, orig_item['ious']), 'rb') as f:
            ious = np.load(f)
        w, l = random.sample(range(len(ious)), 2)
        if ious[l] > ious[w]:
            w, l = l, w
        # print("W", ious[w], "L", ious[l])
        item = self.tokenizer('<|im_start|>')
        item['input_ids'] = [self.tokenizer.pad_token_id] * self.n_points + item['input_ids']
        item['attention_mask'] = [-1] * self.n_points + item['attention_mask']
        item['py_string'] = py_string
        with open(os.path.join(self.path, orig_item[w]), 'r') as f:
            item['py_string_w'] = f.read() + '<|endoftext|>'
        with open(os.path.join(self.path, orig_item[l]), 'r') as f:
            item['py_string_l'] = f.read() + '<|endoftext|>'
        point_cloud = mesh_to_point_cloud(mesh, self.n_points)
        point_cloud[:, :3] = point_cloud[:, :3] / self.normalize_std
        item['point_cloud'] = point_cloud.astype(np.float32)
        item['ious'] = ious
        item['mesh'] = mesh
        return item


class FilteredDataset(Dataset):
    def __init__(self, dataset, filtered_idxs):
        super().__init__()
        self.dataset = dataset
        self.filtered_idxs = filtered_idxs

    def __len__(self):
        return len(self.filtered_idxs)

    def __getitem__(self, index):
        # -> dict with input_ids of py_string, attention_mask, point_cloud
        return self.dataset[self.filtered_idxs[index]]


class RealDataset(Dataset):
    def __init__(self, path, file_name, tokenizer, n_points=256):
        super().__init__()
        self.start_token_id = tokenizer('<|im_start|>')['input_ids'][0]
        self.pad_token_id = tokenizer.pad_token_id
        self.n_points = n_points
        self.path = path

        with open(os.path.join(path, file_name), 'rb') as f:
            self.annotations = pickle.load(f)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        try:
            mesh_path = os.path.join(self.path, self.annotations[idx]['mesh_path'])
            mesh = trimesh.load_mesh(mesh_path)
            mesh = transform_real_mesh(mesh)

            vertices = mesh_to_point_cloud(mesh, self.n_points)
            point_cloud = vertices
            input_ids = [self.pad_token_id] * self.n_points + [self.start_token_id]
            attention_mask = [-1] * self.n_points + [1]
            return {
                "point_cloud": point_cloud.astype(np.float32),
                "input_ids": np.array(input_ids),
                "attention_mask": np.array(attention_mask),
                "mesh_path": mesh_path,
                "mesh": mesh
            }
        except:
            return self[(idx + 1) % len(self)]


class RealDPODataset(Dataset):
    def __init__(self, path, file_name, tokenizer, n_points):
        super().__init__()
        self.path = path
        self.tokenizer = tokenizer
        self.n_points = n_points
        with open(os.path.join(path, file_name), 'rb') as f:
            self.annotations = pickle.load(f)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # -> dict with input_ids of py_string, attention_mask, point_cloud
        orig_item = self.annotations[index]
        # print(orig_item)
        mesh_path = os.path.join(self.path, self.annotations[index]['mesh_path'].replace('data/deepcad_val/', ''))
        mesh = trimesh.load_mesh(mesh_path)
        mesh = transform_real_mesh(mesh)

        with open(os.path.join(self.path, orig_item['ious']), 'rb') as f:
            ious = np.load(f)
        w, l = random.sample(range(len(ious)), 2)
        if ious[l] > ious[w]:
            w, l = l, w
        # print("W", ious[w], "L", ious[l])
        item = self.tokenizer('<|im_start|>')
        item['input_ids'] = [self.tokenizer.pad_token_id] * self.n_points + item['input_ids']
        item['attention_mask'] = [-1] * self.n_points + item['attention_mask']
        item['mesh'] = mesh
        with open(os.path.join(self.path, orig_item[w]), 'r') as f:
            item['py_string_w'] = f.read() + '<|endoftext|>'
        with open(os.path.join(self.path, orig_item[l]), 'r') as f:
            item['py_string_l'] = f.read() + '<|endoftext|>'
        vertices = mesh_to_point_cloud(mesh, self.n_points)
        point_cloud = vertices
        item['point_cloud'] = point_cloud.astype(np.float32)
        item['ious'] = ious
        return item


class RealDatasetMM(Dataset):
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

    def get_point_cloud(self, mesh):
        mesh = self._augment_pc(mesh)
        point_cloud = mesh_to_point_cloud(mesh, self.n_points)

        input_item = {
            'point_cloud': point_cloud,
            'description': 'Generate cadquery code',
        }
        return input_item

    def _augment_pc(self, mesh):
        if self.noise_scale_pc is not None and np.random.rand() < 0.5:
            mesh.vertices += np.random.normal(loc=0, scale=self.noise_scale_pc, size=mesh.vertices.shape)
        return mesh


class RealDPODatasetMM(Dataset):
    def __init__(self, path, file_name, tokenizer, n_points=256, mode='pc',
                 img_size=128, noise_scale_pc=None, size=None):
        super().__init__()
        self.n_points = n_points
        self.path = path
        self.img_size = img_size
        self.noise_scale_pc = noise_scale_pc
        self.tokenizer = tokenizer
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
        # -> dict with input_ids of py_string, attention_mask, point_cloud
        orig_item = self.annotations[idx]
        # print(orig_item)
        mesh_path = os.path.join(self.path, self.annotations[idx]['mesh_path'])
        mesh = trimesh.load_mesh(mesh_path)
        mesh = transform_real_mesh(mesh)

        with open(os.path.join(self.path, orig_item['ious']), 'rb') as f:
            ious = np.load(f)
        w, l = random.sample(range(len(ious)), 2)
        if ious[l] > ious[w]:
            w, l = l, w
        # print("W", ious[w], "L", ious[l])
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

        with open(os.path.join(self.path, orig_item[w]), 'r') as f:
            input_item['py_string_w'] = f.read() + self.tokenizer.eos_token
        with open(os.path.join(self.path, orig_item[l]), 'r') as f:
            input_item['py_string_l'] = f.read() + self.tokenizer.eos_token

        input_item['mesh_path'] = mesh_path
        input_item['mesh'] = mesh
        input_item['idx'] = idx

        return input_item

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

        input_item = {
            'video': images,
            'description': 'Generate cadquery code',
        }
        return input_item

    def get_point_cloud(self, mesh):
        mesh = self._augment_pc(mesh)
        point_cloud = mesh_to_point_cloud(mesh, self.n_points)

        input_item = {
            'point_cloud': point_cloud,
            'description': 'Generate cadquery code',
        }
        return input_item

    def _augment_pc(self, mesh):
        if self.noise_scale_pc is not None and np.random.rand() < 0.5:
            mesh.vertices += np.random.normal(loc=0, scale=self.noise_scale_pc, size=mesh.vertices.shape)
        return mesh


class Text2CADDataset(Dataset):
    def __init__(self, path, file_name, idx_offset=0, n_samples=None):
        super().__init__()
        self.path = path
        self.n_samples = n_samples
        self.idx_offset = idx_offset
        with open(os.path.join(path, file_name), 'rb') as f:
            self.annotations = pickle.load(f)

    def __len__(self):
        return self.n_samples if self.n_samples is not None else len(self.annotations)

    def __getitem__(self, index):
        item = self.annotations[index]

        mesh_path = os.path.join(self.path, item['uid'] + '.stl')
        mesh = trimesh.load_mesh(mesh_path)
        mesh = transform_real_mesh(mesh)

        input_item = {
            'description': item['description'],
            'file_name': item['uid'],
            'mesh': mesh,
            'mesh_path': mesh_path,
            'idx': index + self.idx_offset,
        }

        return input_item


class IndexBuffer:
    def __init__(self, max_size=200):
        self.buffer = deque()
        self.max_size = max_size

    def add(self, index):
        """Add a single index to the buffer."""
        self.buffer.append(index)
        self._enforce_max_size()

    def add_many(self, indices):
        """Add multiple indices to the buffer."""
        self.buffer.extend(indices)
        self._enforce_max_size()

    def sample(self, n):
        """Randomly sample n indices from the buffer."""
        if n > len(self.buffer):
            raise ValueError("Not enough elements in the buffer to sample.")
        return random.sample(self.buffer, n)

    def _enforce_max_size(self):
        """Ensure buffer doesn't exceed max size."""
        if self.max_size is not None:
            while len(self.buffer) > self.max_size:
                self.buffer.popleft()

    def __len__(self):
        return len(self.buffer)

    def __repr__(self):
        return f"IndexBuffer({list(self.buffer)})"
