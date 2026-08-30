"""Triangle-mesh oracle carrier using deterministic ray casting."""

from __future__ import annotations

import numpy as np
import torch

from .base import Camera, ProjectionTable, SparseAdjacency, SurfaceCarrier


class MeshCarrier(SurfaceCarrier):
    def __init__(self, vertices: torch.Tensor, triangles: torch.Tensor) -> None:
        vertices = torch.as_tensor(vertices, dtype=torch.float32).cpu()
        triangles = torch.as_tensor(triangles, dtype=torch.long).cpu()
        if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] == 0:
            raise ValueError("vertices must have shape [V, 3]")
        if triangles.ndim != 2 or triangles.shape[1] != 3 or triangles.shape[0] == 0:
            raise ValueError("triangles must have shape [F, 3]")
        if int(triangles.min()) < 0 or int(triangles.max()) >= vertices.shape[0]:
            raise ValueError("triangle index outside vertex domain")
        self.vertices = vertices
        self.triangles = triangles
        first, second, third = (vertices[triangles[:, index]] for index in range(3))
        face_normals = torch.linalg.cross(second - first, third - first)
        vertex_normals = torch.zeros_like(vertices)
        for index in range(3):
            vertex_normals.index_add_(0, triangles[:, index], face_normals)
        self.normals = torch.nn.functional.normalize(vertex_normals, dim=-1, eps=1e-12)
        self._scene = None
        self._adjacency: SparseAdjacency | None = None
        self._projection_cache: dict[str, ProjectionTable] = {}

    @classmethod
    def from_open3d(cls, mesh: object) -> "MeshCarrier":
        return cls(
            torch.from_numpy(np.asarray(mesh.vertices).copy()),
            torch.from_numpy(np.asarray(mesh.triangles).copy()),
        )

    @property
    def num_elements(self) -> int:
        return int(self.vertices.shape[0])

    def _raycasting_scene(self):
        if self._scene is None:
            try:
                import open3d as o3d
            except ImportError as error:
                raise RuntimeError("MeshCarrier requires the optional open3d dependency") from error
            legacy = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(self.vertices.numpy().astype(np.float64)),
                o3d.utility.Vector3iVector(self.triangles.numpy().astype(np.int32)),
            )
            tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(legacy)
            scene = o3d.t.geometry.RaycastingScene()
            scene.add_triangles(tensor_mesh)
            self._scene = scene
        return self._scene

    def project(self, camera: Camera) -> ProjectionTable:
        if camera.key in self._projection_cache:
            return self._projection_cache[camera.key]
        try:
            import open3d as o3d
        except ImportError as error:
            raise RuntimeError("MeshCarrier requires the optional open3d dependency") from error
        rows, columns = np.indices((camera.height, camera.width), dtype=np.float32)
        intrinsic = camera.intrinsic.numpy()
        directions_camera = np.stack(
            [
                (columns + 0.5 - intrinsic[0, 2]) / intrinsic[0, 0],
                (rows + 0.5 - intrinsic[1, 2]) / intrinsic[1, 1],
                np.ones_like(columns),
            ],
            axis=-1,
        )
        pose = camera.camera_to_world.numpy()
        directions_world = directions_camera @ pose[:3, :3].T
        origins = np.broadcast_to(pose[:3, 3], directions_world.shape)
        rays = np.concatenate([origins, directions_world], axis=-1).astype(np.float32)
        answer = self._raycasting_scene().cast_rays(o3d.core.Tensor(rays))
        primitive_ids = answer["primitive_ids"].numpy().astype(np.int64)
        primitive_uvs = answer["primitive_uvs"].numpy().astype(np.float32)
        depths = answer["t_hit"].numpy().astype(np.float32)
        valid = np.isfinite(depths) & (primitive_ids >= 0) & (primitive_ids < self.triangles.shape[0])
        pixels = np.flatnonzero(valid.reshape(-1))
        if pixels.size == 0:
            result = ProjectionTable(
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.long),
                torch.empty(0),
                torch.empty(0),
                self.num_elements,
                camera.height,
                camera.width,
                metadata={"backend": "mesh_raycast"},
            )
            self._projection_cache[camera.key] = result
            return result
        face_ids = primitive_ids.reshape(-1)[pixels]
        faces = self.triangles.numpy()[face_ids]
        uv = primitive_uvs.reshape(-1, 2)[pixels]
        barycentric = np.stack([1.0 - uv[:, 0] - uv[:, 1], uv[:, 0], uv[:, 1]], axis=-1)
        barycentric = np.clip(barycentric, 0.0, 1.0)
        barycentric /= np.maximum(barycentric.sum(-1, keepdims=True), 1e-12)
        element_ids = faces.reshape(-1)
        pixel_ids = np.repeat(pixels, 3)
        weights = barycentric.reshape(-1)
        sparse_depths = np.repeat(depths.reshape(-1)[pixels], 3)
        nonzero = weights > 1e-8
        result = ProjectionTable(
            element_ids=torch.from_numpy(element_ids[nonzero].copy()),
            pixel_ids=torch.from_numpy(pixel_ids[nonzero].copy()),
            depths=torch.from_numpy(sparse_depths[nonzero].copy()),
            weights=torch.from_numpy(weights[nonzero].copy()),
            num_elements=self.num_elements,
            height=camera.height,
            width=camera.width,
            normalization="weighted_mean",
            metadata={
                "backend": "mesh_raycast",
                "visible_pixel_count": int(pixels.size),
                "pose_convention": "camera_to_world",
                "pixel_sampling": "centres",
            },
        )
        self._projection_cache[camera.key] = result
        return result

    def neighbors(self) -> SparseAdjacency:
        if self._adjacency is None:
            faces = self.triangles.numpy()
            undirected = np.concatenate(
                [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0
            )
            undirected.sort(axis=1)
            undirected = np.unique(undirected, axis=0)
            directed = np.concatenate([undirected, undirected[:, ::-1]], axis=0)
            edge_index = torch.from_numpy(directed.T.copy())
            first = self.vertices[edge_index[0]]
            second = self.vertices[edge_index[1]]
            weights = (first - second).norm(dim=-1).clamp_min(1e-12).reciprocal()
            self._adjacency = SparseAdjacency(edge_index, weights, self.num_elements)
        return self._adjacency
