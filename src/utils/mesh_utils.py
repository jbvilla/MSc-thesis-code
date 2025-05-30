import numpy as np
import trimesh
from trimesh import Geometry


def _subdivide_mesh(mesh: Geometry | list[Geometry]) -> Geometry:
    subdivided_mesh = trimesh.remesh.subdivide(mesh.vertices, mesh.faces)
    return trimesh.Trimesh(vertices=subdivided_mesh[0], faces=subdivided_mesh[1])


def subdivide_update_mesh(path_obj_file: str, count: int = 1) -> None:
    """
    Subdivide the mesh and update the obj file
    :param path_obj_file: path to the obj file
    :param count: number of times to subdivide the mesh
    :return: None
    """
    mesh = trimesh.load_mesh(path_obj_file)
    for _ in range(count):
        mesh = _subdivide_mesh(mesh)

    mesh.export(path_obj_file)


def get_max_distance_edge(path_obj_file: str) -> float:
    """
    Get the maximum distance edge of the mesh
    :param path_obj_file: path to the obj file
    :return: the distance of the edge
    """
    mesh = trimesh.load_mesh(path_obj_file)

    edges = mesh.edges_unique
    vertices = mesh.vertices

    # Calculate the vector between the vertices of the edges
    # then calculate the length of the vectors (norm) (axis=1 because we want the rows)
    edge_lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)

    return np.max(edge_lengths)
