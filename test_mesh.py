import argparse

from mesh_to_sdf import mesh_to_voxels

import trimesh
import skimage

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check if a mesh is compatible with sdf sampler. If this output does resemble the original mesh, then it is not compatible")
    parser.add_argument('--file', '-f', type=str, help="path to a .obj file")
    args = parser.parse_args()

    mesh = trimesh.load(args.file)

    voxels = mesh_to_voxels(mesh, 64, pad=True)

    vertices, faces, normals, _ = skimage.measure.marching_cubes(voxels, level=0)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
    mesh.show()
