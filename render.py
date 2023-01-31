import argparse
import trimesh
import skimage
from mesh_to_sdf import mesh_to_voxels
import numpy as np
import torch
import matplotlib.pyplot as plt
import pyrender
import os
import ast

from networks import DeepSDF

def render_sdf(f, dim: int, t = None, res: int = 100):
    """
    Renders a model of SDF function evaluated in the 2x2 square/2x2x2 cube centered at the origin

    Args:
        f: the sdf function
        dim: the dimension of the space. Must be equal to 2 or 3
        t: the latent code (optional)
        res: the resolution of the grid
    
    """

    assert dim == 2 or dim == 3

    # Init grid points
    cell_size = 2 / res
    offset = cell_size / 2

    points = np.arange(-1 + offset, 1, cell_size)[..., np.newaxis]
    for _ in range(1, dim):
        points = np.concatenate(
            (
                np.repeat(points, res, axis=0),
                np.tile(points[:,-1], res)[..., np.newaxis]
            ),
            axis=1
        )

    # Evaluate function at grid points

    X = torch.tensor(points, dtype=torch.float)
    
    # Build latents
    if t is not None:
        t = torch.tensor(t, dtype=torch.float)
        t = torch.tile(t, (points.shape[0], 1))
        X = torch.cat((X, t), dim=1)

    y = f(X).detach().numpy().squeeze()

    # Check if the sdf defines a surface
    if y.min() > 0:
        print("This SDF does not contain any negative values in the unit cube. This probably means the model needs to train more.")
        return

    # Display grid
    if dim == 2:
        mag = np.abs(y)
        mag = 1 - (mag / np.max(mag))

        color = np.zeros((res*res, 3))
        color[y <= 0, 2] = mag[y <= 0]
        color[y > 0, 1] = mag[y > 0]

        color = color.reshape((res, res, 3))

        plt.figure()
        plt.imshow(color)
        plt.show()
    elif dim == 3:
        voxels = y.reshape((res, res, res))
        voxels = np.pad(voxels, 1, mode='constant', constant_values=1)

        vertices, faces, normals, _ = skimage.measure.marching_cubes(voxels, level=0)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
        mesh.show()


def render_pc(f, points):
    """
    Renders a point cloud with a point being red if the sdf function is positive and blue if the sdf function is negative

    Args:
        f: the sdf function
        points: the point cloud

    """

    sdf = f(points).squeeze()

    colors = np.zeros(points.shape)
    colors[sdf < 0, 2] = 1
    colors[sdf > 0, 0] = 1
    cloud = pyrender.Mesh.from_points(points[:,:3], colors=colors)
    scene = pyrender.Scene()
    scene.add(cloud)
    viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Display the learned SDF of a trained model")
    parser.add_argument('obj_names', type=str, nargs="+", help="The names of the objects whose learned SDFs will be rendered")
    parser.add_argument('--epoch', '-e', type=int, required=True, help="The number of epochs the model was trained for (interval of 5)")
    parser.add_argument('--lipschitz', '-l', default=False, const=True, action="store_const",
                            help="Whether or not to use the Lipschitz regularized model")
    parser.add_argument("-t", help="The latent code to be used")
    args = parser.parse_args()

    args.obj_names.sort()
    filename = "_".join(args.obj_names)

    is_lipschitz = args.lipschitz and len(args.obj_names) > 1
    if is_lipschitz:
        filename += "_lipschitz"
    filename += f"_{args.epoch}.pth"

    if len(args.obj_names) < 3:
        latent_dim = len(args.obj_names) - 1
    else:
        latent_dim = len(args.obj_names)

    if latent_dim == 0:
        t = None
    else:
        t = ast.literal_eval(args.t)

    path = os.path.join("data", "models", filename)
    if os.path.exists(path):
        model = DeepSDF(input_dim=3, latent_dim=latent_dim, is_lipschitz=is_lipschitz)
        model.load_state_dict(torch.load(path))

        model.eval()
        with torch.no_grad():
            render_sdf(model, dim=3, t=t)
    else:
        print("This model does not exist.")
    