import os

import torch
import numpy as np

from mesh_to_sdf import sample_sdf_near_surface
import trimesh

import argparse


class SdfDataset:
    """
    A class to store SDF data sampled from a single object

    Attributes:
        points: a tensor of the sampled points (# of points x # of dimensions)
        sdf: a tensor of the signed-distance of each point (# of points x 1)

    """

    def __init__(self, points, sdf):
        """
        Initializes the dataset

        Args:
            points: a list of sampled points
            sdf: a list of the signed-distance of each point

        """
        self.points = torch.tensor(points, dtype=torch.float)
        self.sdf = torch.tensor(sdf, dtype=torch.float)

        assert self.points.size(0) == self.sdf.size(0)
    
    def __len__(self):
        return self.points.size(0)
    
    def __getitem__(self, idx):
        return self.points[idx], self.sdf[idx]


class MultiSdfDataset:
    """
    A class to store SDF data sampled from multiple objects equipped with latent codes

    Attributes:
        latent_dim: the dimension of the latent codes
        points: a tensor containing all sampled points with concatenated latent codes (# of points x # of dimensions)
        sdf: a tensor containing the signed-distances of the sampled points (# of points x 1)

    """

    def __init__(self, points_list: list, sdf_list: list):
        """
        Initializes the dataset

        Args:
            points_list: a list of the sampled points for different objects
            sdf_list: a list of the signed-distances of the sampled points
        
        """

        assert len(points_list) == len(sdf_list)

        self.latent_dim = len(points_list) if len(points_list) > 2 else 1

        self.points = torch.tensor(points_list[0], dtype=torch.float)
        self.sdf = torch.tensor(sdf_list[0], dtype=torch.float)
        t = torch.tile(self.compute_latent(0), (self.points.size(0), 1))

        self.boundaries = [0, self.points.size(0)]

        for i in range(1, len(points_list)):
            self.points = torch.concatenate(
                (self.points, torch.tensor(points_list[i], dtype=torch.float)),
                dim=0
            )
            self.sdf = torch.concatenate(
                (self.sdf, torch.tensor(sdf_list[i], dtype=torch.float)),
                dim=0
            )
            t = torch.concatenate(
                (
                    t, 
                    torch.tile(self.compute_latent(i), (self.points.size(0) - t.size(0), 1))
                ),
                dim=0
            )
            self.boundaries.append(self.points.size(0))
        
        self.points = torch.concatenate((self.points, t), dim=1)

    def compute_latent(self, i: int):
        """
        Computes the latent code of the ith object in the dataset

        Args:
            i: the index of an object in the dataset
        
        Returns:
            The latent code of the object

        """
        if self.latent_dim == 1:
            assert i in [0, 1]

            return torch.tensor(0 if i == 0 else 1, dtype=torch.float)
        else:
            assert i < self.latent_dim

            t = torch.zeros((self.latent_dim,), dtype=torch.float)
            t[i] = 1

            return t
    
    def get_object_points(self, i: int):
        """
        Fetches the points in the dataset that were sampled from object i

        Args:
            i: the index of an object in the dataset
        
        Returns:
            The points in the dataset that were sampled from object i

        """

        assert i < len(self.boundaries) - 1
        return self.points[self.boundaries[i]:self.boundaries[i+1]]
    
    def __len__(self):
        return self.points.size(0)

    def __getitem__(self, idx):
        return self.points[idx], self.sdf[idx]
        

def generate_dataset_from_mesh(object_name: str, N: int = 500000):
    """
    Generates a dataset of (point, signed-distance) pairs for the mesh with the given name
    The dataset is saved to data/datasets/{object_name}_{train/test}.npz

    Args:
        object_name: the name of the object
        N: the number of points to sample 

    """

    mesh = trimesh.load(os.path.join('data', 'objects', f'{object_name}.obj'))

    points, sdf = sample_sdf_near_surface(mesh, number_of_points=N)
    np.savez(os.path.join('data', 'datasets', f'{object_name}_train.npz'), points=points, sdf=sdf)

    points, sdf = sample_sdf_near_surface(mesh, number_of_points = N // 10)
    np.savez(os.path.join('data', 'datasets', f'{object_name}_test.npz'), points=points, sdf=sdf)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--object', '-o', type=str, help="The name of the obj file in data/objects to sample from")
    parser.add_argument('-N', type=int, default=500000, help="The number of points to sample")
    args = parser.parse_args()

    generate_dataset_from_mesh(args.object.replace('.obj', ''), args.N)