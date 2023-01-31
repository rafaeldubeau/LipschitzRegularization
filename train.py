import os

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

import numpy as np

from networks import DeepSDF, train_loop, eval_loop, device
from datasets import SdfDataset, MultiSdfDataset
from render import render_pc

import argparse


def multi(names: list[str], epochs: int, starting_epoch: int, lipschitz: bool):
    names.sort()

    if len(names) == 1:
        lipschitz = False

    filename = "_".join(names)
    if lipschitz:
        filename += "_lipschitz"

    # Hyperparameters
    learning_rate = 10e-4
    batch_size = 512
    alpha = 10e-6

    # Load dataset
    if len(names) == 1:
        train_dataset_file = np.load(os.path.join('data', 'datasets', f'{names[0]}_train.npz'))
        test_dataset_file = np.load(os.path.join('data', 'datasets', f'{names[0]}_test.npz'))

        train_data = SdfDataset(train_dataset_file["points"], train_dataset_file["sdf"])
        test_data = SdfDataset(test_dataset_file["points"], test_dataset_file["sdf"])

        latent_dim = 0
    else:
        train_dataset_files = []
        test_dataset_files = []
        for name in names:
            train_dataset_files.append(np.load(os.path.join('data', 'datasets', f'{name}_train.npz')))
            test_dataset_files.append(np.load(os.path.join('data', 'datasets', f'{name}_test.npz')))

        train_data = MultiSdfDataset(
            [file["points"] for file in train_dataset_files], 
            [file["sdf"] for file in train_dataset_files]
        )
        test_data = MultiSdfDataset(
            [file["points"] for file in test_dataset_files], 
            [file["sdf"] for file in test_dataset_files]
        )

        latent_dim = train_data.latent_dim

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    # Init model
    model = DeepSDF(input_dim=3, latent_dim=latent_dim, is_lipschitz=lipschitz).to(device)
    if starting_epoch > 0:
        model.load_state_dict(torch.load(os.path.join("data", "models", f"{filename}_{starting_epoch}.pth")))

    base_loss = nn.MSELoss()
    if lipschitz:
        loss_fn = lambda pred, y, model: base_loss(pred.squeeze(), y.squeeze()) + alpha * model.get_lipschitz_bound()
    else:
        loss_fn = lambda pred, y, model: base_loss(pred.squeeze(), y.squeeze())

    optimizer = Adam(model.parameters(), lr=learning_rate)


    # Train loop
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(model, train_dataloader, optimizer, loss_fn)
        eval_loop(model, test_dataloader, loss_fn)
        if (t+1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join("data", "models", f"{filename}_{starting_epoch+t+1}.pth"))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model to learn the SDFs of some 3D models")
    parser.add_argument('obj_names', type=str, nargs="+", help="The names of the objects whose SDFs will be learned")
    parser.add_argument('--epochs', '-e', type=int, default=30, help="The number of epochs to train")
    parser.add_argument('--startepoch', '-s', type=int, default=0, help="The number of epochs of the pretrained model to load")
    parser.add_argument('--lipschitz', '-l', default=False, const=True, action="store_const",
                            help="Whether or not the model with be trained using Lipschitz regularization or not. If the the model is only being trained on 1 object, then this will have no effect.")
    args = parser.parse_args()

    multi(args.obj_names, args.epochs, args.startepoch, args.lipschitz)