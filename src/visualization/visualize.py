import os
import sys

import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

sys.path.append("../models")


class visualize(object):
    def __init__(self):
        # Loads a pre-trained network
        model = torch.load(
            os.path.abspath(__file__ + "/../../../" + "models/best_model.pth")
        )

        # Extract intermediate representation of the data (training set) from your cnn
        # This could be the features just before the final classification layer
        train_dataset = torch.load(
            os.path.abspath(__file__ + "/../../../data/processed/train_dataset.pth")
        )
        trainloader = DataLoader(train_dataset, batch_size=1000, shuffle=True)
        image, labels = next(iter(trainloader))

        output1, output2 = model.get_intermediate_outputs(image)

        # Visualize features in a 2D space using t-SNE to do the dimensionality reduction
        output1_embedded = TSNE(
            n_components=2, learning_rate="auto", init="random"
        ).fit_transform(output1.detach().numpy())
        output2_embedded = TSNE(
            n_components=2, learning_rate="auto", init="random"
        ).fit_transform(output2.detach().numpy())

        # Save the visualization to a file in the reports/figures/ folder

        plt.figure()
        plt.scatter(
            output1_embedded.T[0],
            output1_embedded.T[1],
            marker="x",
            c=list(labels.numpy()),
        )
        plt.title("Intermediate Representation 1 - after the first layer", size=10)
        plt.savefig(
            os.path.abspath(
                __file__ + "/../../../reports/figures/intermediate_representation1.png"
            )
        )

        plt.figure()
        plt.scatter(
            output2_embedded.T[0],
            output2_embedded.T[1],
            marker="x",
            c=list(labels.numpy()),
        )
        plt.title(
            "Intermediate Representation 2 - just before the final layer", size=10
        )
        plt.savefig(
            os.path.abspath(
                __file__ + "/../../../reports/figures/intermediate_representation2.png"
            )
        )


if __name__ == "__main__":
    visualize()
