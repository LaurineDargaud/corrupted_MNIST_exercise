import argparse
import os
import sys

import matplotlib.pyplot as plt
import torch
from model import MyAwesomeModel
from torch.utils.data import DataLoader


class train_model(object):
    """
    Train model on train dataset
    then generate the plot of training loss VS steps
    """

    def __init__(self):
        self.train()

    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description="Training arguments")
        parser.add_argument("--lr", default=0.1, type=float)
        # add any additional argument that you want
        parser.add_argument("--nb_epochs", default=200, type=int)
        parser.add_argument("--save_file", default="best_model")
        parser.add_argument("--criterion", default="NLLLoss")
        parser.add_argument("--optimizer", default="SGD")
        args = parser.parse_args(sys.argv[1:])
        print(args)

        # TODO: Implement training loop here
        # get model
        model = MyAwesomeModel()

        # get training dataset
        train_dataset = torch.load(
            os.path.abspath(__file__ + "/../../../data/processed/train_dataset.pth")
        )
        trainloader = DataLoader(train_dataset, batch_size=1000, shuffle=True)

        criterion = eval(f"torch.nn.{args.criterion}()")
        optimizer = eval(
            f"torch.optim.{args.optimizer}(model.parameters(), lr = {args.lr})"
        )

        epochs = args.nb_epochs
        steps = 0

        loss_tracking = {}
        best_loss = None

        for e in range(epochs):
            running_loss = 0
            for images, labels in trainloader:

                # set model to train mode
                model = model.train()

                optimizer.zero_grad()

                log_ps = model(images)
                loss = criterion(log_ps, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            loss_tracking[steps] = running_loss
            steps += 1
            print(f"Epoch {e+1}/{epochs}...   Loss: {running_loss}")

            # save best model
            if best_loss is None or running_loss < best_loss:
                best_loss = running_loss
                saving_path = os.path.abspath(
                    __file__ + "/../../../models/" + f"{args.save_file}.pth"
                )
                torch.save(model, saving_path)

            # save figure with training loss VS steps
            plt.plot(list(loss_tracking.keys()), list(loss_tracking.values()))
            plt.xlabel("Steps")
            plt.ylabel("Training Loss")
            plt.title(
                f"Training Loss evolution using {args.criterion} criterion,{args.optimizer} optimizer and {args.nb_epochs} epochs",
                size=10,
            )
            plt.savefig(
                os.path.abspath(
                    __file__ + "/../../../reports/figures/training_loss_plot.png"
                )
            )


if __name__ == "__main__":
    train_model()
