import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader


class predict_model(object):
    """Test a trained model on a test dataset then display obtained accuracy"""

    def __init__(self):
        self.predict()

    def predict(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description="Training arguments")
        parser.add_argument("load_model_from", default="")
        # add any additional argument that you want
        parser.add_argument("test_dataset", default="")
        args = parser.parse_args(sys.argv[1:])
        print(args)

        # TODO: Implement evaluation logic here

        # get trained model
        model = torch.load(
            os.path.abspath(__file__ + "/../../../" + args.load_model_from)
        )

        # get test dataloader
        test_dataset = torch.load(
            os.path.abspath(__file__ + "/../../../" + args.test_dataset)
        )
        testloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

        with torch.no_grad():
            # set model to evaluation mode
            model = model.eval()
            all_equals = []
            for images, labels in testloader:
                probas = torch.exp(model(images))
                _, top_class = probas.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                all_equals.append(equals)
            equals = torch.cat(all_equals, 0)
            accuracy = torch.mean(equals.type(torch.FloatTensor)).item()
            print(f"Accuracy: {round(accuracy*100,2)}%")


if __name__ == "__main__":
    predict_model()
