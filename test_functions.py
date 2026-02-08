import numpy as np
import torch
from model import UNetModel
import torch.nn as nn
from dataset import MmWhsDataset
from evaluate import DICE, pixel_accuracy, accuracy_graph, segmentation_overlay

def test_dataset(batch_size = 16):
    file_root = "/zhome/2d/4/206216/Unet_project_jan"
    testing_files = "data/splits/test.txt" # testing
    file_ids = np.loadtxt(testing_files, dtype=str)

    # use GPU
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print(f"Using device: {device}")
    u_net = UNetModel()
    u_net.load_state_dict(torch.load("model.pth"))
    u_net.to(device)    

    testing_set = MmWhsDataset(file_ids, file_root)
    testing_loader = torch.utils.data.DataLoader(testing_set, batch_size=batch_size, num_workers=0, shuffle=False)

    # loss function
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        u_net.eval()

        test_accuracies = []
        test_dices = []
        test_losses = []

        print("Testing...")

        for i, (images, labels) in enumerate(testing_loader):
            images, labels = images.to(device), labels.to(device)
            output = u_net(images)

            # getting accuracies
            test_accuracies.append(pixel_accuracy(output, labels).item())

            # calculating dice
            test_dices.append(DICE(output, labels).item())

            # calculating loss 
            test_losses.append(criterion(output, labels).item())

            # calling segmentation overlay
            if i % 10 == 0:
                save_path = (f"{file_root}/data/data_from_model/overlay_batch{i}.png")
                segmentation_overlay(images[0], output[0], labels[0], save_path)
                
        test_accuracy = np.mean(test_accuracies)
        test_dice = np.mean(test_dices)
        test_loss = np.mean(test_losses)

        print("__________________________________________")
        print(f"Test accuracy:      {test_accuracy}")
        print(f"Test DICE score:    {test_dice}")
        print(f"Test loss:          {test_loss}")
        print("__________________________________________")


if __name__ == '__main__':
    test_dataset()