import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import UNetModel
from dataset import MmWhsDataset
from evaluate import DICE, pixel_accuracy, accuracy_graph

def train_model(epochs=20, lr=0.001, batch_size=16):
    file_root = "/zhome/2d/4/206216/Unet_project_jan"
    training_files = "data/splits/train.txt" # training
    validation_files = "data/splits/val.txt" # validation
    file_ids = np.loadtxt(training_files, dtype=str)
    file_ids_val = np.loadtxt(validation_files, dtype=str)
    
    # use GPU
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print(f"Using device: {device}")
    u_net = UNetModel()
    u_net.to(device)

    # Training dataloader
    training_set = MmWhsDataset(file_ids, file_root)
    training_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, num_workers=4, shuffle=True)
    print(f"Number of batches in dataset is {len(training_loader)}")

    # Validation dataloader
    validation_set = MmWhsDataset(file_ids_val, file_root)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, num_workers=0, shuffle=False)

    # loss function
    criterion = nn.BCEWithLogitsLoss()
    # optimizer
    optimizer = optim.Adam(u_net.parameters(), lr = lr)
    
    # initialize lists
    accuracy_total=[]
    DICE_total=[]
    loss_total=[]

    val_accuracy_total=[]
    val_DICE_total=[]
    val_loss_total=[]
    
    print("BEGIN TRAINING...")
    for e in range(epochs):
        u_net.train()
        
        # pr. epoch metrics
        epoch_loss = [] # tracks losses for this epoch
        epoch_accuracy = []
        epoch_DICE = []
        
        for image_batch, label_batch in training_loader:
            # in the format (batches, channels, width, height)
            image_batch = image_batch.to(device)
            label_batch = label_batch.to(device)

            optimizer.zero_grad()

            # forward prediction based on input
            predicted_heart = u_net(image_batch)
            heart_label = label_batch

            # _____ CALCULATE METRICS ______

            # accuracy
            batch_accuracy = pixel_accuracy(predicted_heart, heart_label)
            epoch_accuracy.append(batch_accuracy.item())

            # DICE
            batch_dice = DICE(predicted_heart, heart_label)
            epoch_DICE.append(batch_dice.item())

            # calculate loss
            batch_loss = criterion(predicted_heart, heart_label)
            epoch_loss.append(batch_loss.item())

            # _____ UPDATING PARAMETERS ______

            batch_loss.backward()
            optimizer.step()
        
        # average epoch metrics
        avg_epoch_accuracy = np.mean(epoch_accuracy)
        avg_epoch_dice = np.mean(epoch_DICE)
        avg_epoch_loss = np.mean(epoch_loss)

        # performing validation
        with torch.no_grad():
            u_net.eval()
            
            # initializing lists for validation metrics
            val_accuracy = []
            val_DICE = []
            val_loss = []
            
            for image_batch, label_batch in validation_loader:
                # in the format (batches, channels, width, height)
                image_batch = image_batch.to(device)
                label_batch = label_batch.to(device)

                # forward predicition
                outputs = u_net(image_batch)

                # _____ CALCULATE METRICS ______

                # calculate accuracy
                batch_accuracy = pixel_accuracy(outputs, label_batch)
                val_accuracy.append(batch_accuracy.item())

                # DICE
                batch_dice = DICE(outputs, label_batch)
                val_DICE.append(batch_dice.item())

                # calculate loss
                batch_loss = criterion(outputs, label_batch)
                val_loss.append(batch_loss.item())
            
            # average val metrics
            avg_val_accuracy = np.mean(val_accuracy)
            avg_val_dice = np.mean(val_DICE)
            avg_val_loss = np.mean(val_loss)

        # pr. epoch metrics
        print("__________________________________________")
        print(f"Epoch {e} pixel accuracy    {avg_epoch_accuracy}")
        print(f"Epoch {e} DICE score        {avg_epoch_dice}")
        print(f"Epoch {e} loss              {avg_epoch_loss} \n")
        print(f"Validation accuracy         {avg_val_accuracy}")
        print(f"Validation DICE score       {avg_val_dice}")
        print(f"Validation loss             {avg_val_loss}")
        print("__________________________________________")

        # adding to total metrics
        accuracy_total.append(avg_epoch_accuracy)
        DICE_total.append(avg_epoch_dice)
        loss_total.append(avg_epoch_loss)
        val_accuracy_total.append(avg_val_accuracy)
        val_DICE_total.append(avg_val_dice)
        val_loss_total.append(avg_val_loss)
    
    # saving model
    torch.save(u_net.state_dict(), "model.pth")

    # plotting graph
    accuracy_graph(accuracy_total, val_accuracy_total, "Accuracy")
    accuracy_graph(DICE_total, val_DICE_total, "DICE")
    accuracy_graph(loss_total, val_loss_total, "Loss")

if __name__ == '__main__':
    train_model()