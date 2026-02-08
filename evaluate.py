import torch
import matplotlib.pyplot as plt
import numpy as np

def DICE(prediction, target, threshold=0.5, epsilon = 1e-6):
    # convert predictions to probabilities
    prediction = torch.sigmoid(prediction)

    # make binary using threshold
    prediction = (prediction > threshold).float()

    # flatten to 1D
    prediction = prediction.view(-1)
    target = target.view(-1)

    # calculate intersection
    intersection = (prediction * target).sum()
    dice_score = (2 * intersection + epsilon)/(prediction.sum() + target.sum()+ epsilon)

    return dice_score

def pixel_accuracy(output, labels):
    preds = (output > 0.0).float()
    correct = (preds == labels).sum().float()
    total = torch.numel(labels)
    return correct/total

def accuracy_graph(training_list, validation_list, metric):
    epochs = range(1, len(training_list) + 1)
    plt.figure()
    plt.plot(epochs, training_list, label='train')
    plt.plot(epochs, validation_list, label='validation')
    plt.xlabel('Epochs')
    plt.ylabel(f'{metric}')
    plt.legend()
    plt.title(f"Train and validation {metric}")
    plt.savefig(f"data/data_from_model/{metric}.png")
    plt.close()

def segmentation_overlay(image, prediction, label, save_path):
    # loading the images
    img = image.squeeze().cpu().numpy()
    labl = label.squeeze().cpu().numpy()
    pred = torch.sigmoid(prediction).squeeze().cpu().detach().numpy()

    plt.figure(figsize = [15,5])
    
    # original image
    plt.subplot(1,3,1)
    plt.imshow(img, cmap="gray")
    plt.title("Original Image")
    plt.axis('off')

    # predicted image
    plt.subplot(1,3,2)
    plt.imshow(img, cmap="gray")
    masked_pred = np.ma.masked_where(pred == 0, pred)
    plt.imshow(masked_pred, cmap = "Reds", alpha = 0.5)
    plt.title("Predicted Segmentation")
    plt.axis('off')

    # ground truth image
    plt.subplot(1,3,3)
    plt.imshow(img, cmap="gray")
    masked_labl = np.ma.masked_where(labl == 0, labl)
    plt.imshow(labl, cmap = "Reds", alpha = 0.5)
    plt.title("Ground Truth Segmentation")
    plt.axis('off')

    plt.savefig(save_path)
    plt.close()