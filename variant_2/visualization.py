import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

def imshow(img, mean, std):
    img = img.numpy().transpose((1, 2, 0))

    mean = np.array(mean)
    std = np.array(std)
    img = std * img + mean
    img = np.clip(img, 0, 1)

    plt.imshow(img)

def visualize_eda(dataloader, strategy):
    dataiter = iter(dataloader)
    images, labels = next(dataiter)
    fig = plt.figure(figsize=(10, 10))

    for i in range(9):
        ax = fig.add_subplot(3, 3, i + 1, xticks=[], yticks=[])
        imshow(images[i], strategy.mean, strategy.std)
        ax.set_title(strategy.classes[labels[i]])
    plt.show()

def plot_metrics(train_losses, val_losses, train_accs, val_accs):
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(epochs, train_losses, 'b-', label='Train')
    ax1.plot(epochs, val_losses, 'r-', label='Validation')
    ax1.set_title('Loss over Epochs')
    ax1.legend()

    ax2.plot(epochs, train_accs, 'b-', label='Train')
    ax2.plot(epochs, val_accs, 'r-', label='Validation')
    ax2.set_title('Accuracy over Epochs')
    ax2.legend()

    plt.show()

def visualize_predictions(model, dataloader, strategy, device):
    model.eval()
    dataiter = iter(dataloader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

    images = images.cpu()
    labels = labels.cpu()
    predicted = predicted.cpu()

    fig = plt.figure(figsize=(15, 6))
    for idx in range(10):
        ax = fig.add_subplot(2, 5, idx + 1, xticks=[], yticks=[])
        imshow(images[idx], strategy.mean, strategy.std)
        ax.set_title(f"True: {strategy.classes[labels[idx]]}\nPred: {strategy.classes[predicted[idx]]}", color=("green" if labels[idx] == predicted[idx] else "red"))

    plt.show()

def plot_confusion_matrix_heatmap(true_labels, pred_labels, classes):
    cm = confusion_matrix(true_labels, pred_labels)
    fig, ax = plt.subplots(figsize=(10, 8))

    cax = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    fig.colorbar(cax)

    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), xticklabels=classes, yticklabels=classes, title='Confusion Matrix', ylabel='True Label', xlabel='Predicted Label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    plt.show()

def plot_worst_predictions(worst_images, worst_trues, worst_preds, worst_losses, strategy):
    fig = plt.figure(figsize=(15, 6))
    fig.suptitle('Top 10 Worst Predictions', fontsize=16)

    for idx in range(len(worst_images)):
        ax = fig.add_subplot(2, 5, idx + 1, xticks=[], yticks=[])
        imshow(worst_images[idx], strategy.mean, strategy.std)
        ax.set_title(f"T: {strategy.classes[worst_trues[idx]]}\n" f"P: {strategy.classes[worst_preds[idx]]}\n" f"L: {worst_losses[idx]:.2f}", color="red", fontsize=10)

    plt.tight_layout()
    plt.show()