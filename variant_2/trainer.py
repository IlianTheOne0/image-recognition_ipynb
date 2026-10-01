import torch
import numpy as np
from sklearn.metrics import classification_report

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta

        self.counter = 0

        self.best_loss = None
        self.early_stop = False

    def __call__(self, validation_loss):
        if self.best_loss is None: self.best_loss = validation_loss
        elif validation_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience: self.early_stop = True
        else:
            self.best_loss = validation_loss
            self.counter = 0

class ModelCheckpoint:
    def __init__(self, filepath):
        self.filepath = filepath
        self.best_acc = 0.0

    def __call__(self, model, acc):
        if acc > self.best_acc:
            print(f"Validation accuracy improved to {acc:.4f}. Saving model...")
            torch.save(model.state_dict(), self.filepath)
            self.best_acc = acc

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total

def get_all_predictions(model, dataloader, device):
    model.eval()
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(predicted.cpu().numpy())
    return true_labels, pred_labels

def generate_classification_report(true_labels, pred_labels, classes):
    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels, target_names=classes))

def get_worst_predictions(model, dataloader, device, top_k=10):
    model.eval()
    losses = []
    images = []
    trues = []
    preds = []

    criterion_none = torch.nn.CrossEntropyLoss(reduction='none')

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion_none(outputs, labels)

            _, predicted = outputs.max(1)

            losses.extend(loss.cpu().numpy())
            images.extend(inputs.cpu())
            trues.extend(labels.cpu().numpy())
            preds.extend(predicted.cpu().numpy())

    losses = np.array(losses)
    worst_indices = np.argsort(losses)[-top_k:][::-1]

    worst_images = [images[i] for i in worst_indices]
    worst_trues = [trues[i] for i in worst_indices]
    worst_preds = [preds[i] for i in worst_indices]
    worst_losses = [losses[i] for i in worst_indices]

    return worst_images, worst_trues, worst_preds, worst_losses