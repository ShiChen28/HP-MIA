import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import classification_report
import numpy as np

# Contains functions for model training and evaluation and for training and evaluating the attacker model


def train(model, data_loader, criterion, optimizer, verbose=False):
    """
    Function for model training step
    """
    running_loss = 0
    model.train()
    for step, (batch_img, batch_label) in enumerate(data_loader):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        batch_img, batch_label = batch_img.to(device),batch_label.to(device)
        optimizer.zero_grad()  # Set gradients to zero
        output = model(batch_img)  # Forward pass
        loss = criterion(output, batch_label)
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
        running_loss += loss
        # Print loss for each minibatch
        if verbose:
            print("[%d/%d] loss = %f" % (step, len(data_loader), loss.item()))
    return running_loss


def eval_model(model, test_loader, report=False):
    """
    Simple evaluation with the addition of a classification report with precision and recall
    """
    total = 0
    correct = 0
    gt = []
    preds = []
    with torch.no_grad():  # Disable gradient calculation
        model.eval()
        for step, (batch_img, batch_label) in enumerate(test_loader):
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            batch_img, batch_label = batch_img.to(device),batch_label.to(device)
            output = model(batch_img)

            predicted = torch.argmax(output, dim=1)
            preds.append(predicted)
            gt.append(batch_label)
            total += batch_img.size(0)
            correct += torch.sum(batch_label == predicted)

    accuracy = 100 * torch.true_divide(correct, total)
    if report:
        gt = torch.cat(gt, 0)
        preds = torch.cat(preds, 0)
        print(classification_report(gt.cpu(), preds.cpu()))

    return accuracy

