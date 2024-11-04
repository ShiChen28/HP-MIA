import torch
import torch.nn as nn
from torchvision import datasets,transforms, models
import torch.optim as optim
import os
import numpy as np
from my_dataloader import dataloader
from train_eval import train, eval_model
import matplotlib.pyplot as plt  

def Train_shadowmodel(shadow_net, shadow_train_loader, shadow_loss, shadow_optim, n_epochs, testloader, shadow_path, save_new_models =1):
    print("start training shadow model: ")
    for epoch in range(n_epochs):
        loss_train_shadow = train(shadow_net, shadow_train_loader, shadow_loss, shadow_optim, verbose=False)
        # Evaluate model after every five epochs
        if (epoch+1) %100 == 0 or epoch + 1 == n_epochs:
            accuracy_train_shadow = eval_model(shadow_net, shadow_train_loader, report=False)
            accuracy_test_shadow = eval_model(shadow_net, testloader, report=False)
            print("Shadow model: epoch[%d/%d] Train loss: %.5f training set accuracy: %.5f  test set accuracy: %.5f"
                      % (epoch + 1, n_epochs, loss_train_shadow, accuracy_train_shadow, accuracy_test_shadow))
    if save_new_models:
        if not os.path.exists("./models"):
            os.mkdir("./models")  # Create the folder models if it doesn't exist
            # Save model after each epoch if argument is true
        torch.save(shadow_net.state_dict(), shadow_path)


def Train_targetmodel(target_net, target_train_loader, target_loss, target_optim,  n_epochs, testloader, target_path, save_new_models = 1):
    print("start training target model: ")
    for epoch in range(n_epochs):
        loss_train_target = train(target_net, target_train_loader, target_loss, target_optim, verbose=False)
            # Evaluate model after every five epochs
        if (epoch + 1) %100 == 0 or epoch == n_epochs - 1:
            accuracy_train_target = eval_model(target_net, target_train_loader, report=False)
            accuracy_test_target = eval_model(target_net, testloader, report=False)
            print("Target model: epoch[%d/%d] Train loss: %.5f training set accuracy: %.5f  test set accuracy: %.5f"
                      % (epoch + 1, n_epochs, loss_train_target, accuracy_train_target, accuracy_test_target))
        if save_new_models:
                    # Save model after each epoch
            if not os.path.exists("./models"):
                os.mkdir("./models")  # Create the folder models if it doesn't exist
            torch.save(target_net.state_dict(), target_path)
            
                
def accuracy(preds, labels):
    return (preds == labels).mean()
                
def Load_shadowmodel(shadow_path, shadow_net):
    if os.path.exists(shadow_path):
        print("Load shadow model")
        shadow_net.load_state_dict(torch.load(shadow_path))
    return shadow_net

def Load_targetmodel(target_path, target_net):
    if os.path.exists(target_path):
        print("Load target model")
        target_net.load_state_dict(torch.load(target_path))    
    return target_net

