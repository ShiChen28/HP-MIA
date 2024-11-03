import torch
import torch.nn as nn
from torchvision import datasets,transforms, models
import torch.optim as optim
import os
import numpy as np
from my_dataloader import dataloader
from train_eval import train, eval_model
import matplotlib.pyplot as plt  
from opacus.utils.batch_memory_manager import BatchMemoryManager

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
                
def Train_dp_targetmodel(model, train_loader, testloader, optimizer, epochs, max_physical_batch_size, device, privacy_engine, DELTA, save_path, save_new_models = 1):
    print("start training target model:")
    for epoch in range(epochs):
        DP_train(model, train_loader, optimizer,  epoch, epochs, max_physical_batch_size, device, privacy_engine, DELTA, save_path, save_new_models = 1)
        if (epoch + 1)%100 == 0 or epoch == epochs - 1:
            accuracy_train_target = eval_model(model,train_loader,report=False)
            accuracy_test_target = eval_model(model,testloader,report = False)
            print("Target model: epoch[%d/%d]  training set accuracy: %.5f  test set accuracy: %.5f"
                      % (epoch + 1, epochs,  accuracy_train_target, accuracy_test_target))
        
        if (epoch % 10 == 0 or epoch == epochs - 1) and save_new_models:
                # Save model after each epoch
            if not os.path.exists("./models"):
                os.mkdir("./models")  # Create the folder models if it doesn't exist
            torch.save(model.state_dict(), save_path)

                
def accuracy(preds, labels):
    return (preds == labels).mean()
                
def DP_train(model, train_loader, optimizer, epoch, epochs, max_physical_batch_size, device, privacy_engine, DELTA, save_path, save_new_models = 1):
    model.train()
    criterion = nn.CrossEntropyLoss()

    losses = []
    top1_acc = []
    
    with BatchMemoryManager(
        data_loader=train_loader, 
        max_physical_batch_size=max_physical_batch_size, 
        optimizer=optimizer
    ) as memory_safe_data_loader:

        for i, (images, target) in enumerate(memory_safe_data_loader):   
            optimizer.zero_grad()
            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()

            # measure accuracy and record loss
            acc = accuracy(preds, labels)

            losses.append(loss.item())
            top1_acc.append(acc)

            loss.backward()
            optimizer.step()

        if epoch == epochs - 1:
            epsilon = privacy_engine.get_epsilon(DELTA)
            print(
                f"\tTrain Epoch: {epoch} \t"
                f"Loss: {np.mean(losses):.6f} "
                f"(ε = {epsilon:.2f}, δ = {DELTA})"
            )


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

