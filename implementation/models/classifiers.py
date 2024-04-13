from torchvision import models
from torch import nn, cat, FloatTensor, no_grad, save
from tqdm import tqdm
from metrics.accuracy import calculate_mean_roc_auc
import sys
import time
from datetime import datetime
from accelerate import Accelerator

def chexnet(num_classes, device):
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT).to(device)

    # Assign parts like this to avoid anonymous members of the classification block
    classifier_block = nn.Sequential()
    classifier_block.linear1 = nn.Linear(1024, num_classes)
    classifier_block.sigmoid1 = nn.Sigmoid()
    model.classifier = classifier_block.to(device)
    return model

def chexnet_test_net(model, test_loader, device):
        model.eval()
        running_loss = 0.

        gt_labels = FloatTensor().to(device)
        output_labels = FloatTensor().to(device)

        with no_grad():
            for i, batch in enumerate(tqdm(val_loader, leave=False, miniters=50, mininterval=80)):
                images, labels = batch
                outputs = model(images.to(device))
                loss = loss_fn(outputs, labels.to(device))
                running_loss += loss
    
                gt_labels = cat((gt_labels, labels.to(device)), 0)
                output_labels = cat((output_labels, outputs), 0)
    
        avg_val_loss = running_loss / len(val_loader)
        avg_val_loss_series.append(avg_val_loss.item())
        lr_scheduler.step(avg_val_loss)
    
        m_auc = calculate_mean_roc_auc(gt_labels, output_labels, class_count)
        return m_auc

def chexnet_train_one_epoch(model, optimizer, loss_fn, loss_calc_interval, train_loader, device, accelerator=None):
    running_loss = 0.
    last_loss = 0.
    loader_length = len(train_loader)
    if loss_calc_interval > loader_length:
        loss_calc_interval = loader_length
        
    for i, batch in enumerate(tqdm(train_loader, leave=False, miniters=100, mininterval=100)):
        
        optimizer.zero_grad()

        images, labels = batch
        images.requires_grad_(True)
        labels.requires_grad_(True)
        prediction = model(images.to(device))
        loss = loss_fn(prediction.to(device), labels.to(device))

        if accelerator is None:
            loss.backward()
        else:
            accelerator.backward(loss)

        optimizer.step()

        running_loss += loss.item()
        
        if (i + 1) % loss_calc_interval == 0:
            last_loss = running_loss/loss_calc_interval
            print("Last average training loss at batch {} out of {}: {}".format(i + 1, loader_length, last_loss))
            running_loss= 0
            
            if i + loss_calc_interval > loader_length:
                loss_calc_interval = loader_length
    
    return last_loss

def fit_chexnet(model, optimizer, lr_scheduler, loss_fn, train_loader, val_loader, num_epochs, class_count, result_folder, device, accelerator=None):
    start = datetime.now()
    best_val_loss = sys.maxsize
    loss_fn = nn.CrossEntropyLoss(reduction="mean")
    avg_train_loss_series = []
    avg_val_loss_series = []
    avg_val_acc_series = []
    
    for epoch in tqdm(range(num_epochs)):
        print("Starting epoch {} out of {}".format(epoch+ 1, num_epochs))
        
        model.train()
        avg_train_loss = chexnet_train_one_epoch(model, optimizer, loss_fn, 500, train_loader, device, accelerator)
        avg_train_loss_series.append(avg_train_loss)
        
        model.eval()
        running_loss = 0.

        gt_labels = FloatTensor().to(device)
        output_labels = FloatTensor().to(device)

        with no_grad():
            for i, batch in enumerate(tqdm(val_loader, leave=False, miniters=50, mininterval=80)):
                images, labels = batch
                # images.requires_grad_(True)
                # labels.requires_grad_(True)
                outputs = model(images.to(device))
                loss = loss_fn(outputs, labels.to(device))
                running_loss += loss
    
                gt_labels = cat((gt_labels, labels.to(device)), 0)
                output_labels = cat((output_labels, outputs), 0)
    
        avg_val_loss = running_loss / len(val_loader)
        avg_val_loss_series.append(avg_val_loss.item())
        lr_scheduler.step(avg_val_loss)
    
        m_auc = calculate_mean_roc_auc(gt_labels, output_labels, class_count)
        avg_val_acc_series.append(m_auc)
        
        if avg_val_loss < best_val_loss:
            print("New best average validation loss of {} with accuracy {} during epoch {}".format(avg_val_loss, m_auc, epoch+1))
            best_val_loss = avg_val_loss
            model_path = "{}/chexnet_best_model{}.pt".format(result_folder, int(time.time()))
            save(model.state_dict(), model_path)
        else:
            print("Average loss of {} is not an improvement over {}. New model will not be saved".format(avg_val_loss, best_val_loss))
    end = datetime.now()
    print("Time to fit: {}".format(end-start))
    return avg_train_loss_series, avg_val_loss_series, avg_val_acc_series