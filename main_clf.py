## Utilities
import time
import os
import logging
import yaml
from timeit import default_timer as timer

## Libraries
import numpy as np
from box import box_from_file
from pathlib import Path

## Torch
import torch
import torch.nn as nn
from torch.utils import data
import torch.optim as optim

## Custom Imports
from utils.logger import setup_logs
from utils.seed import set_seed
from utils.train import train_clf, snapshot
from utils.validation import validation_clf
from utils.dataset import SentimentAnalysis
from model.models import CPCv1, TxtClassifier

class ScheduledOptim(object):
    """A simple wrapper class for learning rate scheduling"""
    def __init__(self, optimizer, n_warmup_steps):
        self.optimizer = optimizer
        self.d_model = 128 
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0 
        self.delta = 1

    def state_dict(self):
        self.optimizer.state_dict()

    def step(self):
        """Step by the inner optimizer"""
        self.optimizer.step()

    def zero_grad(self):
        """Zero out the gradients by the inner optimizer"""
        self.optimizer.zero_grad()

    def increase_delta(self):
        self.delta *= 2

    def update_learning_rate(self):
        """Learning rate scheduling per step"""

        self.n_current_steps += self.delta
        new_lr = np.power(self.d_model, -0.5) * np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr

############ Control Center and Hyperparameter ###############
run_name = "cpc-clf" + time.strftime("-%Y-%m-%d_%H_%M_%S")
config_encoder = box_from_file(Path('config_cpc.yaml'), file_type='yaml')
config = box_from_file(Path('config_clf.yaml'), file_type='yaml')
global_timer = timer() # global timer
logger = setup_logs(config.training.logging_dir, run_name) # setup logs
logger.info('### Experiment {} ###'.format(run_name))
logger.info('### Hyperparameter summary below ###\n {}\n'.format(config))

use_cuda = not config.training.no_cuda and torch.cuda.is_available()
print('use_cuda is', use_cuda)
device = torch.device("cuda" if use_cuda else "cpu")
# set seed for reproducibility
set_seed(config.training.seed, use_cuda)
# Load pretrained CPC model
cpc_model = CPCv1(config=config_encoder)
checkpoint = torch.load(config.txt_classifier.cpc_path)
cpc_model.load_state_dict(checkpoint['state_dict'])
cpc_model.to(device)
# Change Embedding layer with the expanded one
if config.txt_classifier.expanded_vocab:
    embeddings = np.load('vocab_expansion/embeddings_expanded.npy')
    cpc_model.embedding = nn.Embedding.from_pretrained(torch.from_numpy(embeddings).float(), padding_idx=config.dataset_classifier.padding_idx).to(device)
    config.dataset_classifier.vocab_file_path = 'vocab_expansion/vocab_cpc_expanded.pkl'
# freeze weights    
for param in cpc_model.parameters():
    param.requires_grad = False
    
txt_model = TxtClassifier(config).to(device)

## Loading the dataset
logger.info('===> loading train, validation and test dataset')
training_set = SentimentAnalysis(config,'train')
testing_set = SentimentAnalysis(config,'test')
validation_set = SentimentAnalysis(config,'dev')
# create dataloader
train_loader = data.DataLoader(training_set, batch_size=config.training.batch_size, shuffle=True) # set shuffle to True
validation_loader = data.DataLoader(validation_set, batch_size=config.training.batch_size, shuffle=False) # set shuffle to False
test_loader = data.DataLoader(testing_set, batch_size=config.training.batch_size, shuffle=False) # set shuffle to False

# optimizer  
optimizer = ScheduledOptim(
    optim.Adam(
        filter(lambda p: p.requires_grad, txt_model.parameters()), 
        betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-4, amsgrad=True),
    config.training.n_warmup_steps)

model_params = sum(p.numel() for p in txt_model.parameters() if p.requires_grad)
logger.info('### Model summary below###\n {}\n'.format(str(txt_model)))
logger.info('===> Model total parameter: {}\n'.format(model_params))

## Start training
best_acc = 0
best_loss = np.inf
best_epoch = -1 

for epoch in range(1, config.training.epochs + 1):
    epoch_timer = timer()
    # Train and validate
    train_clf(cpc_model, txt_model, device, train_loader, optimizer, epoch, config.training.log_interval)
    val_acc, val_loss = validation_clf(cpc_model, txt_model, device, validation_loader)
    # Save
    if val_acc > best_acc: 
        best_acc = max(val_acc, best_acc)
        snapshot(config.training.logging_dir, run_name, {
            'epoch': epoch,
            'validation_acc': val_acc,
            'validation_loss': val_loss,
            'state_dict': txt_model.state_dict(),
            'optimizer': optimizer.state_dict(),
        })
        best_epoch = epoch
    elif epoch - best_epoch > 2:
        optimizer.increase_delta()
        best_epoch = epoch
    end_epoch_timer = timer()
    logger.info("#### End epoch {}/{}, elapsed time: {}".format(epoch, config.training.epochs, end_epoch_timer - epoch_timer))

logger.info("################## Success #########################")
logger.info("#### Best Validation Accuracy: {}, Epoch: {}".format(best_acc, best_epoch))

## evaluation on test set 
logger.info('===> loading best model for test evaluation')
checkpoint = torch.load(os.path.join(config.training.logging_dir, run_name + '-model_best.pth'))
txt_model.load_state_dict(checkpoint['state_dict'])
logger.info("############## Results on Test Set #################")    
_, _ = validation_clf(cpc_model, txt_model, device, test_loader)

## end 
end_global_timer = timer()
logger.info("Total elapsed time: %s" % (end_global_timer - global_timer))