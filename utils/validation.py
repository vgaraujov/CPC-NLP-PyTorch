import numpy as np
import logging
import torch
import torch.nn.functional as F

## Get the same logger from main"
logger = logging.getLogger("cpc")

def validation(step, experiment, model, data_loader, device, timestep):
    with experiment.validate():
        logger.info("Starting Validation")
        model.eval()
        total_loss = {i: 0.0 for i in range(1, timestep + 1)}
        total_acc = {i: 0.0 for i in range(1, timestep + 1)}
        with torch.no_grad():
            for batch_idx, data in enumerate(data_loader):
                output = model(data.to(device))
                acc = torch.mean(output[1], 0)
                loss = torch.mean(output[0], 0)
                for i, (a, l) in enumerate(zip(acc, loss)):
                    total_loss[i+1] += l.detach().item()
                    total_acc[i+1] += a.detach().item()

        # average loss # average acc
        final_acc = sum(total_acc.values())/len(data_loader)
        final_loss = sum(total_loss.values())/len(data_loader)
        if experiment:
            experiment.log_metrics({'loss': final_loss,
                                    'acc': final_acc},
                                    step = step)
        logger.info('===> Validation set: Average loss: {:.4f}\tAccuracy: {:.4f}\n'.format(
                    final_loss, final_acc))
        
    return final_acc, final_loss


def validation_clf(cpc_model, clf_model, device, data_loader):
    logger.info("Starting Validation")
    cpc_model.eval() # not training cdc model 
    clf_model.eval()
    total_loss = 0
    total_acc  = 0 

    with torch.no_grad():
        for [data, target] in data_loader:
            data = data.to(device)
            target = target.to(device)
            embedding = cpc_model.get_sentence_embedding(data)
            output = clf_model.forward(embedding)
            total_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            total_acc += pred.eq(target.view_as(pred)).sum().item()

    total_loss /= len(data_loader.dataset) # average loss
    total_acc  /= 1.*len(data_loader.dataset) # average acc

    logger.info('===> Validation set: Average loss: {:.4f}\tAccuracy: {:.4f}\n'.format(
                total_loss, total_acc))

    return total_acc, total_loss