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