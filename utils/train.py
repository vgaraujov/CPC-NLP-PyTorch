import torch
import logging
import os
import torch.nn.functional as F

## Get the same logger from main"
logger = logging.getLogger("cpc")

def snapshot(dir_path, run_name, state):
    snapshot_file = os.path.join(dir_path,
                    run_name + '-model_best.pth')
    
    torch.save(state, snapshot_file)
    logger.info("Snapshot saved to {}\n".format(snapshot_file))

def train(step, experiment, model, data_loader, device, optimizer, epoch, timestep, log_interval):
    with experiment.train():
        model.train()
        total_loss = {i: 0.0 for i in range(1, timestep + 1)}
        total_acc = {i: 0.0 for i in range(1, timestep + 1)}
        for batch_idx, data in enumerate(data_loader):
            optimizer.zero_grad()
            output = model(data.to(device))
            acc = torch.mean(output[1], 0)
            loss = torch.mean(output[0], 0)
            step += 1
            for i, (a, l) in enumerate(zip(acc, loss)):
                total_loss[i+1] += l.detach().item()
                total_acc[i+1] += a.detach().item()
                if experiment:
                    experiment.log_metrics({'loss_{}'.format(i+1): total_loss[i+1]/(batch_idx+1),
                                            'acc_{}'.format(i+1): total_acc[i+1]/(batch_idx+1)},
                                            step = step)
            if experiment:
                experiment.log_metrics({'loss': sum(total_loss.values())/(batch_idx+1),
                                        'acc': sum(total_acc.values())/(batch_idx+1)},
                                        step = step)
            loss.sum().backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tAccuracy: {:.4f}\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(data_loader) * data_loader.batch_size,
                    100. * batch_idx / len(data_loader), acc.sum().detach().item(), loss.sum().detach().item()))
        # average loss # average acc
        final_acc = sum(total_acc.values())/len(data_loader)
        final_loss = sum(total_loss.values())/len(data_loader)
        logger.info('===> Training set: Average loss: {:.4f}\tAccuracy: {:.4f}'.format(
                    final_loss, final_acc))
    
    return final_acc, final_loss, step