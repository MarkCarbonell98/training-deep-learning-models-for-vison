
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import torch
import numpy as np


# apply training for one epoch
def train(model, loader, optimizer, loss_function,
        epoch, log_interval=50, log_image_interval=20, tb_logger=None, device='cpu'):

    # set the model to train mode
    model.train()

    # iterate over the batches of this epoch
    for batch_id, (x, y) in enumerate(loader):
        # move input and target to the active device (either cpu or gpu)
        x = x.to(device)
        y = y.to(device)
        
        # zero the gradients for this iteration
        optimizer.zero_grad()
        
        # apply model, calculate loss and run backwards pass
        pred = model(x)

        loss = loss_function(pred, y)
        loss.backward()

        optimizer.step()
        
        # log to console
        if batch_id % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_id * len(x),
                len(loader.dataset),
                100. * batch_id / len(loader), loss.item()))

    # log to tensorboard
        if tb_logger is not None:
            step = epoch * len(loader) + batch_id
            tb_logger.add_scalar(tag='train_loss', scalar_value=loss.item(), global_step=step)
            # check if we log images in this iteration
            if step % log_image_interval == 0:
                tb_logger.add_images(tag='input', img_tensor=x.to('cpu'), global_step=step)
                tb_logger.add_images(tag='target', img_tensor=y.to('cpu'), global_step=step)
                tb_logger.add_images(tag='prediction', img_tensor=pred.to('cpu').detach(), global_step=step)



# run validation after training epoch
def validate(model, loader, loss_function, metric, step=None, tb_logger=None, device='cpu', optimizer=None, name=''):
    # set model to eval mode
    model.eval()
    # running loss and metric values
    val_loss = 0
    val_metric = 0

    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=.5, patience=1)

    # disable gradients during validation
    pred = 0
    total_val_metric = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)

            loss = loss_function(pred, y)
            eval_score = metric(pred, y)

            val_loss += loss
            val_metric += eval_score
            eval_score_copy = eval_score.detach().clone()

            scheduler.step(eval_score_copy)


    # normalize loss and metric
    val_loss /= len(loader)
    val_metric /= len(loader)
    
    if tb_logger is not None:
        assert step is not None, "Need to know the current step to log validation results"
        tb_logger.add_scalar(tag='val_loss', scalar_value=val_loss, global_step=step)
        tb_logger.add_scalar(tag='val_metric', scalar_value=val_metric, global_step=step)
        # we always log the last validation images
        tb_logger.add_images(tag='val_input', img_tensor=x.to('cpu'), global_step=step)
        tb_logger.add_images(tag='val_target', img_tensor=y.to('cpu'), global_step=step)
        tb_logger.add_images(tag='val_prediction', img_tensor=pred.to('cpu'), global_step=step)
        
    print('\nValidate: Average loss: {:.4f}, Average Metric: {:.4f}\n'.format(val_loss, val_metric))
    return val_loss, val_metric

