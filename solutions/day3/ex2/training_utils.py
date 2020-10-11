
import os
import sys
sys.path.append(os.path.abspath('utils'))
import utils
import torch

"""## Training"""

# apply training for one epoch
def train(model, loader, optimizer, loss_function,
          epoch, log_interval=100, tb_logger=None, device='cpu'):

    # set the model to train mode
    model.train()
    # iterate over the batches of this epoch
    for batch_id, (x, y) in enumerate(loader):
        # move input and target to the active device (either cpu or gpu)
        x, y = x.to(device), y.to(device)
        
        # zero the gradients for this iteration
        optimizer.zero_grad()
        
        # apply model, calculate loss and run backwards pass
        prediction = model(x)
        loss = loss_function(prediction, y)
        loss.backward()
        
        # perform a single optimization step
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
                
                x, y, prediction = utils.clip_to_uint8(x), utils.clip_to_uint8(y), utils.clip_to_uint8(prediction)
                tb_logger.add_images(tag='input', img_tensor=x.to('cpu'), global_step=step)
                tb_logger.add_images(tag='target', img_tensor=y.to('cpu'), global_step=step)
                tb_logger.add_images(tag='prediction', img_tensor=prediction.to('cpu').detach(), global_step=step)

# run validation after training epoch
def validate(model, loader, loss_function, metric, step=None, tb_logger=None,  device='cpu'):
    # set model to eval mode
    model.eval()
    # running loss and metric values
    val_loss = 0
    val_metric = 0
    
    # disable gradients during validation
    with torch.no_grad():
        # iterate over validation loader and update loss and metric values
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            prediction = model(x)
            loss = loss_function(prediction, y)
            eval_score = metric(y, prediction)
            
            val_loss += loss
            val_metric += eval_score
    
    # normalize loss and metric
    val_loss /= len(loader)
    val_metric /= len(loader)
    
    if tb_logger is not None:
        assert step is not None, "Need to know the current step to log validation results"
        tb_logger.add_scalar(tag='val_loss', scalar_value=val_loss, global_step=step)
        tb_logger.add_scalar(tag='val_metric', scalar_value=val_metric, global_step=step)
        # we always log the last validation images
        x, y, prediction = utils.clip_to_uint8(x), utils.clip_to_uint8(y), utils.clip_to_uint8(prediction)
        tb_logger.add_images(tag='val_input', img_tensor=x.to('cpu'), global_step=step)
        tb_logger.add_images(tag='val_target', img_tensor=y.to('cpu'), global_step=step)
        tb_logger.add_images(tag='val_prediction', img_tensor=prediction.to('cpu'), global_step=step)
        
    print('\nValidate: Average loss: {:.4f}, Average Metric: {:.4f}\n'.format(val_loss, val_metric))

