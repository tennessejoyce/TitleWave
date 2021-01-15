import torch
from time import time


def fit(model, data_loader, optimizer, loss_function, scheduler,
        train_steps_per_epoch, val_steps_per_epoch, num_epochs):
    """
    Training loop for a pytorch model. Uses half precision when possible with AMP.
    """
    steps_per_epoch = train_steps_per_epoch + val_steps_per_epoch
    scaler = torch.cuda.amp.GradScaler()
    for i, (X, y) in enumerate(data_loader):
        print(X.shape)
        print(y.shape)
        i_mod = i % steps_per_epoch
        if i_mod == 0:
            # Beginning of epoch
            train_loss = 0
            val_loss = 0
            epoch_start_time = time()
        if i_mod < train_steps_per_epoch:
            # Training
            model.train()
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                output = model(X)
                loss = loss_function(output, y, w)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Validation
            model.eval()
            with torch.no_grad():
                output = model(X)
                loss = loss_function(output, y, w)
        if i_mod == train_steps_per_epoch - 1:
            # End of epoch
            print(f'Finished Epoch {steps_per_epoch + 1}/{num_epochs} in {time() - epoch_start_time:.1f} s')
            print(f'Train loss: {train_loss/train_steps_per_epoch}')
            print(f'Validation loss: {val_loss/train_steps_per_epoch}')
            print()
            # Adjust the learning rate according to the scheduler
            scheduler.step()






