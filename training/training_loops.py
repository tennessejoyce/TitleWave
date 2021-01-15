import torch
from time import time
import numpy as np

def fit(model, data_loader, optimizer, loss_function, scheduler,
        train_steps_per_epoch, val_steps_per_epoch, num_epochs):
    """
    Training loop for a pytorch model. Uses half precision when possible with AMP.
    """
    steps_per_epoch = train_steps_per_epoch + val_steps_per_epoch
    scaler = torch.cuda.amp.GradScaler()
    for i, (input_ids, attention_mask, labels) in enumerate(data_loader):
        i_mod = i % steps_per_epoch
        if i_mod == 0:
            # Beginning of epoch
            train_loss = 0
            val_loss = 0
            train_loss_sq = 0
            val_loss_sq = 0
            epoch_start_time = time()
            epoch = int(i/steps_per_epoch) + 1
        if i_mod < train_steps_per_epoch:
            # Training
            model.train()
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                output = model(input_ids, attention_mask).logits[:,0]
                loss = loss_function(output, labels)
            train_loss += loss.item()
            train_loss_sq += loss.item()**2
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Validation
            model.eval()
            with torch.no_grad():
                output = model(input_ids, attention_mask).logits[:,0]
                loss = loss_function(output, labels)
                val_loss += loss.item()
                val_loss_sq += loss.item()**2
        if i_mod == steps_per_epoch - 1:
            # End of epoch

            # Calculate the average loss, and the standard error on that estimate.
            train_loss = train_loss/train_steps_per_epoch
            train_loss_std_error = np.sqrt(train_loss_sq/train_steps_per_epoch - train_loss**2)/np.sqrt(train_steps_per_epoch)
            val_loss = val_loss/val_steps_per_epoch
            val_loss_std_error = np.sqrt(val_loss_sq/train_steps_per_epoch - val_loss**2)/np.sqrt(val_steps_per_epoch)
            
            # Print the results to track progress
            print(f'Finished Epoch {epoch}/{num_epochs} in {time() - epoch_start_time:.1f} s')
            print(f'Train (Validation) Loss: {train_loss:.3f} +/- {train_loss_std_error:.3f} ({val_loss:.3f} +/- {val_loss_std_error:.3f})')
            print()
            if epoch>=num_epochs:
                # Stop training after specified number of epochs
                break
            # Adjust the learning rate according to the scheduler
            scheduler.step()






