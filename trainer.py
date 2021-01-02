import os
from datetime import datetime
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from FOTS.model.loss import DetectionLoss 

class Trainer:
    def __init__(self, model, optimizer, lr_scheduler, num_epochs=1,
        patience=2, ckpt_filename='checkpoint.pt', log_dir='./logs', 
        start_epoch=0, min_val_loss=float('inf'), no_improve_count=0):
        """
        Args:
            num_epochs (int): Maximum number of epochs to run.
            patience (int): How many epochs to wait to terminate training
                after last time validation loss improved.
            ckpt_filename (str): Checkpoint filename.
            log_dir (str): Directory for storing logs.
            start_epoch (int): The starting number of epoch count.
            min_val_loss (float): Lowest validation loss seen during training.
            no_improve_count (int): How many epoch has the validation loss 
                stops improving.
        """
        self.model = model
        if isinstance(self.model, nn.DataParallel):
            self.model = self.model.module
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.start_epoch = start_epoch
        self.num_epochs = num_epochs
        self.patience = patience
        self.ckpt_filename = ckpt_filename
        self.log_dir = log_dir
        self.min_val_loss = min_val_loss
        self.no_improve_count = no_improve_count
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = DetectionLoss()

    def train(self, train_dataloader, val_dataloader):
        dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}
        self.model.to(self.device)
        writer = SummaryWriter(os.path.join(self.log_dir, self.get_curr_time()))

        for epoch in range(self.start_epoch + self.num_epochs):
            epoch_loss = {"train": {}, "val": {}}
            for phase in ["train", "val"]:
                if phase == "train":
                    self.model.train()
                else:
                    self.model.eval()
                
                pbar = tqdm(dataloaders_dict[phase], desc=f"Epoch {epoch} ({phase})")
                for batch in pbar:
                    # put data in GPU (if available)
                    batch = [x.to(self.device) if isinstance(x, torch.Tensor) else x for x in batch]
                    _, imgs, _, _, _, *detect_true, training_masks = batch
                    # zero the parameter gradients
                    self.optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == "train"): # enable gradient tracking if training
                        # forward pass
                        detect_pred = self.model(imgs)
                        # compute loss values
                        loss_dict = self.criterion(*detect_pred, *detect_true, training_masks)
                        if phase == "train":
                             # backpropagation
                            loss_dict["detect_loss"].backward()
                            # optimize
                            self.optimizer.step()

                    # now that we have done backprogagation, we just need the 
                    # values of the loss
                    loss_dict = {loss: val.item() for loss, val in loss_dict.items()}

                    # accumulate the loss
                    for loss, val in loss_dict.items():
                        epoch_loss[phase][loss] = epoch_loss[phase].get(loss, 0)
                    pbar.set_postfix(loss_dict)
                
                # log the loss values (averaged over all batches in this epoch)
                # to tensorboard
                for loss, val in epoch_loss[phase].items():
                    val /= len(dataloaders_dict[phase].dataset)
                    writer.add_scalar(f"{phase}/{loss}", val, epoch)

            # adjust learning rate
            self.lr_scheduler.step()

            # check if the validation loss stops improving
            if self.early_stopping(epoch_loss["val"]["detect_loss"], epoch):
                break
            
        writer.close()

    def early_stopping(self, val_loss, epoch):
        if self.min_val_loss > val_loss:
            self.min_val_loss = val_loss
            self.no_improve_count = 0
            self.save_checkpoint(epoch)
        else:
            self.no_improve_count += 1
            if self.no_improve_count == self.patience:
                print(f"\nTraining is terminated because validation loss has stopped decreasing for {self.patience} epochs.")
                return True
        return False

    def save_checkpoint(self, epoch):
        state_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'trainer_config': {
                'start_epoch': epoch + 1,
                'min_val_loss': self.min_val_loss,
                'no_improve_count': self.no_improve_count
            }
        }
        torch.save(state_dict, self.ckpt_filename)

    def get_curr_time(self):
        return datetime.now().strftime("%Y%m%d-%H%M%S")