import os

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from FOTS.utils.metrics import precision_recall_f1

class Trainer:
    def __init__(self, model, optimizer=None, lr_scheduler=None,  start_epoch=0,
        num_epochs=10, best_score=float('inf'), monitor='val_detect_loss', 
        checkpoint_freq=1, checkpoint_dir='./checkpoints', log_dir='./logs'):
        """
        Args:
            start_epoch (int): The starting number of epoch count.
            num_epochs (int): Maximum number of epochs to run.
            best_score (float): Lowest validation loss seen during training.
            monitor (str): Quantity to be monitored.
            checkpoint_freq (int): How often should a checkpoint be created 
                (in epoch).
            checkpoint_dir (str): Directory for storing checkpoints.
            log_dir (str): Directory for storing logs.
        """
        self.model = model
        if isinstance(self.model, nn.DataParallel):
            self.model = self.model.module
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.start_epoch = start_epoch
        self.num_epochs = num_epochs
        self.best_score = best_score
        self.monitor = monitor
        self.checkpoint_freq = checkpoint_freq
        self.checkpoint_dir = checkpoint_dir
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.writer = SummaryWriter(log_dir)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def tensor_to_device(self, *args):
        return [x.to(self.device) if isinstance(x, torch.Tensor) else x for x in args]

    def fit(self, train_dataloader, val_dataloader=None):
        for epoch in range(self.start_epoch, self.start_epoch + self.num_epochs):
            # training
            running_train_loss = {}
            train_sample_size = 0
            train_pbar = tqdm(train_dataloader)
            train_pbar.set_description(f'Epoch {epoch + 1}')
            self.model.train()
            for batch in train_pbar:
                # zero the parameter gradients
                self.optimizer.zero_grad()
                # put data in GPU (if available)
                batch = self.tensor_to_device(*batch)
                # forward pass
                train_outputs = self.model.training_step(batch)
                # backpropagation
                train_outputs['loss'].backward()
                # optimize
                self.optimizer.step()
                # accumulate the loss
                train_sample_size += len(batch[0]) # batch[0] are img_filenames
                for key, value in train_outputs['log'].items():
                    running_train_loss[key] = running_train_loss.get(key, 0) + value
                train_pbar.set_postfix(train_outputs['log'])

            # adjust learning rate
            self.lr_scheduler.step(train_outputs['loss'], epoch)

            # write the average loss to Tensorboard
            for key in running_train_loss:
                running_train_loss[key] /= train_sample_size
                self.writer.add_scalar(f'train/{key}', running_train_loss[key], epoch)

            # save the model
            if self.checkpoint_freq > 0 and (epoch + 1) % self.checkpoint_freq == 0:
                self.save_checkpoint(f'epoch{epoch}', epoch)
            
            # validation
            if not val_dataloader:
                continue
            running_val_loss = {}
            val_sample_size = 0
            val_pbar = tqdm(val_dataloader)
            val_pbar.set_description(f'Epoch {epoch + 1}')
            self.model.eval()
            with torch.no_grad():
                for batch in val_pbar:
                    batch = self.tensor_to_device(*batch)
                    val_outputs = self.model.validation_step(batch)
                    val_sample_size += len(batch[0])
                    for key, value in val_outputs['log'].items():
                        running_val_loss[key] = running_val_loss.get(key, 0) + value
                    val_pbar.set_postfix(val_outputs['log'])
            for key in running_val_loss:
                running_val_loss[key] /= val_sample_size
                self.writer.add_scalar(f'val/{key}', running_val_loss[key], epoch)
            if self.best_score > running_val_loss[self.monitor]:
                self.best_score = running_val_loss[self.monitor]
                self.save_checkpoint('best_score', epoch)

        self.save_checkpoint(f'epoch{epoch}_last_epoch', epoch)
        self.writer.close()

    def test(self, test_dataloader):
        true_positives = 0
        false_positives = 0
        num_of_gt_bboxes = 0
        self.model.eval()
        with torch.no_grad():
            for batch in test_dataloader:
                batch = self.tensor_to_device(*batch)
                test_outputs = self.model.test_step(batch)
                true_positives += test_outputs['true_positives']
                false_positives += test_outputs['false_positives']
                num_of_gt_bboxes += test_outputs['num_of_gt_bboxes']
        return precision_recall_f1(true_positives, false_positives, num_of_gt_bboxes)

    def save_checkpoint(self, filename, epoch):
        state_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'trainer_config': {
                'start_epoch': epoch + 1,
                'best_score': self.best_score
            }
        }
        torch.save(state_dict, os.path.join(self.checkpoint_dir, f'{filename}.pt'))