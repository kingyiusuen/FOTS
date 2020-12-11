import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Recognizer(nn.Module):
    def __init__(self, num_of_classes):
        super().__init__()
        self.cnn = CNN(in_channels=64)
        self.bilstm = BiDirectionalLSTM(num_of_classes=num_of_classes)
    
    def forward(self, rois, seq_lens):
        x = self.cnn(rois)
        # height of x is 1 so we can remove that dimension
        x = x.squeeze(dim=2)
        # as described in the FOTS paper, permute the extracted higher-level 
        # feature maps from CNN to time major form before feeding into BiLSTM 
        # for encoding
        x = x.permute(0, 2, 1)
        x = self.bilstm(x, seq_lens)
        return x

class CNN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.vgg_block1 = self._vgg_block(2, in_channels, 64)
        self.vgg_block2 = self._vgg_block(2, 64, 128)
        self.vgg_block3 = self._vgg_block(2, 128, 256)
    
    def forward(self, x):
        """
        Args:
            x (Tensor): (N, 64, H, W) where N is the number of ROIs.
        
        Returns:
            x (Tensor): (N, 256, 1, W).
        """
        x = self.vgg_block1(x)
        x = self.vgg_block2(x)
        x = self.vgg_block3(x)
        return x
    
    def _vgg_block(self, num_convs, in_channels, out_channels):
        layers = []
        for _ in range(num_convs):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
            in_channels = out_channels
        # height-max pooling to reduce feature dimension along height axis
        layers.append(nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)))
        return nn.Sequential(*layers)

class BiDirectionalLSTM(nn.Module):
    def __init__(self, num_of_classes, input_size=256, hidden_size=256):
        super().__init__()
        self.bilstm = nn.LSTM(input_size, hidden_size, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_of_classes)

    def forward(self, seqs, seq_lens):
        """
        Args:
            seqs (Tensor): (N, W, 256).
            seq_lens (list): (N). The bounding box widths before padding.

        Returns:
            x (Tensor): (N).
        """
        # https://discuss.pytorch.org/t/why-do-we-need-flatten-parameters-when-using-rnn-with-dataparallel/46506
        self.bilstm.flatten_parameters()
        # https://discuss.pytorch.org/t/why-do-we-need-to-pack-padded-batches-of-sequences-in-pytorch/47977
        total_length = seqs.shape[1]  # get the max sequence length
        packed_seq = pack_padded_sequence(seqs, seq_lens, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.bilstm(packed_seq)
        unpacked_output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=total_length)
        num_of_rois, total_length, hidden_size = list(unpacked_output.shape)
        x = unpacked_output.contiguous().view(total_length * num_of_rois, hidden_size)
        x = self.fc(x)
        x = x.view(num_of_rois, total_length, -1)
        log_probs = F.log_softmax(x, dim=-1) # -1 is the last dimension
        log_probs = log_probs.permute(1, 0, 2) # required by CTCLoss()
        return log_probs