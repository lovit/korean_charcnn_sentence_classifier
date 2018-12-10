import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class KorCharCNN(nn.Module):

    def __init__(self, embedding_dim=80, num_filters=100, windows=[2, 3, 4, 5], num_classes=2):

        super(KorCharCNN, self).__init__()

        # in_channels, out_channels, 
        self.convs = [nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=(n, embedding_dim), bias=False) for n in windows]
        for idx, conv in enumerate(self.convs):
            self.add_module('conv%d'%idx, conv)
        self.fc = nn.Linear(len(windows) * num_filters, num_classes, bias=False)
        self.ranges = np.asarray([n for n in windows for _ in range(num_filters)])

    def forward(self, x):
        # apply convolutional layer
        out = [F.relu(conv(x)).squeeze(3) for conv in self.convs]

        # max pooling for each filter
        out = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in out]

        # concatenation
        out = torch.cat(out, 1)

        # fully connected neural network
        logit = self.fc(out) # (batch, target_size)

        return logit