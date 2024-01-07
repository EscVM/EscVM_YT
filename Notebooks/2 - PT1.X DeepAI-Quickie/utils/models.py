# Copyright 2022 Vittorio Mazzia. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from abc import ABC
from typing import List, Callable

import torch
from tqdm.notebook import tqdm
from utils.layers import FFLinear
from torch.nn import Module, Linear
from torch.utils.data import DataLoader
from utils.dataset_utils import TrainingDatasetFF


class FFSequentialModel(ABC):
    """Utility Class to create Sequential Model compatible with Forward-Forward Algorithm.
    It has usefull functions to train one layer at a time or all togther, and test the network.
    """

    def __init__(self):
        self.layers = []  # store layers

    def predict_accomulate_goodness(
        self, X: torch.Tensor, pos_gen_fn: Callable, n_class: int = None
    ) -> torch.Tensor:
        """Use the network to make predictions on a batch of samples. It makes use of pos_gen_fn function
        to overlay labels on samples.

        Args:
            X (torch.Tensor): batch of input samples
            pos_gen_fn (Callable): overlay labels on samples
            n_class (int, optional): number of classes. Defaults to None.

        Returns:
            torch.Tensor: batch of net predictions
        """
        goodness_per_label = []
        for label in range(n_class):
            h = pos_gen_fn(X, label, True)
            goodness = []
            for layer in self.layers:
                h = layer(h)
                goodness += [h.pow(2).mean(1)]
            goodness_per_label += [sum(goodness).unsqueeze(1)]
        goodness_per_label = torch.cat(goodness_per_label, 1)
        return goodness_per_label.argmax(1)

    def train_batch(
        self, X_pos: torch.Tensor, X_neg: torch.Tensor, before: bool = False
    ) -> List:
        """Train all network layers at the same time with the given batch of positive and negative samples.

        Args:
            X_pos (torch.Tensor): batch of positive examples
            X_neg (torch.Tensor): batch of negative examples
            before (bool, optional): successive layers get previous layers output before update. Defaults to False.

        Returns:
            List: all layer losses (pos + neg)
        """
        layers_losses = []

        for layer in self.layers:
            X_pos, X_neg, layer_loss = layer.train_layer(X_pos, X_neg, before=before)
            layers_losses.append(layer_loss)
        return layers_losses

    def train_batch_progressive(
        self, epochs: int, train_loader_progressive: DataLoader
    ) -> None:
        """Train network layers one at a time.

        Args:
            epochs (int): number of epochs per layer
            train_loader_progressive (DataLoader): dataloader with training examples. Positive and negatives
        """
        for i, layer in tqdm(enumerate(self.layers), total=len(self.layers)):
            for epoch in range(epochs):
                for X_pos, X_neg in train_loader_progressive:
                    _, _, layer_loss = layer.train_layer(X_pos, X_neg, before=False)
                    print(
                        f"Epoch: {epoch+1}/{epochs}, Layer {i}: {layer_loss}", end="\r"
                    )
            batch_size = train_loader_progressive.batch_size
            train_loader_progressive = TrainingDatasetFF(
                (layer(X_pos), layer(X_neg))
                for X_pos, X_neg in train_loader_progressive
            )

            train_loader_progressive = torch.utils.data.DataLoader(
                train_loader_progressive, batch_size=batch_size, shuffle=True
            )
            print("\n")


class FFMultiLayerPerceptron(Module, FFSequentialModel):
    """MLP model Forward-Forward compatible"""

    def __init__(
        self,
        hidden_dimensions: List,
        activation: torch.nn,
        optimizer: torch.optim,
        layer_optim_learning_rate: float,
        threshold: float,
        loss_fn: Callable,
    ):
        """Initialize MLP model

        Args:
            hidden_dimensions (List): list with hidden dimensions. First dim is the input
            activation (torch.nn): activation for each layer
            optimizer (torch.optim): layer level optimizer
            layer_optim_learning_rate (float): learning rate TODO: Future implementaton should not have it here. Create class loss
            threshold (float): loss function threshold
            loss_fn (Callable): layer level loss function
        """
        super(FFMultiLayerPerceptron, self).__init__()

        self.layers = torch.nn.ModuleList()
        for i in range(len(hidden_dimensions) - 1):
            self.layers += [
                FFLinear(
                    hidden_dimensions[i],
                    hidden_dimensions[i + 1],
                    activation,
                    optimizer,
                    layer_optim_learning_rate,
                    threshold,
                    loss_fn,
                )
            ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Model forward function

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """
        for layer in self.layers:
            x = layer(x)
        return x


class MultiLayerPerceptron(Module):
    """Classic PyTorch MLP model"""

    def __init__(self, hidden_dimensions: List, activation: torch.nn):
        """Initialize MLP model

        Args:
            hidden_dimensions (List): list of layer dimensions. First is the input dim and last output dim (n_classes)
            activation (torch.nn): activation for each layer
        """
        super(MultiLayerPerceptron, self).__init__()
        self.activation = activation
        self.layers = torch.nn.ModuleList()
        for i in range(len(hidden_dimensions) - 1):
            self.layers += [Linear(hidden_dimensions[i], hidden_dimensions[i + 1])]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Model forward function

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
        x = self.layers[-1](x)
        return x
