# Copyright 2019 Ondrej Skopek.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .vae import ModelVAE
from ...data import VaeDataset
from ..components import Component


class FeedForwardVAE(ModelVAE):

    def __init__(self, h_dim: int, depth: int, norm_method: str, input_norm: bool, dropout: float,
                 components: List[Component], dataset: VaeDataset, scalar_parametrization: bool) -> None:
        super().__init__(h_dim, components, dataset, scalar_parametrization)

        self.in_dim = dataset.in_dim
        self.dropout = dropout

        # init model list
        self.f_enc = nn.ModuleList()
        self.f_dec = nn.ModuleList()
        self.f_enc_norm = nn.ModuleList()
        self.f_dec_norm = nn.ModuleList()

        assert norm_method in ['bn', 'ln', 'None']

        if norm_method == 'bn':
            if input_norm:
                self.f_enc_norm.append(nn.BatchNorm1d(self.in_dim))
                self.f_dec_norm.append(nn.BatchNorm1d(self.total_z_dim))
            else:
                self.f_enc_norm.append(nn.Identity())
                self.f_dec_norm.append(nn.Identity())
            if depth == 1:
                # just linear layer i.e. logistic regression
                self.f_enc.append(nn.Linear(self.in_dim, h_dim))
                self.f_dec.append(nn.Linear(self.total_z_dim, self.in_dim))
            else:
                self.f_enc.append(nn.Linear(self.in_dim, h_dim))
                self.f_dec.append(nn.Linear(self.total_z_dim, h_dim))

                self.f_enc_norm.append(nn.BatchNorm1d(h_dim))
                self.f_dec_norm.append(nn.BatchNorm1d(h_dim))
                for _ in range(depth - 2):
                    self.f_enc.append(nn.Linear(h_dim, h_dim))
                    self.f_dec.append(nn.Linear(h_dim, h_dim))

                    self.f_enc_norm.append(nn.BatchNorm1d(h_dim))
                    self.f_dec_norm.append(nn.BatchNorm1d(h_dim))

                self.f_enc.append(nn.Linear(h_dim, h_dim))
                self.f_dec.append(nn.Linear(h_dim, self.in_dim))
        elif norm_method == 'ln':
            if input_norm:
                self.f_enc_norm.append(nn.LayerNorm(self.in_dim))
                self.f_dec_norm.append(nn.LayerNorm(self.total_z_dim))
            else:
                self.f_enc_norm.append(nn.Identity())
                self.f_dec_norm.append(nn.Identity())
            if depth == 1:
                # just linear layer i.e. logistic regression
                self.f_enc.append(nn.Linear(self.in_dim, h_dim))
                self.f_dec.append(nn.Linear(self.total_z_dim, self.in_dim))
            else:
                self.f_enc.append(nn.Linear(self.in_dim, h_dim))
                self.f_dec.append(nn.Linear(self.total_z_dim, h_dim))

                self.f_enc_norm.append(nn.LayerNorm(h_dim))
                self.f_dec_norm.append(nn.LayerNorm(h_dim))
                for _ in range(depth - 2):
                    self.f_enc.append(nn.Linear(h_dim, h_dim))
                    self.f_dec.append(nn.Linear(h_dim, h_dim))

                    self.f_enc_norm.append(nn.LayerNorm(h_dim))
                    self.f_dec_norm.append(nn.LayerNorm(h_dim))

                self.f_enc.append(nn.Linear(h_dim, h_dim))
                self.f_dec.append(nn.Linear(h_dim, self.in_dim))
        else:
            self.f_enc_norm.append(nn.Identity())
            self.f_dec_norm.append(nn.Identity())
            if depth == 1:
                # just linear layer i.e. logistic regression
                self.f_enc.append(nn.Linear(self.in_dim, h_dim))
                self.f_dec.append(nn.Linear(self.total_z_dim, self.in_dim))
            else:
                self.f_enc.append(nn.Linear(self.in_dim, h_dim))
                self.f_dec.append(nn.Linear(self.total_z_dim, h_dim))

                self.f_enc_norm.append(nn.Identity())
                self.f_dec_norm.append(nn.Identity())
                for _ in range(depth - 2):
                    self.f_enc.append(nn.Linear(h_dim, h_dim))
                    self.f_dec.append(nn.Linear(h_dim, h_dim))

                    self.f_enc_norm.append(nn.Identity())
                    self.f_dec_norm.append(nn.Identity())

                self.f_enc.append(nn.Linear(h_dim, h_dim))
                self.f_dec.append(nn.Linear(h_dim, self.in_dim))

    def encode(self, x: Tensor) -> Tensor:
        assert len(x.shape) == 2
        bs, dim = x.shape
        assert dim == self.in_dim
        x = x.view(bs, self.in_dim)
        x = self.f_enc_norm[0](x)
        for i, lin in enumerate(self.f_enc[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.f_enc_norm[i + 1](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.f_enc[-1](x)
        return x.view(bs, -1)

    def decode(self, concat_z: Tensor) -> Tensor:
        assert len(concat_z.shape) >= 2
#         print(concat_z.shape)
        bs = concat_z.size(-2)
        x = self.f_dec_norm[0](concat_z)
        for i, lin in enumerate(self.f_dec[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.f_dec_norm[i + 1](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.f_dec[-1](x)
        x = x.view(-1, bs, self.in_dim)  # flatten
        return x.squeeze(dim=0)  # in case we're not doing LL estimation
