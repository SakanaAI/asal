# Copyright (c) 2018 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.

import torch

from .activations import identity_activation, tanh_activation
from .cppn import clamp_weights_, create_cppn, get_coord_inputs


class LinearNet:
    def __init__(
        self,
        w_node,
        b_node,
        input_coords,
        output_coords,
        weight_threshold=0.2,
        weight_max=3.0,
        activation=tanh_activation,
        cppn_activation=identity_activation,
        device="mps",
    ):

        self.w_node = w_node
        self.b_node = b_node

        self.n_inputs = len(input_coords)
        if not isinstance(input_coords, torch.Tensor):
            self.input_coords = torch.tensor(
                input_coords, dtype=torch.float32, device=device
            )
        else:
            self.input_coords = input_coords.to(device=device)
            
        self.n_outputs = len(output_coords)
        self.output_coords = torch.tensor(
            output_coords, dtype=torch.float32, device=device
        )

        self.weight_threshold = weight_threshold
        self.weight_max = weight_max

        self.activation = activation
        self.cppn_activation = cppn_activation

        self.device = device
        self.reset()

    def get_init_weights(self, in_coords, out_coords, w_node):
        (x_out, y_out, z_out), (x_in, y_in, z_in) = get_coord_inputs(in_coords, out_coords)

        weights = self.cppn_activation(
            w_node(
                x_out=x_out,
                y_out=y_out,
                z_out=z_out,
                x_in=x_in,
                y_in=y_in,
                z_in=z_in,
            )
        )
        # weights += torch.randn_like(weights) * 0.5
        clamp_weights_(weights, self.weight_threshold, self.weight_max)

        return weights

    def reset(self):
        with torch.no_grad():
            self.weights = (self.get_init_weights(
                    self.input_coords, self.output_coords, self.w_node
                ).unsqueeze(0))
            bias_coords = torch.zeros(
                (1, 3), dtype=torch.float32, device=self.device)
            self.biases = self.get_init_weights(
                bias_coords, self.output_coords, self.b_node
            ).unsqueeze(0)

    def activate(self, inputs):
        """
        inputs: (batch_size, n_inputs)

        returns: (batch_size, n_outputs)
        """
        with torch.no_grad():
            if not isinstance(inputs, torch.Tensor):
                inputs = torch.tensor(
                    inputs, dtype=torch.float32, device=self.device
                ).unsqueeze(2)
            else:
                inputs = inputs.unsqueeze(2).to(device=self.device)

            outputs = self.activation(self.weights.matmul(inputs))

        return outputs.squeeze(2)

    @staticmethod
    def create(
        genome,
        config,
        input_coords,
        output_coords,
        weight_threshold=0.2,
        weight_max=3.0,
        output_activation=None,
        activation=tanh_activation,
        cppn_activation=identity_activation,
        device="cuda:0",
    ):

        nodes = create_cppn(
            genome,
            config,
            ["x_in", "y_in", "z_in", "x_out", "y_out", "z_out"],
            ["w", "b"],
            output_activation=cppn_activation,
            device=device
        )

        w_node = nodes[0]
        b_node = nodes[1]

        return LinearNet(
            w_node,
            b_node,
            input_coords,
            output_coords,
            weight_threshold=weight_threshold,
            weight_max=weight_max,
            activation=activation,
            cppn_activation=cppn_activation,
            device=device,
        )
