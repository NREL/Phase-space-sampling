import sys

import torch
from torch.utils import data

import uips.nn as nn_
from uips.nde import distributions, flows, transforms
from uips.utils.torchutils import (
    create_alternating_binary_mask,
    create_random_binary_mask,
)


def create_base_transform(
    i,
    base_transform_type,
    hidden_features=64,
    num_blocks=5,
    tail_bound=5,
    dim=2,
    num_bins=16,
):
    if base_transform_type == "affine":
        return transforms.AffineCouplingTransform(
            mask=create_alternating_binary_mask(
                features=dim, even=(i % 2 == 0)
            ),
            transform_net_create_fn=lambda in_features, out_features: nn_.ResidualNet(
                in_features=in_features,
                out_features=out_features,
                hidden_features=hidden_features,
                num_blocks=num_blocks,
                use_batch_norm=True,
            ),
        )
    else:
        return transforms.PiecewiseRationalQuadraticCouplingTransform(
            mask=create_alternating_binary_mask(
                features=dim, even=(i % 2 == 0)
            ),
            transform_net_create_fn=lambda in_features, out_features: nn_.ResidualNet(
                in_features=in_features,
                out_features=out_features,
                hidden_features=hidden_features,
                num_blocks=num_blocks,
                use_batch_norm=True,
            ),
            tails="linear",
            tail_bound=5,
            num_bins=num_bins,
            apply_unconditional_transform=False,
        )


def create_base_transform_random(
    i,
    base_transform_type,
    hidden_features=64,
    num_blocks=5,
    tail_bound=5,
    dim=2,
    num_bins=16,
):
    if base_transform_type == "affine":
        return transforms.AffineCouplingTransform(
            mask=create_random_binary_mask(features=dim),
            transform_net_create_fn=lambda in_features, out_features: nn_.ResidualNet(
                in_features=in_features,
                out_features=out_features,
                hidden_features=hidden_features,
                num_blocks=num_blocks,
                use_batch_norm=True,
            ),
        )
    else:
        return transforms.PiecewiseRationalQuadraticCouplingTransform(
            mask=create_random_binary_mask(features=dim),
            transform_net_create_fn=lambda in_features, out_features: nn_.ResidualNet(
                in_features=in_features,
                out_features=out_features,
                hidden_features=hidden_features,
                num_blocks=num_blocks,
                use_batch_norm=True,
            ),
            tails="linear",
            tail_bound=5,
            num_bins=num_bins,
            apply_unconditional_transform=False,
        )
