# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Transformer speech recognition model (pytorch)."""

import logging
import numpy
import torch

from espnet.nets.pytorch_backend.ctc import CTC
from espnet.nets.pytorch_backend.nets_utils import (
    make_non_pad_mask,
    th_accuracy,
)
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.decoder import Decoder
from espnet.nets.pytorch_backend.transformer.encoder import Encoder
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import LabelSmoothingLoss
from espnet.nets.pytorch_backend.transformer.mask import target_mask


class DynamicFeatureModule(torch.nn.Module):
    """
    Module to compute and fuse dynamic features (velocity and acceleration)
    Idea 2: Learning dynamic changes (lip speed, acceleration)
    """
    def __init__(self, input_channels=1):
        super().__init__()
        # Project concatenated features (Original + Delta + DeltaDelta) back to original channels
        self.projection = torch.nn.Conv3d(input_channels * 3, input_channels, kernel_size=1)
        
        # Initialization (Suggestion 2)
        # Initialize to Identity-like mapping to preserve original features initially
        # Weights: [Out, In, K, K, K] -> [C, 3C, 1, 1, 1]
        torch.nn.init.zeros_(self.projection.weight)
        torch.nn.init.zeros_(self.projection.bias)
        
        # Set weights for the original features (first C channels) to Identity (1.0)
        # This ensures that at start, output ~= input, and deltas are ignored until learned
        for i in range(input_channels):
            self.projection.weight.data[i, i, 0, 0, 0] = 1.0

    def forward(self, x):
        # Expecting 5D input: [Batch, Channel, Time, Height, Width]
        if x.dim() != 5:
            return x
            
        # 1. Compute First Order Delta (Velocity)
        # Pad time dimension to maintain size
        x_pad = torch.nn.functional.pad(x, (0, 0, 0, 0, 1, 1), mode='replicate')
        delta = x_pad[:, :, 2:] - x_pad[:, :, :-2] # Central difference
        
        # 2. Compute Second Order Delta (Acceleration)
        d_pad = torch.nn.functional.pad(delta, (0, 0, 0, 0, 1, 1), mode='replicate')
        delta2 = d_pad[:, :, 2:] - d_pad[:, :, :-2]
        
        # 3. Concatenate and Project
        # [B, 3*C, T, H, W]
        combined = torch.cat([x, delta, delta2], dim=1)
        
        return self.projection(combined)


class E2E(torch.nn.Module):
    def __init__(self, odim, args, ignore_id=-1):
        torch.nn.Module.__init__(self)

        # Initialize Dynamic Feature Module
        # Only used during inference, disabled during training
        self.dynamic_feature_module = DynamicFeatureModule(input_channels=1)
        self.use_dynamic_features = False  # Disabled during training, enable for inference

        self.encoder = Encoder(
            attention_dim=args.adim,
            attention_heads=args.aheads,
            linear_units=args.eunits,
            num_blocks=args.elayers,
            input_layer=args.transformer_input_layer,
            dropout_rate=args.dropout_rate,
            positional_dropout_rate=args.dropout_rate,
            attention_dropout_rate=args.transformer_attn_dropout_rate,
            encoder_attn_layer_type=args.transformer_encoder_attn_layer_type,
            macaron_style=args.macaron_style,
            use_cnn_module=args.use_cnn_module,
            cnn_module_kernel=args.cnn_module_kernel,
            zero_triu=getattr(args, "zero_triu", False),
            a_upsample_ratio=args.a_upsample_ratio,
            relu_type=getattr(args, "relu_type", "swish"),
        )

        self.transformer_input_layer = args.transformer_input_layer
        self.a_upsample_ratio = args.a_upsample_ratio

        self.proj_decoder = None
        if args.adim != args.ddim:
            self.proj_decoder = torch.nn.Linear(args.adim, args.ddim)

        if args.mtlalpha < 1:
            self.decoder = Decoder(
                odim=odim,
                attention_dim=args.ddim,
                attention_heads=args.dheads,
                linear_units=args.dunits,
                num_blocks=args.dlayers,
                dropout_rate=args.dropout_rate,
                positional_dropout_rate=args.dropout_rate,
                self_attention_dropout_rate=args.transformer_attn_dropout_rate,
                src_attention_dropout_rate=args.transformer_attn_dropout_rate,
            )
        else:
            self.decoder = None
        self.blank = 0
        self.sos = odim - 1
        self.eos = odim - 1
        self.odim = odim
        self.ignore_id = ignore_id

        # self.lsm_weight = a
        self.criterion = LabelSmoothingLoss(
            self.odim,
            self.ignore_id,
            args.lsm_weight,
            args.transformer_length_normalized_loss,
        )

        self.adim = args.adim
        self.mtlalpha = args.mtlalpha
        if args.mtlalpha > 0.0:
            self.ctc = CTC(
                odim, args.adim, args.dropout_rate, ctc_type=args.ctc_type, reduce=True
            )
            # Intermediate CTC (Suggestion 3)
            # Use same config as main CTC
            self.ctc_inter = CTC(
                odim, args.adim, args.dropout_rate, ctc_type=args.ctc_type, reduce=True
            )
            self.interctc_weight = 0.0 # Disabled - was causing CTC to stagnate
        else:
            self.ctc = None
            self.ctc_inter = None

    def forward(self, x, lengths, label):
        if self.transformer_input_layer == "conv1d":
            lengths = torch.div(lengths, 640, rounding_mode="trunc")
        padding_mask = make_non_pad_mask(lengths).to(x.device).unsqueeze(-2)

        # Apply Dynamic Feature Extraction (only during inference)
        if self.use_dynamic_features and x.dim() == 5:
             x = x.permute(0, 2, 1, 3, 4)
             x = self.dynamic_feature_module(x)
             x = x.permute(0, 2, 1, 3, 4)
             
        # Concatenate C into H or W or flatten for linear input if needed
        # Standard ESPnet Conformer expects [B, T, D] or [B, T, C, H, W] for Conv2d/Conv3d layers
        if x.dim() == 5 and self.transformer_input_layer in ["conv2d", "conv3d", "vgg2l"]:
             # Keep as is, ESPnet encoder will handle [B, T, C, H, W]
             pass
        elif x.dim() == 5:
             # Flatten [B, T, C, H, W] to [B, T, C*H*W]
             b, t, c, h, w = x.shape
             x = x.view(b, t, c * h * w)

        # Enable intermediate output for Inter-CTC
        x, _, x_inter = self.encoder(x, padding_mask, return_intermediate=True)

        # ctc loss
        if self.ctc is not None:
            loss_ctc, ys_hat = self.ctc(x, lengths, label)
        else:
            # If mtlalpha==0 the ctc module is disabled — return zero loss and None hypothesis
            loss_ctc = torch.tensor(0.0, device=x.device)
            ys_hat = None
        
        # inter-ctc loss
        loss_ctc_inter = 0
        if self.ctc_inter is not None and x_inter is not None:
            loss_ctc_inter, _ = self.ctc_inter(x_inter, lengths, label)

        if self.proj_decoder:
            x_proj = self.proj_decoder(x)
        else:
            x_proj = x

        # decoder loss
        ys_in_pad, ys_out_pad = add_sos_eos(label, self.sos, self.eos, self.ignore_id)
        ys_mask = target_mask(ys_in_pad, self.ignore_id)
        pred_pad, _ = self.decoder(ys_in_pad, ys_mask, x_proj, padding_mask)
        loss_att = self.criterion(pred_pad, ys_out_pad)
        
        # Combine losses: MTL + InterCTC
        # Loss = alpha * CTC + (1-alpha) * Att + beta * InterCTC
        loss_main = self.mtlalpha * loss_ctc + (1 - self.mtlalpha) * loss_att
        loss = loss_main + (self.interctc_weight * loss_ctc_inter if hasattr(self, 'interctc_weight') else 0)

        acc = th_accuracy(
            pred_pad.view(-1, self.odim), ys_out_pad, ignore_label=self.ignore_id
        )

        # Return encoder output 'x' for Semantic Alignment (Idea 1)
        return loss, loss_ctc, loss_att, acc, x
