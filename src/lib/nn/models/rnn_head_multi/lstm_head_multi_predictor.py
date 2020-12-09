from typing import Optional

import numpy as np
import torch
from torch import nn, Tensor
import timm


class LSTMHeadMultiPredictor(nn.Module):

    def __init__(
        self,
        backbone: str,
        in_channels: int,
        encoder_hidden_dim: int = 128,
        decoder_hidden_dim: int = 128,
        num_modes: int = 3,
        future_len: int = 50,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.encoder_hidden_dim = encoder_hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.num_modes = num_modes
        self.future_len = future_len

        self.backbone = timm.create_model(
            model_name=backbone,
            pretrained=True,
            num_classes=0,
            in_chans=in_channels
        )
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.feature_dim = self.backbone.num_features + encoder_hidden_dim
        self.feat_to_confidence = nn.Linear(self.feature_dim, num_modes)
        self.feat_to_dec_hidden = nn.Linear(self.feature_dim, num_modes * decoder_hidden_dim)
        self.encoder = nn.LSTM(input_size=3, hidden_size=encoder_hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(input_size=1, hidden_size=decoder_hidden_dim, batch_first=True)
        self.dec_hidden_to_target = nn.Linear(decoder_hidden_dim, 2)

    def forward(self, image: Tensor, history_positions: Tensor, history_availabilities: Tensor):
        batch_size = image.shape[0]

        feat = self.backbone.forward_features(image)
        feat = self.avg_pool(feat).reshape(batch_size, -1)

        enc_input = torch.cat([history_positions, history_availabilities[..., np.newaxis]], dim=-1)
        enc_input = torch.flip(enc_input, dims=(1,))
        _, (enc_hidden, _) = self.encoder(enc_input)
        assert enc_hidden.shape == (1, batch_size, self.encoder_hidden_dim)
        feat = torch.cat([feat, enc_hidden[0]], dim=1)

        confidence = torch.softmax(self.feat_to_confidence(feat), dim=1)
        dec_hidden = self.feat_to_dec_hidden(feat)
        dec_hidden = dec_hidden.reshape(1, batch_size * self.num_modes, self.decoder_hidden_dim)

        dec_input = torch.linspace(0, 1, self.future_len, device=dec_hidden.device)
        dec_input = dec_input.reshape(1, self.future_len, 1).expand(batch_size * self.num_modes, -1, -1)
        dec_output, _ = self.decoder(dec_input, (dec_hidden, torch.zeros_like(dec_hidden)))
        dec_output = dec_output.reshape(batch_size * self.num_modes * self.future_len, self.decoder_hidden_dim)
        pred = self.dec_hidden_to_target(dec_output)
        pred = pred.reshape(batch_size, self.num_modes, self.future_len, 2)

        return pred, confidence
