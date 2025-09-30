import torch
import torch.nn as nn
import math

class LearningPositionalEncoding(nn.Module):
    def __init__(self, feature_dim: int, num_windows: int = 20):
        super(LearningPositionalEncoding, self).__init__()
        self.pe = nn.Embedding(num_windows, feature_dim)

    def forward(self, x):
        feature_shape = x.shape
        positions_index = torch.arange(0, feature_shape[-2], dtype=torch.long, device=x.device).unsqueeze(0)
        position_embedding = self.pe(positions_index)
        return position_embedding + x

class RotaryPositionalEncoding(nn.Module):
    def __init__(self, dim: int, num_windows: int = 10):

        super(RotaryPositionalEncoding, self).__init__()
        assert dim % 2 == 0, " The dimension must be even"


        frequencies = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(num_windows, dtype=torch.float32)
        freqs = torch.outer(t, frequencies)  # [max_seq_len, dim // 2]
        emb = torch.cat((freqs.sin(), freqs.cos()), dim=-1)  # [max_seq_len, dim]
        self.register_buffer('emb', emb)

    def forward(self, x: torch.Tensor):

        bs, seq_len, dim = x.shape
        # Obtain the sin/cos encoding for the corresponding position [1, seq_1en, dim]
        sin_cos = self.emb[:seq_len].unsqueeze(0)

        # Split into sin and cos parts [1, seq_1en, dim/2]
        sin, cos = sin_cos.chunk(2, dim=-1)

        # Split the input into the first half and the second half
        x1, x2 = x.chunk(2, dim=-1)

        # Apply rotation formula
        rotated_x = torch.cat(
            [x1 * cos - x2 * sin,
             x1 * sin + x2 * cos],
            dim=-1
        )

        return rotated_x

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, feature_dim: int, num_windows: int = 20):
        super(SinusoidalPositionalEncoding, self).__init__()

        pe = torch.zeros(num_windows, feature_dim)
        position = torch.arange(0, num_windows, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, feature_dim, 2).float() * (-math.log(10000.0) / feature_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)


        self.register_buffer('pe', pe.unsqueeze(0))  # [1, num_windows, feature_dim]

    def forward(self, x):
        """
        x: Features ，shape： [B, num_windows, feature_dim]

        """
        batch_size, seq_len = x.size(0), x.size(1)
        return x + self.pe[:, :seq_len, :]


class FeatureFusionTransformer(nn.Module):
    def __init__(self,num_windows, feature_dim=64, num_heads=8, num_layers=4, num_hiddens=128, dropout=0.1,pe="learning"):
        super(FeatureFusionTransformer, self).__init__()
        self.num_windows = num_windows

        encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=num_heads,
                                                   dim_feedforward=num_hiddens, dropout=dropout, batch_first=True)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        if pe == "learning":
            self.position_embedding = nn.Embedding(num_windows, feature_dim)
        elif pe == "sin":
            self.position_embedding = SinusoidalPositionalEncoding(feature_dim, num_windows=num_windows)
        elif pe == "rotary":
            self.position_embedding = RotaryPositionalEncoding(feature_dim, num_windows=num_windows)

    def forward(self, feature):
        """
        Parameters
        ----------
        feature  dims[num_windows,bs, num_negative + 1,dims]

        Returns
        -------
        output dims[bs, num_negative + 1,dims]
        """
        # [num_windows,bs, num_negative + 1,dims]->[bs,num_negative + 1,num_windows,dims]
        feature = feature.permute(1, 2, 0, 3)
        feature_shape = feature.shape

        # when valid and test model , the third dims values equal node number  ,is too large
        negative_nums = feature_shape[1]
        final_output = list()
        # data too large ,split to valid
        for i in range((negative_nums - 1) // 65 + 1):
            max_idx = (i + 1) * 65
            if max_idx > negative_nums:
                spilt = feature[:, i * 65:, :, :]
            else:
                spilt = feature[:, i * 65:(i + 1) * 65, :, :]
            spilt_shape = spilt.shape
            # [bs,num_negative + 1,num_windows,dims]->[bs*num_negative + 1,num_windows ,dims]
            feature_new = spilt.reshape(-1, spilt_shape[-2], spilt_shape[3])
            # generate position index
            positions_index = torch.arange(0, feature_shape[-2], dtype=torch.long, device=feature.device).unsqueeze(0)
            # gain position embedding
            position_embedding = self.position_embedding(positions_index)

            feature_new+= position_embedding
            output = self.transformer_encoder(feature_new)
            #recovery origin shape [bs*num_negative + 1,num_windows ,dims]->[bs,num_negative + 1,num_windows,dims]
            output = output.reshape(spilt_shape[0], spilt_shape[1], spilt_shape[2], -1)
            final_output.append(output)

        final_feature = torch.cat(final_output, dim=1)

        return final_feature[:, :, -1, :]
