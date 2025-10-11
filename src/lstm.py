import torch.nn as nn


class LSTMFusion(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(LSTMFusion, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True,dropout=0.2)

    def forward(self, features):
        # [num_windows,bs, num_negative + 1,dims]->[bs,num_negative + 1,num_windows,dims]
        feature = features.permute(1, 2, 0, 3)
        feature_shape = feature.shape
        # [bs * (num_negative + 1),num_windows,dims]
        feature_new = feature.reshape(-1, feature_shape[-2], feature_shape[3])
        output, (h_n, c_n) = self.lstm(feature_new)
        output=output.reshape(feature_shape[0], feature_shape[1], feature_shape[2], -1)
        return output[:, :, -1, :]
