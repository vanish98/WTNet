import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import degree
from torch_scatter import scatter


class TimeEncode(torch.nn.Module):
    '''
    This class refer to the Bochner's time embedding
    time_dim: int, dimension of temporal entity embeddings
    relation_specific: bool, whether use relation specific freuency and phase. 是否使用特定于关系的频率和相
    num_relations: number of relations.
    '''

    def __init__(self, time_dim, relation_specific=False, num_relations=None):
        """
        :param time_dim: 64
        :param relation_specific: 是否使用特定于关系的频率和相位 True
        """
        super(TimeEncode, self).__init__()
        self.time_dim = time_dim
        self.relation_specific = relation_specific

        if relation_specific:  # shape: num_relations * time_dim
            self.basis_freq = torch.nn.Parameter(
                torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.time_dim)).float().unsqueeze(dim=0).repeat(
                    num_relations, 1))
            self.phase = torch.nn.Parameter(
                torch.zeros(self.time_dim).float().unsqueeze(dim=0).repeat(num_relations, 1))
        else:  # shape: time_dim
            self.basis_freq = torch.nn.Parameter(
                torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.time_dim)).float())
            self.phase = torch.nn.Parameter(torch.zeros(self.time_dim).float())

    def forward(self, ts, relations=None):
        '''
        :param ts: [edge_num, seq_len]
        :param relations: which relations do we extract their time embeddings.
        :return: [edge_num, seq_len, time_dim]
        '''
        edge_num = ts.size(0)
        seq_len = ts.size(1)  # seq_len = 1
        ts = torch.unsqueeze(ts, dim=2)

        if self.relation_specific:
            # self.basis_freq[relations]:  [edge_num, time_dim]
            map_ts = ts * self.basis_freq[relations].unsqueeze(dim=1)  # [edge_num, 1, time_dim]
            map_ts += self.phase[relations].unsqueeze(dim=1)
        else:
            # self.basis_freq:  [time_dim]
            map_ts = ts * self.basis_freq.view(1, 1, -1)  # [edge_num, 1, time_dim]
            map_ts += self.phase.view(1, 1, -1)
        harmonic = torch.cos(map_ts)
        return harmonic


class TemporalPathAgg(MessagePassing):

    def __init__(self, input_dim, output_dim, num_relation, query_input_dim,
                 layer_norm=False, activation="relu", time_encoding=True,
                 time_encoding_independent=True):
        """
        :param input_dim: 64
        :param output_dim: 64
        :param query_input_dim: 64
        """
        super(TemporalPathAgg, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relation = num_relation
        self.query_input_dim = query_input_dim
        self.time_encoding = time_encoding
        self.time_encoding_independent = time_encoding_independent

        if time_encoding:
            self.time_encoder = TimeEncode(time_dim=self.input_dim, relation_specific=time_encoding_independent,
                                           num_relations=num_relation)
            self.relation4time = nn.Sequential(
                nn.Linear(input_dim * 2, input_dim),
                nn.ReLU()
            )

        if layer_norm:
            self.layer_norm = nn.LayerNorm(output_dim)
        else:
            self.layer_norm = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation


        self.linear = nn.Linear(input_dim * 13, output_dim)


        self.relation_linear = nn.Linear(query_input_dim, num_relation * input_dim)

    def forward(self, input, query, initial_stat, edge_index, edge_type, edge_time, size, all_graph_min_times,
                edge_weight=None):
        """
        :param input last  result
        :param query:query embedding
        :param iniital_stat when 1th, input=initial_stat
        :param size graph size
        :param all_graph_min_times  1th window graph start time
        """
        batch_size = len(query)
        # layer-specific relation features as a projection of query r embeddings Qr
        relation = self.relation_linear(query).view(batch_size, self.num_relation, self.input_dim)
        if edge_weight is None:  # Wr
            edge_weight = torch.ones(len(edge_type), device=input.device)
        output = self.propagate(input=input, relation=relation, initial_stat=initial_stat, edge_index=edge_index,
                                edge_type=edge_type, edge_time=edge_time, min_times=all_graph_min_times, size=size,
                                edge_weight=edge_weight)
        return output

    def propagate(self, edge_index, size=None, **kwargs):
        return super(TemporalPathAgg, self).propagate(edge_index, size, **kwargs)

    def message(self, input_j, relation, initial_stat, edge_type, edge_time, min_times):
        '''

        Parameters
        ----------
        input_j 领域节点的特征
        relation
        initial_stat
        edge_type
        edge_time
        min_times

        Returns
        -------

        '''
        # gain current batch Qr
        relation_emb = relation.index_select(self.node_dim, edge_type)

        # time encoding
        if self.time_encoding:
            # Time embedding
            if self.time_encoding_independent:
                time_emb = self.time_encoder((edge_time - min_times).unsqueeze(1), edge_type)
            else:
                time_emb = self.time_encoder((edge_time - min_times).unsqueeze(1))
            time_emb = torch.squeeze(time_emb, 1)
            # query-aware temporal representation
            relation_j = self.relation4time(
                torch.cat([relation_emb, time_emb.repeat(relation_emb.shape[0], 1, 1)], dim=-1))
        else:
            relation_j = relation_emb

        # distmult
        message = input_j * relation_j

        # augment messages with the initial_stat
        message = torch.cat([message, initial_stat],
                            dim=self.node_dim)  # (batch_size, num_edges + num_nodes, input_dim)

        return message  # 17263

    def aggregate(self, input, edge_weight, index, dim_size):
        index = torch.cat([index, torch.arange(dim_size, device=input.device)])
        edge_weight = torch.cat([edge_weight, torch.ones(dim_size, device=input.device)])
        shape = [1] * input.ndim
        shape[self.node_dim] = -1  # node_dim 节点数量所在的维度
        edge_weight = edge_weight.view(shape)

        eps = 1e-6
        # scatter根据相同的索引值按reduce的方式计算
        mean = scatter(input * edge_weight, index, dim=self.node_dim, dim_size=dim_size, reduce="mean")
        sq_mean = scatter(input ** 2 * edge_weight, index, dim=self.node_dim, dim_size=dim_size, reduce="mean")
        max = scatter(input * edge_weight, index, dim=self.node_dim, dim_size=dim_size, reduce="max")
        min = scatter(input * edge_weight, index, dim=self.node_dim, dim_size=dim_size, reduce="min")
        std = (sq_mean - mean ** 2).clamp(min=eps).sqrt()
        features = torch.cat([mean.unsqueeze(-1), max.unsqueeze(-1), min.unsqueeze(-1), std.unsqueeze(-1)], dim=-1)
        features = features.flatten(-2)
        degree_out = degree(index, dim_size).unsqueeze(0).unsqueeze(-1)
        scale = degree_out.log()
        scale = scale / scale.mean()
        scales = torch.cat([torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=-1)
        output = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2)
        return output


    def update(self, update, input):
        output = self.linear(torch.cat([input, update], dim=-1))
        if self.layer_norm:
            output = self.layer_norm(output)
        if self.activation:
            output = self.activation(output)
        return output
