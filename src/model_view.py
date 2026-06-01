import torch
from torch import nn, autograd
from torch.nn import functional as F
from torch_scatter import scatter_add

import layers
from transformer import FeatureFusionTransformer


class WTNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_nodes, num_relation, history_len, windows_size, num_mlp_layer,
                 num_heads, num_fusion_layers, num_transformer_hiddens, dropout, num_windows,
                 message_func="distmult", aggregate_func="pna", short_cut=False, layer_norm=False, activation="relu"
                 , time_encoding=True, time_encoding_independent=True, values_way='mean'):
        """
        :param input_dim: 64
        :param hidden_dims:[64, 64, 64, 64]
        :param short_cut :True
        :param layer_norm: True
        :param time_encoding: True
        :param time_encoding_independent: True

        """
        super(WTNet, self).__init__()
        # [64，64, 64, 64, 64]
        self.dims = [input_dim] + list(hidden_dims)
        self.num_nodes = num_nodes
        self.num_relation = num_relation * 2  # reverse rel type should be added
        # default value ：True
        self.short_cut = short_cut  # whether to use residual connections between layers
        self.windows_size = windows_size
        self.history_len = history_len
        self.num_windows = num_windows
        # 64+64=128
        self.feature_dim = hidden_dims[-1] + input_dim
        # Learnable Relation Representation [batch_size,input_dim]
        self.query = nn.Embedding(self.num_relation, input_dim)
        self.values_way = values_way
        self.fusion_layer = FeatureFusionTransformer(self.num_windows, feature_dim=input_dim, num_heads=num_heads,
                                                     num_layers=num_fusion_layers, num_hiddens=num_transformer_hiddens,
                                                     dropout=dropout)
        self.layers = nn.ModuleList()
        # 4 layers
        for i in range(len(self.dims) - 1):  # num of hidden layers
            self.layers.append(layers.TemporalPathAgg(self.dims[i], self.dims[i + 1], self.num_relation,
                                                      self.dims[0], message_func, aggregate_func, layer_norm,
                                                      activation, time_encoding, time_encoding_independent))

        mlp = []
        for i in range(num_mlp_layer - 1):
            mlp.append(nn.Linear(self.feature_dim, self.feature_dim))
            mlp.append(nn.ReLU())
        mlp.append(nn.Linear(self.feature_dim, 1))
        self.mlp = nn.Sequential(*mlp)

    def negative_sample_to_tail(self, h_index, t_index, r_index):
        # convert p(h | t, r) to p(t' | h', r')
        # h' = t, r' = r^{-1}, t' = h
        is_t_neg = (h_index == h_index[:, [0]]).all(dim=-1, keepdim=True)
        new_h_index = torch.where(is_t_neg, h_index, t_index)
        new_t_index = torch.where(is_t_neg, t_index, h_index)
        new_r_index = torch.where(is_t_neg, r_index, r_index + self.num_relation // 2)
        return new_h_index, new_t_index, new_r_index

    # Query-aware Temporal Path Processing
    def pathProc(self, history_graph, h_index, query, all_graph_min_times, initial_stat, separate_grad=False):
        """
        :param history_graph:
        :param h_index: [16,]
        :param query: query_embedding [batch_size, 64]
        :param separate_grad:
        :return:
        """

        # batch_size = len(h_index)
        # # [batch_size, 64]
        # index = h_index.unsqueeze(-1).expand_as(query)
        #
        # # initialize all pairs states as zeros in memory
        # initial_stat = torch.zeros(batch_size, history_graph.num_nodes(), self.dims[0], device=h_index.device)
        #
        # # Temporal Path Initialization
        # initial_stat.scatter_add_(1, index.unsqueeze(1), query.unsqueeze(1))

        size = (history_graph.num_nodes(), history_graph.num_nodes())
        edge_weight = torch.ones(history_graph.num_edges(), device=h_index.device)

        edge_weights = []
        layer_input = initial_stat
        # TemporalPathAgg 4 layers
        hidden = ""
        for layer in self.layers:
            if separate_grad:
                edge_weight = edge_weight.clone().requires_grad_()
            # layers iteration
            hidden = layer(layer_input, query, initial_stat,
                           torch.stack(history_graph.edges()), history_graph.edata['type'], history_graph.edata['time'],
                           size, all_graph_min_times, edge_weight=edge_weight)
            # residual connections
            if self.short_cut and hidden.shape == layer_input.shape:
                # shortcut setting
                hidden = hidden + layer_input
            edge_weights.append(edge_weight)
            layer_input = hidden

        return {
            "node_feature": hidden,
            "edge_weights": edge_weights,
        }

    def forward(self, history_graph_list, query_triple):
        h_index, r_index, t_index = query_triple.unbind(-1)
        shape = h_index.shape
        batch_size = shape[0]

        # turn all triples in a batch into a tail prediction mode
        h_index, t_index, r_index = self.negative_sample_to_tail(h_index, t_index, r_index)
        # verify every element in each row equals the first element of that row
        assert (h_index[:, [0]] == h_index).all()
        assert (r_index[:, [0]] == r_index).all()

        # initialize queries (relation types of the given triples) embeddings[batch_size,64]
        # query_embedding
        query = self.query(r_index[:, 0])
        # original query (relation type) embeddings
        # node_query = query.unsqueeze(1).expand(-1, history_graph_list[-1].num_nodes(), -1)

        index = h_index[:, 0].unsqueeze(-1).expand_as(query)
        # initialize all pairs states as zeros in memory
        initial_stat = torch.zeros(batch_size, history_graph_list[-1].num_nodes(), self.dims[0], device=h_index.device)

        # Temporal Path Initialization
        initial_stat.scatter_add_(1, index.unsqueeze(1), query.unsqueeze(1))

        feature_t_list = list()
        all_graph_min_times = history_graph_list[0].edata['time'].min()
        for ind, history_graph in enumerate(history_graph_list):
            # Query-aware Temporal Path Processing
            # output{node_feature,edge_weights}
            output = self.pathProc(history_graph, h_index[:, 0], query, all_graph_min_times, initial_stat)
            feature = output["node_feature"]
            # feature = torch.cat([feature, node_query], dim=-1)
            index = t_index.unsqueeze(-1).expand(-1, -1, feature.shape[-1])
            feature_t = feature.gather(1, index)
            feature_t_list.append(feature_t)

        # [history-1/windows_size+1,bs,node_nums,dims]
        feature_ts = torch.stack(feature_t_list)

        # [num_windows,bs, num_negative + 1,dims]->[bs, num_negative + 1,dims]
        fusion_feature = self.fusion_layer(feature_ts)

        query = query.unsqueeze(1).expand(-1, fusion_feature.shape[1], -1)

        fusion_feature = torch.cat([fusion_feature, query], dim=-1)
        # (batch_size, num_negative + 1, dim) -> (batch_size, num_negative + 1)
        score = self.mlp(fusion_feature).squeeze(-1)

        return score.view(shape)

    def get_loss(self, args, pred):

        target = torch.zeros_like(pred)
        target[:, 0] = 1
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        neg_weight = torch.ones_like(pred)
        if args.adversarial_temperature > 0:
            with torch.no_grad():
                neg_weight[:, 1:] = F.softmax(pred[:, 1:] / args.adversarial_temperature, dim=-1)
        else:
            neg_weight[:, 1:] = 1 / args.negative_num
        loss = (loss * neg_weight).sum(dim=-1) / neg_weight.sum(dim=-1)
        loss = loss.mean()

        tmp = torch.mm(self.query.weight, self.query.weight.permute(1, 0))
        orthogonal_regularizer = torch.norm(tmp - 1 * torch.diag(torch.ones(self.num_relation, device=pred.device)), 2)

        loss = loss + orthogonal_regularizer
        return loss

    def visualize(self, history_graph_list, batch, num_beam=10, k_per_window=5, max_total_paths=20):
        """
        Window-aware path visualization for TiPNN.

        Parameters
        ----------
        history_graph_list: list[DGLGraph]
            The same window graph list used by forward().
            Each graph corresponds to one historical window.
        batch: torch.LongTensor, shape = [1, 3]
            One query triple: [h, r, t].
            For head prediction explanation, pass inverse triple [t, r + num_rel, h].
        num_beam: int
            Beam size for path search.
        k_per_window: int
            Number of paths retained from each window.
        max_total_paths: int
            Maximum number of paths printed after merging all windows.

        Returns
        -------
        results: list[dict]
            Each dict contains:
            {
                "window_id": int,
                "window_time_min": int,
                "window_time_max": int,
                "path": list[(h, t, r, ts)],
                "weight": float
            }
        """
        assert batch.shape == (1, 3)

        h_index, r_index, t_index = batch.unbind(-1)
        device = h_index.device

        # Query relation embedding: [1, input_dim]
        query = self.query(r_index)

        batch_size = len(h_index)
        num_nodes = history_graph_list[-1].num_nodes()

        # Same initialization logic as forward()
        index = h_index.unsqueeze(-1).expand_as(query)
        initial_stat = torch.zeros(
            batch_size,
            num_nodes,
            self.dims[0],
            device=device
        )
        initial_stat.scatter_add_(1, index.unsqueeze(1), query.unsqueeze(1))

        # Same global minimum time logic as forward()
        valid_min_times = []
        for g in history_graph_list:
            if g.num_edges() > 0 and "time" in g.edata:
                valid_min_times.append(g.edata["time"].min())

        if len(valid_min_times) > 0:
            all_graph_min_times = torch.stack(valid_min_times).min()
        else:
            all_graph_min_times = torch.tensor(0, device=device)

        feature_t_list = []
        window_edge_weights = []

        # 1. Run pathProc for every window graph, exactly like forward()
        for history_graph in history_graph_list:
            output = self.pathProc(
                history_graph,
                h_index,
                query,
                all_graph_min_times,
                initial_stat,
                separate_grad=True
            )

            feature = output["node_feature"]
            edge_weights = output["edge_weights"]

            # Gather target entity feature in this window
            target_index = t_index.view(batch_size, 1, 1).expand(
                -1,
                1,
                feature.shape[-1]
            )
            feature_t = feature.gather(1, target_index)

            feature_t_list.append(feature_t)
            window_edge_weights.append(edge_weights)

        # 2. Fuse all window-level target features, same as forward()
        feature_ts = torch.stack(feature_t_list)
        fusion_feature = self.fusion_layer(feature_ts)

        query_expand = query.unsqueeze(1).expand(
            -1,
            fusion_feature.shape[1],
            -1
        )

        fusion_feature = torch.cat([fusion_feature, query_expand], dim=-1)

        # Scalar score for this single query
        score = self.mlp(fusion_feature).squeeze(-1).sum()

        # 3. Compute gradients from final score back to edges in every window
        flat_edge_weights = []
        flat_window_ids = []

        for w_id, edge_weights in enumerate(window_edge_weights):
            for ew in edge_weights:
                flat_edge_weights.append(ew)
                flat_window_ids.append(w_id)

        edge_grads_flat = autograd.grad(
            score,
            flat_edge_weights,
            retain_graph=False,
            allow_unused=True
        )

        # Replace unused gradients with zeros
        edge_grads_flat = [
            torch.zeros_like(ew) if grad is None else grad
            for grad, ew in zip(edge_grads_flat, flat_edge_weights)
        ]

        # Regroup gradients by window
        edge_grads_by_window = [[] for _ in range(len(history_graph_list))]

        for grad, w_id in zip(edge_grads_flat, flat_window_ids):
            edge_grads_by_window[w_id].append(grad)

        # 4. Run beam-search path extraction inside each window
        all_results = []

        for w_id, history_graph in enumerate(history_graph_list):
            if history_graph.num_edges() == 0:
                continue

            graph_simple_data = {
                "num_nodes": history_graph.num_nodes(),
                "edge_index": torch.stack(history_graph.edges()),
                "edge_type": history_graph.edata["type"],
                "edge_time": history_graph.edata["time"]
            }

            distances, back_edges = self.beam_search_distance(
                graph_simple_data,
                edge_grads_by_window[w_id],
                h_index,
                t_index,
                num_beam=num_beam
            )

            paths, weights = self.topk_average_length(
                distances,
                back_edges,
                t_index,
                k=k_per_window
            )

            if len(paths) == 0:
                continue

            if "time" in history_graph.edata and history_graph.num_edges() > 0:
                window_time_min = int(history_graph.edata["time"].min().item())
                window_time_max = int(history_graph.edata["time"].max().item())
            else:
                window_time_min = -1
                window_time_max = -1

            for path, weight in zip(paths, weights):
                all_results.append({
                    "window_id": w_id,
                    "window_time_min": window_time_min,
                    "window_time_max": window_time_max,
                    "path": path,
                    "weight": float(weight)
                })

        all_results = sorted(
            all_results,
            key=lambda x: x["weight"],
            reverse=True
        )[:max_total_paths]

        return all_results

    @torch.no_grad()
    def beam_search_distance(self, data, edge_grads, h_index, t_index, num_beam=10):
        num_nodes = data['num_nodes']
        input = torch.full((num_nodes, num_beam), float("-inf"), device=h_index.device)
        input[h_index, 0] = 0
        edge_mask = data['edge_index'][0, :] != t_index

        distances = []
        back_edges = []

        for edge_grad in edge_grads:
            node_in, node_out = data['edge_index'][:, edge_mask]
            relation = data['edge_type'][edge_mask]
            rel_ts = data['edge_time'][edge_mask]
            edge_grad = edge_grad[edge_mask]
            message = input[node_in] + edge_grad.unsqueeze(-1)  # (num_edges, num_beam)

            msg_source = torch.stack([node_in, node_out, relation, rel_ts], dim=-1).unsqueeze(1).expand(-1, num_beam,
                                                                                                        -1)
            is_duplicate = torch.isclose(message.unsqueeze(-1), message.unsqueeze(-2)) & \
                           (msg_source.unsqueeze(-2) == msg_source.unsqueeze(-3)).all(dim=-1)
            is_duplicate = is_duplicate.float() - \
                           torch.arange(num_beam, dtype=torch.float, device=message.device) / (num_beam + 1)

            prev_rank = is_duplicate.argmax(dim=-1, keepdim=True)
            msg_source = torch.cat([msg_source, prev_rank], dim=-1)  # (num_edges, num_beam, 4)

            node_out, order = node_out.sort()
            node_out_set = torch.unique(node_out)
            message = message[order].flatten()  # (num_edges * num_beam)
            msg_source = msg_source[order].flatten(0, -2)  # (num_edges * num_beam, 4)
            size = node_out.bincount(minlength=num_nodes)
            msg2out = size_to_index(size[node_out_set] * num_beam)
            is_duplicate = (msg_source[1:] == msg_source[:-1]).all(dim=-1)
            is_duplicate = torch.cat([torch.zeros(1, dtype=torch.bool, device=message.device), is_duplicate])
            message = message[~is_duplicate]
            msg_source = msg_source[~is_duplicate]
            msg2out = msg2out[~is_duplicate]
            size = msg2out.bincount(minlength=len(node_out_set))

            if not torch.isinf(message).all():
                distance, rel_index = scatter_topk(message, size, k=num_beam)
                abs_index = rel_index + (size.cumsum(0) - size).unsqueeze(-1)
                back_edge = msg_source[abs_index]
                distance = distance.view(len(node_out_set), num_beam)
                back_edge = back_edge.view(len(node_out_set), num_beam, 5)
                distance = scatter_add(distance, node_out_set, dim=0, dim_size=num_nodes)
                back_edge = scatter_add(back_edge, node_out_set, dim=0, dim_size=num_nodes)
            else:
                distance = torch.full((num_nodes, num_beam), float("-inf"), device=message.device)
                back_edge = torch.zeros(num_nodes, num_beam, 5, dtype=torch.long, device=message.device)

            distances.append(distance)
            back_edges.append(back_edge)
            input = distance

        return distances, back_edges

    def topk_average_length(self, distances, back_edges, t_index, k=10):
        paths = []
        average_lengths = []

        for i in range(len(distances)):
            distance, order = distances[i][t_index].flatten(0, -1).sort(descending=True)
            back_edge = back_edges[i][t_index].flatten(0, -2)[order]
            for d, (h, t, r, ts, prev_rank) in zip(distance[:k].tolist(), back_edge[:k].tolist()):
                if d == float("-inf"):
                    break
                path = [(h, t, r, ts)]
                for j in range(i - 1, -1, -1):
                    h, t, r, ts, prev_rank = back_edges[j][h, prev_rank].tolist()
                    path.append((h, t, r, ts))
                paths.append(path[::-1])
                average_lengths.append(d / len(path))

        if paths:
            average_lengths, paths = zip(*sorted(zip(average_lengths, paths), reverse=True)[:k])

        return paths, average_lengths


def size_to_index(size):
    range = torch.arange(len(size), device=size.device)
    index2sample = range.repeat_interleave(size)
    return index2sample


def multi_slice_mask(starts, ends, length):
    values = torch.cat([torch.ones_like(starts), -torch.ones_like(ends)])
    slices = torch.cat([starts, ends])
    mask = scatter_add(values, slices, dim=0, dim_size=length + 1)[:-1]
    mask = mask.cumsum(0).bool()
    return mask


def scatter_extend(data, size, input, input_size):
    new_size = size + input_size
    new_cum_size = new_size.cumsum(0)
    new_data = torch.zeros(new_cum_size[-1], *data.shape[1:], dtype=data.dtype, device=data.device)
    starts = new_cum_size - new_size
    ends = starts + size
    index = multi_slice_mask(starts, ends, new_cum_size[-1])
    new_data[index] = data
    new_data[~index] = input
    return new_data, new_size


def scatter_topk(input, size, k, largest=True):
    index2graph = size_to_index(size)
    index2graph = index2graph.view([-1] + [1] * (input.ndim - 1))

    mask = ~torch.isinf(input)
    max = input[mask].max().item()
    min = input[mask].min().item()
    safe_input = input.clamp(2 * min - max, 2 * max - min)
    offset = (max - min) * 4
    if largest:
        offset = -offset
    input_ext = safe_input + offset * index2graph
    index_ext = input_ext.argsort(dim=0, descending=largest)
    num_actual = size.clamp(max=k)
    num_padding = k - num_actual
    starts = size.cumsum(0) - size
    ends = starts + num_actual
    mask = multi_slice_mask(starts, ends, len(index_ext)).nonzero().flatten()

    if (num_padding > 0).any():
        padding = ends - 1
        padding2graph = size_to_index(num_padding)
        mask = scatter_extend(mask, num_actual, padding[padding2graph], num_padding)[0]

    index = index_ext[mask]  # (N * k, ...)
    value = input.gather(0, index)
    if isinstance(k, torch.Tensor) and k.shape == size.shape:
        value = value.view(-1, *input.shape[1:])
        index = index.view(-1, *input.shape[1:])
        index = index - (size.cumsum(0) - size).repeat_interleave(k).view([-1] + [1] * (index.ndim - 1))
    else:
        value = value.view(-1, k, *input.shape[1:])
        index = index.view(-1, k, *input.shape[1:])
        index = index - (size.cumsum(0) - size).view([-1] + [1] * (index.ndim - 1))

    return value, index
