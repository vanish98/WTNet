import os.path
import sys

import numpy as np
import torch

sys.path.append("..")
import utils
import json
from config import args
from model import WTNet
from datetime import datetime
from torch.utils import data as torch_data
from torch import distributed as dist


def split_subgraph(history_list, num_nodes, num_rels, windows_size):
    '''

    Parameters
    ----------
    history_list: all history subgraph
    num_nodes:
    num_rels:
    windows_siz e

    Returns
    -------

    '''
    history_graph_list = []
    history_list_len = len(history_list)
    # history_list length would < history_len
    # 第一个窗口的长度可以小于windows_size， if hl=14 ,windows_size=5,windows hl is {4,5,5}
    num,first_length = divmod(history_list_len, windows_size)
    start = 0
    for i in range(num+(1 if first_length else 0)):
        end = start + (first_length if i == 0 and first_length else windows_size)
        sub_history_list = np.concatenate(history_list[start:end])
        history_graph_list.append(utils.build_history_graph(num_nodes, num_rels, sub_history_list, device))
        start = end
    return history_graph_list


@torch.no_grad()
def inductive(model, test_list, num_rels, num_nodes, mode="train", model_name = None):

    world_size = utils.get_world_size()
    rank = utils.get_rank()

    if mode == "test":
        # test mode: load parameter form file
        model_name=os.path.join(f"../{args.pretrain_name}",model_name)
        checkpoint = torch.load(model_name, map_location=device)
        if utils.get_rank() == 0:
            print("\nLoad Model name: {}. Using best epoch : {}. \n\nargs:{}.".format(model_name, checkpoint['epoch'], checkpoint['args']))  # use best stat checkpoint
            print("\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\nstart test\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(device)

    idx = [_ for _ in range(len(test_list))] # timestamps index [0,1,2,3,...,n]

    model.eval()
    rankings = []

    for future_sample_id in idx:
        if future_sample_id < args.history_len: continue
        # future_sample as the future graph index
        future_list = test_list[future_sample_id][:,:3]
        # get history graph list
        history_list = test_list[future_sample_id - args.history_len : future_sample_id]
    
        # Generate graph
        # history_g_list = [utils.build_graph(num_nodes, num_rels, snap, device) for snap in history_list]

        # history_list combine
        windows_size = args.windows_size
        history_graph_list = split_subgraph(history_list, num_nodes, num_rels, windows_size)
        future_triple = torch.from_numpy(future_list).long().to(device)

        time_filter_data = {
                'num_nodes': num_nodes,
                'edge_index': torch.stack([future_triple[:,0], future_triple[:,2]]),
                'edge_type': future_triple[:,1]
        }
        sampler = torch_data.DistributedSampler(future_triple, world_size, rank)
        future_loader = torch_data.DataLoader(future_triple, args.batch_size, sampler=sampler, num_workers=args.n_worker)
        
        for batch in future_loader:
            t_batch, h_batch = utils.all_negative(num_nodes, batch)
            t_pred = model(history_graph_list, t_batch)
            h_pred = model(history_graph_list, h_batch)

            pos_h_index, pos_r_index, pos_t_index = batch.t()

            # time_filter Rank
            timef_t_mask, timef_h_mask = utils.strict_negative_mask(time_filter_data, batch[: , [0,2,1]])
            timef_t_ranking = utils.compute_ranking(t_pred, pos_t_index, timef_t_mask)
            timef_h_ranking = utils.compute_ranking(h_pred, pos_h_index, timef_h_mask)
            rankings += [timef_t_ranking, timef_h_ranking]
            utils.synchronize()
    utils.synchronize()

        # This is the end of prediction at 'future_sample_id' time
    # This is the end of prediction at test_set

    ranking = torch.cat(rankings)

    all_size = torch.zeros(world_size, dtype=torch.long, device=device)
    all_size[rank] = len(ranking)
    if world_size > 1:
        dist.all_reduce(all_size, op=dist.ReduceOp.SUM)
    cum_size = all_size.cumsum(0)

    all_ranking = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
    all_ranking[cum_size[rank] - all_size[rank]: cum_size[rank]] = ranking
    
    if world_size > 1:
        dist.all_reduce(all_ranking, op=dist.ReduceOp.SUM)

    if rank == 0:
        metrics_dict = dict()
        for metric in args.metric:
            if metric == "mr":
                score = all_ranking.float().mean()
            elif metric == "mrr":
                score = (1 / all_ranking.float()).mean()
            elif metric.startswith("hits@"):
                values = metric[5:].split("_")
                threshold = int(values[0])
                score = (all_ranking <= threshold).float().mean()
            metrics_dict[metric] = score.item()
        metrics_dict['time'] = datetime.strftime(datetime.now(),'%Y-%m-%d %H:%M:%S')
        json_dumps = json.dumps(metrics_dict, indent=4)
        print(json_dumps)
        result_path =  model_name.replace("../","../inductive/") + ".txt"
        if os.path.exists(result_path):
            os.mkdir(result_path)
        with open(result_path,'a+',encoding='utf-8') as f:
            f.write(f"{args.pretrain_name}->{args.dataset}")
            f.write(json_dumps)
            f.close()
        
    mrr = (1 / all_ranking.float()).mean()

    return mrr


if __name__ == '__main__':

    utils.set_rand_seed(2023)
    working_dir = utils.create_working_directory(args)
    windows_size = args.windows_size
    # bsize:16-neg:64-hislen:8-msg:distmult-aggr:pna-dim:64+[64, 64, 64, 64]|True|True|True|True
    model_name = (f"bsize={args.batch_size}-neg={args.negative_num}-hislen={args.history_len}"
                  f"-msg={args.message_func}-aggr={args.aggregate_func}-heads={args.num_heads}"
                  f"-tlayer={args.num_transformer_layers}-thidden={args.num_transformer_hiddens}"
                  f"-values_way={args.values_way}-dim={args.input_dim}+{args.hidden_dims}-ws={windows_size}"
                  f"_{args.short_cut}_{args.layer_norm}"
                  f"_{args.time_encoding}_{args.time_encoding_independent}")

    model_state_file = model_name + "" + args.parameter_id
    # load datasets
    data = utils.load_data(args.dataset)

    if utils.get_rank() == 0:
        print("# Sanity Check: Inductive MODE, no check setting")

    # change the view of the data
    # [[s,r,o,t],[s,r,o,t],[s,r,o,t],...] -->> [ [ [s,r,o,t],[s,r,o,t] ], [ [s,r,o,t] ],...]
    train_list_sp = utils.split_by_time(data.train, stat_show=False)
    valid_list_sp = utils.split_by_time(data.valid, stat_show=False)
    test_list_sp = utils.split_by_time(data.test, stat_show=False)

    all_list = train_list_sp + valid_list_sp + test_list_sp
    train_list = train_list_sp
    valid_list = train_list[-args.history_len:] + valid_list_sp
    test_list = valid_list[-args.history_len:] + test_list_sp
    history_len = args.history_len
    num_nodes = data.num_nodes
    num_rels = data.num_rels # not include reverse edge type
    windows_size = args.windows_size
    num_windows = (history_len - 1) // windows_size + 1
    # model create
    model = WTNet(
        args.input_dim,
        args.hidden_dims,
        num_nodes,
        num_rels,
        history_len,
        windows_size,
        args.num_heads,
        args.num_transformer_layers,
        args.num_transformer_hiddens,
        args.dropout,
        num_windows=num_windows,
        short_cut=args.short_cut,
        layer_norm=args.layer_norm,
        activation="relu",
        time_encoding=args.time_encoding,
        time_encoding_independent=args.time_encoding_independent,
    )
    device = utils.get_device(args)
    model = model.to(device)

    # note that one should move the pre-trained model to the '../model/<data-name>/' in advance
    inductive(model, test_list, num_rels, num_nodes, mode="test", model_name = model_state_file)
    

    sys.exit()


