import os
import random
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.append("..")
import utils
from torch import nn
import json
from config import args
from model import WTNet
from datetime import datetime
from torch.utils import data as torch_data
from torch import distributed as dist


def train_and_validate(args, model, train_list, valid_list, test_list, num_nodes, num_rels, model_state_file):
    world_size = utils.get_world_size()
    rank = utils.get_rank()
    model_parameter = model_state_file + '_parameter'
    if utils.get_rank() == 0:
        print(
            "\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\nstart training\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    if world_size > 1:
        parallel_model = nn.parallel.DistributedDataParallel(model, device_ids=[device])
    else:
        parallel_model = model
    optimizer = torch.optim.Adam(parallel_model.parameters(), lr=args.lr, weight_decay=5e-6)
    loss = list()
    best_mrr = 0
    current_epoch = 0
    state_dict = ''
    if args.continues and utils.get_rank() == 0:
        print(">>>>>>>>>>>>load model parameter....................")
        checkpoint = torch.load(model_parameter, map_location='cuda')
        state_dict = checkpoint['state_dict']

        if world_size > 1:
            parallel_model.module.load_state_dict(state_dict)
        else:
            parallel_model.load_state_dict(state_dict)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss = checkpoint['loss']
        best_mrr = checkpoint['best_mrr']
        current_epoch = checkpoint['epoch']

    for epoch in range(args.n_epoch):
        if utils.get_rank() == 0:
            print("\nepoch:" + str(epoch) + ' Time: ' + datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S'))
        if args.continues and current_epoch >= epoch:
            continue

        parallel_model.train()
        # total_params = sum(p.numel() for p in parallel_model.parameters())
        # print(f"Total number of parameters: {total_params}")  607745
        idx = [_ for _ in range(len(train_list))]  # timestamps index [0,1,2,3,...,n]
        random.shuffle(idx)
        losses = list()
        for future_sample_id in tqdm(idx, ncols=100):
            # for future_sample_id in idx:
            if future_sample_id == 0: continue
            # future_sample as the future graph index
            future_list = train_list[future_sample_id]
            # get history graph list
            if future_sample_id - args.history_len < 0:
                history_list = train_list[0: future_sample_id]
            else:
                history_list = train_list[future_sample_id - args.history_len:
                                          future_sample_id]

            # history_graph_list combine by windows_size
            windows_size = args.windows_size
            history_graph_list = split_subgraph(history_list, num_nodes, num_rels, windows_size)

            future_triple = torch.from_numpy(future_list).long().to(device)

            sampler = torch_data.DistributedSampler(future_triple, world_size, rank)
            future_loader = torch_data.DataLoader(future_triple, args.batch_size, sampler=sampler,
                                                  num_workers=args.n_worker)
            sampler.set_epoch(future_sample_id)
            for batch in future_loader:
                # sample negative triples for future graph, we will not sample the ground truth edges in the 'future_triple' when the strict is True
                # negative_num：64 batch_future_all ：(h,r,t)
                batch_future_all = utils.negative_sampling(future_triple, batch, args.negative_num, num_nodes, num_rels,
                                                           strict=True)
                pred = parallel_model(history_graph_list, batch_future_all)
                loss = model.get_loss(args, pred)
                losses.append(loss.item())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
                optimizer.step()
                optimizer.zero_grad()
                utils.synchronize()

        utils.synchronize()

        if utils.get_rank() == 0:
            avg_loss = sum(losses) / len(losses)
            print("average binary cross entropy: {}".format(avg_loss))

        # evaluation
        if utils.get_rank() == 0:
            print("valid dataset eval:")
        mrr_valid = test(model, valid_list, num_rels, num_nodes,epoch=epoch)

        if mrr_valid >= best_mrr:
            best_mrr = mrr_valid
            if utils.get_rank() == 0:
                torch.save({'state_dict': model.state_dict(), 'epoch': epoch, 'args': args}, model_state_file)
                print("best_mrr updated(epoch %d)!" % epoch)
            utils.synchronize()

        if utils.get_rank() == 0:
            print("\n---------------------------------")
            # if utils.get_rank() == 0:
            print(">>>>>>>>>>>>>>>>>>>>>save model parameter<<<<<<<<<<<<")
            torch.save({'state_dict': model.state_dict(), 'epoch': epoch, 'args': args, 'losses': losses,
                        'optimizer_state_dict': optimizer.state_dict(), 'loss': loss, 'best_mrr': best_mrr},
                       model_parameter)
        utils.synchronize()

    # testing
    if rank == 0:
        if os.path.exists(model_parameter):
            os.remove(model_parameter)
        print("\nFinal eval test dataset with best model:...")
    mrr_test = test(model, test_list, num_rels, num_nodes, mode="test", model_name=model_state_file)

    return best_mrr


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
def test(model, test_list, num_rels, num_nodes, mode="train", model_name=None,epoch=None):
    world_size = utils.get_world_size()
    rank = utils.get_rank()
    result_name = model_state_file + '.txt'
    if rank == 0 and not os.path.exists('result'):
        os.mkdir('result')
    if mode == "test":
        # test mode: load parameter form file
        checkpoint = torch.load(model_name, map_location=device)
        if utils.get_rank() == 0:
            args____format = "\nLoad Model name: {}. Using best epoch : {}. \n\n args:{}.".format(model_name,
                                                                                                  checkpoint['epoch'],
                                                                                                  checkpoint['args'])
            print(args____format)  # use best stat checkpoint
            with open('result/' + result_name, 'a') as f:
                f.write(args____format)
                f.write("\n")
                f.close()
            print(
                "\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\nstart test\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(device)

    idx = [_ for _ in range(len(test_list))]  # timestamps index [0,1,2,3,...,n]

    model.eval()
    rankings = []
    test_dataset_nums = 0
    for future_sample_id in tqdm(idx):
        if future_sample_id < args.history_len: continue
        # future_sample as the future graph index
        test_dataset_nums += 1
        future_list = test_list[future_sample_id][:, :3]
        # get history graph list
        history_list = test_list[future_sample_id - args.history_len:
                                 future_sample_id]
        # history_graph_list combine by windows_size
        windows_size = args.windows_size

        history_graph_list = split_subgraph(history_list, num_nodes, num_rels, windows_size)
        future_triple = torch.from_numpy(future_list).long().to(device)
        time_filter_data = {
            'num_nodes': num_nodes,
            'edge_index': torch.stack([future_triple[:, 0], future_triple[:, 2]]),
            'edge_type': future_triple[:, 1]
        }
        sampler = torch_data.DistributedSampler(future_triple, world_size, rank)
        future_loader = torch_data.DataLoader(future_triple, args.batch_size, sampler=sampler,
                                              num_workers=args.n_worker)

        for batch in future_loader:
            t_batch, h_batch = utils.all_negative(num_nodes, batch)
            t_pred = model(history_graph_list, t_batch)
            h_pred = model(history_graph_list, h_batch)

            pos_h_index, pos_r_index, pos_t_index = batch.t()

            # time_filter Rank
            timef_t_mask, timef_h_mask = utils.strict_negative_mask(time_filter_data, batch[:, [0, 2, 1]])
            timef_t_ranking = utils.compute_ranking(t_pred, pos_t_index, timef_t_mask)
            timef_h_ranking = utils.compute_ranking(h_pred, pos_h_index, timef_h_mask)
            rankings += [timef_t_ranking, timef_h_ranking]
            utils.synchronize()
    utils.synchronize()
    if utils.get_rank() == 0:
        print(">>>>>>>test dataset number:{}", test_dataset_nums)
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
        metrics_dict['time'] = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
        result = json.dumps(metrics_dict, indent=4)
        print(result)

        with open('result/' + result_name, 'a') as f:
            f.write('\n')
            if epoch is not None:
                f.write('epoch:{} datatime:{}'.format(epoch, datetime.now()))
            else:
                f.write('finally result datatime:{}'.format(epoch, datetime.now()))
            f.write('\n')
            f.write(result)

    mrr = (1 / all_ranking.float()).mean()

    return mrr


if __name__ == '__main__':

    utils.set_rand_seed(2023)
    working_dir = utils.create_working_directory(args)
    windows_size = args.windows_size
    # bsize:16-neg:64-hislen:8-msg:distmult-aggr:pna-dim:64+[64, 64, 64, 64]|True|True|True|True
    model_name = (f"bsize={args.batch_size}-neg={args.negative_num}-hislen={args.history_len}"
                  f"-heads={args.num_heads}-tlayer={args.num_transformer_layers}-thidden={args.num_transformer_hiddens}"
                  f"-dim={args.input_dim}+{args.hidden_dims}-ws={windows_size}"
                  f"_{args.short_cut}_{args.layer_norm}_{args.time_encoding}_{args.time_encoding_independent}")

    model_state_file = model_name + "" + args.parameter_id

    # load datasets
    data = utils.load_data(args.dataset)
    num_nodes = data.num_nodes
    if utils.get_rank() == 0:
        print("# Sanity Check: stat name : {}".format(model_state_file))
        print("# Sanity Check:  entities: {}".format(data.num_nodes))
        print("# Sanity Check:  relations: {}".format(data.num_rels))
        print("# Sanity Check:  edges: {}".format(len(data.train)))

    # change the view of the data
    # [[s,r,o,t],[s,r,o,t],[s,r,o,t],...] -->> [ [ [s,r,o,t],[s,r,o,t] ], [ [s,r,o,t] ],...]
    train_list_sp = utils.split_by_time(data.train, stat_show=False)
    valid_list_sp = utils.split_by_time(data.valid, stat_show=False)
    test_list_sp = utils.split_by_time(data.test, stat_show=False)
    history_len = args.history_len
    all_list = train_list_sp + valid_list_sp + test_list_sp
    train_list = train_list_sp
    valid_list = train_list[-args.history_len:] + valid_list_sp
    test_list = valid_list[-args.history_len:] + test_list_sp

    # not include reverse edge type
    num_rels = data.num_rels
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

    if args.test:
        test(model, test_list, num_rels, num_nodes, mode="test", model_name=model_state_file)
    else:
        train_and_validate(args, model, train_list, valid_list, test_list, num_nodes, num_rels, model_state_file)

    sys.exit()
