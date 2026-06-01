import json
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import torch

sys.path.append("..")
import utils
from config import args
from model_view import WTNet


def icews18ts2datetime():
    start_date = datetime(2018, 1, 1)
    n = 304
    datemap = {}
    for i in range(n):
        current_date = start_date + timedelta(days=i)
        datemap[i] = current_date.strftime('%m/%d/%Y')
    return datemap


def safe_ts_name(ts, ts_vocab=None):
    ts = int(ts)
    if ts_vocab is not None:
        day_id = ts // 24
        if day_id in ts_vocab:
            return ts_vocab[day_id]
    return str(ts)


def split_subgraph_for_pathview(history_list, num_nodes, num_rels, windows_size, device):
    history_graph_list = []
    history_list_len = len(history_list)
    num, first_length = divmod(history_list_len, windows_size)
    start = 0
    for i in range(num + (1 if first_length else 0)):
        end = start + (first_length if i == 0 and first_length else windows_size)
        sub_history_list = np.concatenate(history_list[start:end])
        history_graph_list.append(utils.build_history_graph(num_nodes, num_rels, sub_history_list, device))
        start = end
    return history_graph_list


def path_to_text(path, entity_vocab, relation_vocab, ts_vocab, num_relation):
    triplets = []
    for ph, pt, pr, pts in path:
        ph_name = entity_vocab[int(ph)]
        pt_name = entity_vocab[int(pt)]
        pr_name = relation_vocab[int(pr) % num_relation]
        if int(pr) >= num_relation:
            pr_name += "^(-1)"
        pts_name = safe_ts_name(pts, ts_vocab)
        triplets.append(f"<{ph_name}, {pr_name}, {pt_name}, {pts_name}>")
    return " -> ".join(triplets)


def summarize_top_path_per_window(results, entity_vocab, relation_vocab, ts_vocab, num_relation):
    """
    For results from model.visualize(), keep only the top-1 path per window.
    Return format:
    {
        1: {"weight": ..., "path_text": ..., "time_min": ..., "time_max": ...},
        2: {...}
    }
    """
    best = {}

    for item in results:
        window_id = int(item["window_id"]) + 1
        weight = float(item["weight"])
        path_text = path_to_text(item["path"], entity_vocab, relation_vocab, ts_vocab, num_relation)

        if window_id not in best or weight > best[window_id]["weight"]:
            best[window_id] = {
                "weight": weight,
                "path_text": path_text,
                "time_min": safe_ts_name(item["window_time_min"], ts_vocab),
                "time_max": safe_ts_name(item["window_time_max"], ts_vocab),
            }

    return best


def path_process_collect(model, history_graph_list, triplet, num_nodes, entity_vocab, relation_vocab, ts_vocab,
                         filtered_data=None, rank_threshold=10, num_beam=10, k_per_window=5, max_total_paths=20,
                         future_sample_id=None, future_date=None, keep_modes=("tail",)):
    """
    Collect structured case data for a single query without printing or writing txt.
    """
    num_relation = len(relation_vocab)

    triplet = triplet.unsqueeze(0)
    orig_h, orig_r, orig_t = triplet.squeeze(0).tolist()

    orig_h_name = entity_vocab[orig_h]
    orig_t_name = entity_vocab[orig_t]
    orig_r_name = relation_vocab[orig_r]

    inverse = triplet[:, [2, 1, 0]].clone()
    inverse[:, 1] += num_relation

    t_batch, h_batch = utils.all_negative(num_nodes, triplet)

    with torch.no_grad():
        t_pred = model(history_graph_list, t_batch)
        h_pred = model(history_graph_list, h_batch)

        timef_t_mask, timef_h_mask = utils.strict_negative_mask(filtered_data, triplet[:, [0, 2, 1]])
        pos_h_index, pos_r_index, pos_t_index = triplet.unbind(-1)

        timef_t_ranking = utils.compute_ranking(t_pred, pos_t_index, timef_t_mask).squeeze(0)
        timef_h_ranking = utils.compute_ranking(h_pred, pos_h_index, timef_h_mask).squeeze(0)

    outputs = []

    # -------- tail --------
    if "tail" in keep_modes:
        rank = int(timef_t_ranking.item()) if torch.is_tensor(timef_t_ranking) else int(timef_t_ranking)
        if rank <= rank_threshold:
            results = model.visualize(
                history_graph_list,
                triplet,
                num_beam=num_beam,
                k_per_window=k_per_window,
                max_total_paths=max_total_paths
            )
            best = summarize_top_path_per_window(results, entity_vocab, relation_vocab, ts_vocab, num_relation)

            gw1 = best.get(1, {"weight": None, "path_text": None, "time_min": None, "time_max": None})
            gw2 = best.get(2, {"weight": None, "path_text": None, "time_min": None, "time_max": None})

            if gw1["weight"] is not None and gw2["weight"] is not None:
                dominant_window = "Gw1" if gw1["weight"] > gw2["weight"] else "Gw2"
                weight_gap = abs(gw1["weight"] - gw2["weight"])
            else:
                dominant_window = None
                weight_gap = None

            outputs.append({
                "future_sample_id": future_sample_id,
                "future_date": future_date,
                "mode": "tail",
                "triplet": f"<{orig_h_name}, {orig_r_name}, {orig_t_name}>",
                "query": f"({orig_h_name}, {orig_r_name}, ?)",
                "response": orig_t_name,
                "rank": rank,
                "gw1_top_weight": gw1["weight"],
                "gw1_top_path": gw1["path_text"],
                "gw1_time_range": [gw1["time_min"], gw1["time_max"]],
                "gw2_top_weight": gw2["weight"],
                "gw2_top_path": gw2["path_text"],
                "gw2_time_range": [gw2["time_min"], gw2["time_max"]],
                "dominant_window": dominant_window,
                "weight_gap": weight_gap,
            })

    # -------- head --------
    if "head" in keep_modes:
        rank = int(timef_h_ranking.item()) if torch.is_tensor(timef_h_ranking) else int(timef_h_ranking)
        if rank <= rank_threshold:
            results = model.visualize(
                history_graph_list,
                inverse,
                num_beam=num_beam,
                k_per_window=k_per_window,
                max_total_paths=max_total_paths
            )
            best = summarize_top_path_per_window(results, entity_vocab, relation_vocab, ts_vocab, num_relation)

            gw1 = best.get(1, {"weight": None, "path_text": None, "time_min": None, "time_max": None})
            gw2 = best.get(2, {"weight": None, "path_text": None, "time_min": None, "time_max": None})

            if gw1["weight"] is not None and gw2["weight"] is not None:
                dominant_window = "Gw1" if gw1["weight"] > gw2["weight"] else "Gw2"
                weight_gap = abs(gw1["weight"] - gw2["weight"])
            else:
                dominant_window = None
                weight_gap = None

            outputs.append({
                "future_sample_id": future_sample_id,
                "future_date": future_date,
                "mode": "head",
                "triplet": f"<{orig_t_name}, {orig_r_name}^(-1), {orig_h_name}>",
                "query": f"(?, {orig_r_name}, {orig_t_name})",
                "response": orig_h_name,
                "rank": rank,
                "gw1_top_weight": gw1["weight"],
                "gw1_top_path": gw1["path_text"],
                "gw1_time_range": [gw1["time_min"], gw1["time_max"]],
                "gw2_top_weight": gw2["weight"],
                "gw2_top_path": gw2["path_text"],
                "gw2_time_range": [gw2["time_min"], gw2["time_max"]],
                "dominant_window": dominant_window,
                "weight_gap": weight_gap,
            })

    return outputs


def is_case_matched(record,
                    desired_dominant=("Gw1", "Gw2"),
                    min_weight_gap=0.0,
                    min_top_weight=None,
                    max_rank=None):
    """
    Only keep records that meet the specified criteria.
    """
    if record["dominant_window"] is None:
        return False

    if desired_dominant is not None and record["dominant_window"] not in desired_dominant:
        return False

    if record["weight_gap"] is None or record["weight_gap"] < min_weight_gap:
        return False

    if max_rank is not None and record["rank"] > max_rank:
        return False

    if min_top_weight is not None:
        top_weight = max(record["gw1_top_weight"], record["gw2_top_weight"])
        if top_weight < min_top_weight:
            return False

    return True


def collect_and_save_filtered_cases(args, model, data_list, num_nodes, num_rels, entity_vocab, relation_vocab,
                                    model_name=None,
                                    shownum_each_time=20,
                                    rank_threshold=10,
                                    target_future_ids=None,
                                    keep_modes=("tail",),
                                    desired_dominant=("Gw1", "Gw2"),
                                    min_weight_gap=0.0,
                                    min_top_weight=None,
                                    max_rank=None,
                                    output_dir="result/reasoning_cases_filtered"):
    """
    Core function: only save data that meets the criteria.
    """
    ts_vocab = icews18ts2datetime()

    checkpoint = torch.load(model_name, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    model.eval()

    os.makedirs(output_dir, exist_ok=True)

    idx = [_ for _ in range(len(data_list))]
    saved_records = []

    for future_sample_id in idx:
        if future_sample_id < args.history_len:
            continue

        if target_future_ids is not None and future_sample_id not in target_future_ids:
            continue

        future_list = np.array(data_list[future_sample_id], copy=True)

        # Do not shuffle, keep order traceable
        if shownum_each_time is None:
            future_list_select = future_list[:, :3]
        else:
            future_list_select = future_list[:shownum_each_time, :3]

        future_ts = data_list[future_sample_id][0, 3]
        future_date = safe_ts_name(future_ts, ts_vocab)

        history_list = data_list[future_sample_id - args.history_len: future_sample_id]
        history_graph_list = split_subgraph_for_pathview(
            history_list, num_nodes, num_rels, args.windows_size, device
        )

        future_triple = torch.from_numpy(future_list_select).long().to(device)

        time_filter_data = {
            "num_nodes": num_nodes,
            "edge_index": torch.stack([future_triple[:, 0], future_triple[:, 2]]),
            "edge_type": future_triple[:, 1]
        }

        print(f"[Scanning] future_sample_id={future_sample_id}, date={future_date}, num_queries={len(future_triple)}")

        for triplet in future_triple:
            records = path_process_collect(
                model=model,
                history_graph_list=history_graph_list,
                triplet=triplet,
                num_nodes=num_nodes,
                entity_vocab=entity_vocab,
                relation_vocab=relation_vocab,
                ts_vocab=ts_vocab,
                filtered_data=time_filter_data,
                rank_threshold=rank_threshold,
                num_beam=10,
                k_per_window=5,
                max_total_paths=20,
                future_sample_id=future_sample_id,
                future_date=future_date,
                keep_modes=keep_modes
            )

            for rec in records:
                if is_case_matched(
                        rec,
                        desired_dominant=desired_dominant,
                        min_weight_gap=min_weight_gap,
                        min_top_weight=min_top_weight,
                        max_rank=max_rank
                ):
                    saved_records.append(rec)

    # Save the final filtered results
    save_json = os.path.join(output_dir, "matched_cases.json")
    with open(save_json, "w", encoding="utf-8") as f:
        json.dump(saved_records, f, ensure_ascii=False, indent=2)

    # Also save separately by Gw1/Gw2 for convenient selection
    gw1_records = [r for r in saved_records if r["dominant_window"] == "Gw1"]
    gw2_records = [r for r in saved_records if r["dominant_window"] == "Gw2"]

    with open(os.path.join(output_dir, "gw1_matched.json"), "w", encoding="utf-8") as f:
        json.dump(gw1_records, f, ensure_ascii=False, indent=2)

    with open(os.path.join(output_dir, "gw2_matched.json"), "w", encoding="utf-8") as f:
        json.dump(gw2_records, f, ensure_ascii=False, indent=2)

    print(f"[Saved] {save_json}")
    print(f"[Saved] {os.path.join(output_dir, 'gw1_matched.json')}")
    print(f"[Saved] {os.path.join(output_dir, 'gw2_matched.json')}")
    print(f"[Summary] matched total = {len(saved_records)}, Gw1 = {len(gw1_records)}, Gw2 = {len(gw2_records)}")


if __name__ == '__main__':

    utils.set_rand_seed(2023)
    working_dir = utils.create_working_directory(args)

    model_name = (f"bsize={args.batch_size}-neg={args.negative_num}-hislen={args.history_len}"
                  f"-msg={args.message_func}-aggr={args.aggregate_func}-heads={args.num_heads}"
                  f"-tlayer={args.num_transformer_layers}-thidden={args.num_transformer_hiddens}"
                  f"-values_way={args.values_way}-dim={args.input_dim}+{args.hidden_dims}-ws={args.windows_size}"
                  f"_{args.short_cut}_{args.layer_norm}"
                  f"_{args.time_encoding}_{args.time_encoding_independent}")

    model_state_file = model_name + args.parameter_id

    data = utils.load_data(args.dataset)

    if utils.get_rank() == 0:
        print("# Sanity Check: stat name : {}".format(model_state_file))
        print("# Sanity Check:  entities: {}".format(data.num_nodes))
        print("# Sanity Check:  relations: {}".format(data.num_rels))

    test_list_sp = utils.split_by_time(data.test, stat_show=False)

    num_nodes = data.num_nodes
    num_rels = data.num_rels
    entity_vocab = data.entity_dict
    relation_vocab = data.relation_dict

    num_windows = (args.history_len - 1) // args.windows_size + 1

    model = WTNet(
        args.input_dim,
        args.hidden_dims,
        num_nodes,
        num_rels,
        args.history_len,
        args.windows_size,
        args.num_mlp_layers,
        args.num_heads,
        args.num_transformer_layers,
        args.num_transformer_hiddens,
        args.dropout,
        num_windows=num_windows,
        message_func=args.message_func,
        aggregate_func=args.aggregate_func,
        short_cut=args.short_cut,
        layer_norm=args.layer_norm,
        activation="relu",
        time_encoding=args.time_encoding,
        time_encoding_independent=args.time_encoding_independent,
        values_way=args.values_way
    )
    device = utils.get_device(args)
    model = model.to(device)

    # =========================
    # Adjust your filter criteria here
    # =========================

    collect_and_save_filtered_cases(
        args=args,
        model=model,
        data_list=test_list_sp,
        num_nodes=num_nodes,
        num_rels=num_rels,
        entity_vocab=entity_vocab,
        relation_vocab=relation_vocab,
        model_name=model_state_file,
        shownum_each_time=100,  # Max queries per time point; None means all
        rank_threshold=10,  # Initial filter for path_process_collect
        target_future_ids=None,  # None means all available time points; can also be {30,31,32}
        keep_modes=("tail", "head"),  # Only collect tail; change to ("tail","head") if head is also needed

        desired_dominant=("Gw1", "Gw2"),  # Only Gw1 dominant: ("Gw1",); only Gw2 dominant: ("Gw2",)
        min_weight_gap=0.05,  # Minimum weight gap between two windows to save
        min_top_weight=None,  # Set a threshold for dominant path weight if needed, e.g. 0.1 / 0.2
        max_rank=10,  # Final rank filter when saving

        output_dir="result/reasoning_cases_filtered1"
    )

    sys.exit()
