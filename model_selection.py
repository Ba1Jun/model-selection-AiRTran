#!/usr/bin/python3
import warnings
warnings.filterwarnings("ignore")

import argparse
import logging
import os
import json
import torch
import numpy as np

from tqdm import tqdm
from selection_methods.utils_model_selection import whitening, random_candidate_sampling

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def obtain_train_embeddings(model, train_data_loader):
    model.eval()
    with torch.no_grad():
        src_embeddings = []
        tgt_embeddings = []
        src_ids = []
        for batch in tqdm(train_data_loader, desc='[encoding training src-tgt pairs]', leave=False):
            # forward
            batch_src_embeddings, batch_tgt_embeddings, _ = model(**batch)
            src_embeddings.append(batch_src_embeddings.cpu().numpy())
            tgt_embeddings.append(batch_tgt_embeddings.cpu().numpy())
            src_ids.append(batch["src_ids"].cpu().numpy())
        src_embeddings = np.concatenate(src_embeddings, 0)
        tgt_embeddings = np.concatenate(tgt_embeddings, 0)
        src_ids = np.concatenate(src_ids, 0)

    return src_embeddings, tgt_embeddings, src_ids


def dataset_encoding(args):
    cached_dir = f"./cached_data/{args.dataset}"
    if not os.path.exists(cached_dir):
        os.makedirs(cached_dir)
    plm_name = [s for s in args.model_name_or_path.split('/') if s !=''][-1]
    cached_dataset_embedding_file = os.path.join(cached_dir, f'train_embedding_{plm_name}')
    # load processed dataset or process the original dataset
    if os.path.exists(cached_dataset_embedding_file):
        logging.info("Loading encoded dataset from cached file %s", cached_dataset_embedding_file)
        data_dict = torch.load(cached_dataset_embedding_file)
        train_src_embeddings = data_dict["train_src_embeddings"]
        train_tgt_embeddings = data_dict["train_tgt_embeddings"]
        train_src_ids = data_dict["train_src_ids"]
    else:
        # load dataset
        from utils_data import ReQADataset as Dataset
        # initialize datasets
        train_dataset = Dataset(args, split="train", is_ms=True)
        
        logging.info(f"train data size: {train_dataset.__len__()}")

        train_data_loader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=False,
            batch_size=args.batch_size,
            collate_fn=train_dataset.collate_fn)


        # preparing model
        from models.dual_encoder import RankModel
        model = RankModel(args)
        model.to(args.device)
        train_src_embeddings, train_tgt_embeddings, train_src_ids = obtain_train_embeddings(model, train_data_loader)

        saved_data = {
            "train_src_embeddings": train_src_embeddings,
            "train_tgt_embeddings": train_tgt_embeddings,
            "train_src_ids": train_src_ids,
        }
                
        logging.info("Saving encoded dataset to %s", cached_dataset_embedding_file)
        torch.save(saved_data, cached_dataset_embedding_file)
    return train_src_embeddings, train_tgt_embeddings, train_src_ids


def main(args: argparse.Namespace):
    # set device
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for model_name_or_path in args.model_name_or_paths:
        args.model_name_or_path = model_name_or_path
        embeddings_dict = {}

        train_src_embeddings, train_tgt_embeddings, train_src_ids = dataset_encoding(args)
        train_src_embeddings = np.nan_to_num(train_src_embeddings, nan=0.0)
        train_tgt_embeddings = np.nan_to_num(train_tgt_embeddings, nan=0.0)
        whitened_train_src_embeddings, whitened_train_tgt_embeddings = whitening(np.copy(train_src_embeddings), np.copy(train_tgt_embeddings))
        logging.info(f"Embedding size: {train_src_embeddings.shape}")
        for candidate_size in args.all_candidate_sizes:
            args.candidate_size = str(candidate_size)
            
            for method in args.methods:
                args.method = method
                TransMetric = None
                if args.method.startswith("Rreg"):
                    from selection_methods.AiRTran import AiRTran as TransMetric
                logging.info(f"{args.method} (candidate size {args.candidate_size}) for embeddings from {args.model_name_or_path} on dataset {args.dataset}.")

                if eval(args.save_results):
                    from utils_data import LLM_MODELS
                    if model_name_or_path in LLM_MODELS:
                        save_dir = f"output/{args.dataset}/model_selection_large"
                    else:
                        save_dir = f"output/{args.dataset}/model_selection_small"
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    results_file = f"{save_dir}/{args.method}.json"
                    if os.path.exists(results_file):
                        with open(results_file, "r") as f:
                            results_dict = json.load(f)
                    else:
                        results_dict = dict()


                if eval(args.save_results) and args.candidate_size in results_dict\
                    and args.model_name_or_path in results_dict[args.candidate_size] and not eval(args.overwrite_results):
                    logging.info("Skipping the candidate model already has been scored.")
                    logging.info("-------------------------------END-------------------------------\n")
                    continue
                else:
                    all_scores = []
                    all_times = []
                    for seed in args.seeds:
                        args.seed = int(seed)
                        logging.info(f"running by seed {args.seed}...")
                        metric = TransMetric(args)
                        cur_src_embeddings, cur_tgt_embeddings = whitened_train_src_embeddings, whitened_train_tgt_embeddings
                        repeated_src_embeddings, sampled_candidate_embeddings = random_candidate_sampling(np.copy(cur_src_embeddings), 
                                                                                                            np.copy(cur_tgt_embeddings), 
                                                                                                            np.copy(train_src_ids),
                                                                                                            args)
                        features = (repeated_src_embeddings * sampled_candidate_embeddings).reshape(-1, repeated_src_embeddings.shape[-1])
                        features = np.nan_to_num(features, nan=0.0)
                        labels = []
                        for i in range(repeated_src_embeddings.shape[0]):
                            labels.extend([1] + [0]*(int(args.candidate_size)-1))
                        labels = np.array(labels)

                        score, score_time = metric.score(np.copy(features), np.copy(labels))
                        all_scores.append(score)
                        all_times.append(score_time)

                if eval(args.save_results):
                    results_dict[args.candidate_size] = results_dict.get(args.candidate_size, dict())
                    results_dict[args.candidate_size][args.model_name_or_path] = {
                        "all_scores": all_scores,
                        "all_times": all_times,
                    }
                    with open(results_file, "w") as f:
                        json.dump(results_dict, f, indent=4)

                
                logging.info(f"all scores: {all_scores}")
                logging.info(f"all times: {all_times}")
                logging.info("-------------------------------END-------------------------------\n")
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Framework for Model Selection')
    parser.add_argument('--methods', default=[], nargs='+', help='List of Model selection method.')
    parser.add_argument('--dataset', default="bioasq9b", type=str, nargs='?', help='Dataset from the HuggingFace Dataset library.')
    parser.add_argument('--save_results', default="True", type=str, nargs='?', help='Whether to save results.')
    parser.add_argument('--model_name_or_paths', default=[], nargs='+', help='list of pretrained language model identifiers.')
    parser.add_argument('--cache_dir', type=str, default="./models/")
    parser.add_argument('--overwrite_results', type=str, default="False")
    parser.add_argument("--matching_func", default="dot", type=str)
    parser.add_argument('--pooler', default="mean", type=str, nargs='?', help='pooling strategy for sentence classification (default: None)')
    parser.add_argument('--all_candidate_sizes', default=[], nargs='+', help='list of candidate sizes.')
    parser.add_argument('--batch_size', type=int, default=64, help='maximum number of sentences per batch (default: 64)')
    parser.add_argument('--seeds', default=[2024], nargs='+', help='list of random seeds')

    main(parser.parse_args())
