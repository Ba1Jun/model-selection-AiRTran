#!/usr/bin/python3
import warnings
warnings.filterwarnings("ignore")

import argparse
import logging
import time
import os
import shutil
import json
import jsonlines
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
        for batch in tqdm(train_data_loader, desc='[encoding training src-tgt pairs]', leave=True):
            # forward
            batch_src_embeddings, batch_tgt_embeddings, _ = model(**batch)
            src_embeddings.append(batch_src_embeddings.to(torch.float16).cpu().numpy())
            tgt_embeddings.append(batch_tgt_embeddings.to(torch.float16).cpu().numpy())
            src_ids.append(batch["src_ids"].cpu().numpy())
        src_embeddings = np.concatenate(src_embeddings, 0)
        tgt_embeddings = np.concatenate(tgt_embeddings, 0)
        src_ids = np.concatenate(src_ids, 0)

    return src_embeddings, tgt_embeddings, src_ids


def obtain_test_embeddings(model, test_question_data_loader, test_candidate_data_loader):
    model.eval()
    with torch.no_grad():
        question_embeddings = []
        test_ground_truth = []
        for batch in tqdm(test_question_data_loader, desc='[encoding test questions]', leave=True):
            # forward
            test_ground_truth += list(batch["ground_truth"])
            del batch["ground_truth"]
            question_embedding = model.sentence_encoding(**batch)
            question_embeddings.append(question_embedding.to(torch.float16).cpu().numpy())
        question_embeddings = np.concatenate(question_embeddings, 0)

        candidate_embeddings = []
        for batch in tqdm(test_candidate_data_loader, desc='[encoding test candidates]', leave=True):
            # forward
            candidate_embedding = model.sentence_encoding(**batch)
            candidate_embeddings.append(candidate_embedding.to(torch.float16).cpu().numpy())
        candidate_embeddings = np.concatenate(candidate_embeddings, 0)

    return question_embeddings, candidate_embeddings, test_ground_truth


def dataset_encoding(args):
    cached_dir = f"./cached_data/{args.dataset}"
    if not os.path.exists(cached_dir):
        os.makedirs(cached_dir)
    from utils_data import ReQADataset as Dataset
    model = None
    plm_name = [s for s in args.model_name_or_path.split('/') if s !=''][-1]
    encoded_train_dataset_file = os.path.join(cached_dir, f'encoded_train_dataset_{plm_name}')
    encoded_test_dataset_file = os.path.join(cached_dir, f'encoded_test_dataset_{plm_name}')
    # load processed dataset or process the original dataset
    if os.path.exists(encoded_train_dataset_file):
        logging.info("%s already exists.", encoded_train_dataset_file)
    else:
        train_dataset = Dataset(args, split="train")
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
        
        train_saved_data = {
            "train_src_embeddings": train_src_embeddings,
            "train_tgt_embeddings": train_tgt_embeddings,
            "train_src_ids": train_src_ids,
        }

        logging.info("Saving encoded training dataset to %s", encoded_train_dataset_file)
        torch.save(train_saved_data, encoded_train_dataset_file)


    if os.path.exists(encoded_test_dataset_file):
        logging.info("%s already exists.", encoded_test_dataset_file)
    else:
        test_question_dataset = Dataset(args, split="test", data_type="question")
        test_candidate_dataset = Dataset(args, split="test", data_type="candidate")

        logging.info(f"test question size: {test_question_dataset.__len__()}")
        logging.info(f"test candidate size: {test_candidate_dataset.__len__()}")

        test_question_data_loader = torch.utils.data.DataLoader(
            test_question_dataset,
            shuffle=False,
            batch_size=args.batch_size,
            collate_fn=test_question_dataset.collate_fn)
        
        test_candidate_data_loader = torch.utils.data.DataLoader(
            test_candidate_dataset,
            shuffle=False,
            batch_size=args.batch_size,
            collate_fn=test_candidate_dataset.collate_fn)
        
        if model is None:
            # preparing model
            from models.dual_encoder import RankModel
            model = RankModel(args)
            model.to(args.device)
        
        test_src_embeddings, test_candidate_embeddings, test_ground_truths = obtain_test_embeddings(model, test_question_data_loader, test_candidate_data_loader)


        test_saved_data = {
            "test_src_embeddings": test_src_embeddings,
            "test_candidate_embeddings": test_candidate_embeddings,
            "test_ground_truths": test_ground_truths
        }
                
        logging.info("Saving encoded test dataset to %s", encoded_test_dataset_file)
        torch.save(test_saved_data, encoded_test_dataset_file)

def main(args: argparse.Namespace):
    # set device
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for model_name_or_path in args.model_name_or_paths:
        args.model_name_or_path = model_name_or_path
        dataset_encoding(args)
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Framework for Model Selection')

    parser.add_argument('--dataset', default="bioasq9b", type=str, nargs='?', help='Dataset from the HuggingFace Dataset library.')
    parser.add_argument('--model_name_or_paths', nargs='+', help='list of pretrained language model identifiers.')
    parser.add_argument('--cache_dir', type=str, default="./models/")
    parser.add_argument('--pooler', default="mean", type=str, nargs='?', help='pooling strategy for sentence classification (default: None)')
    parser.add_argument("--matching_func", default="dot", type=str)
    parser.add_argument('--batch_size', type=int, default=64, help='maximum number of sentences per batch (default: 64)')
    
    main(parser.parse_args())
