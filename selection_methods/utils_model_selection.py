import os
import random
import logging
import torch
import numpy as np
from sklearn.decomposition import PCA


def whitening(src_embeddings, tgt_embeddings):
    pca_model = PCA(n_components=min(src_embeddings.shape[0], src_embeddings.shape[1]), whiten=True)\
        .fit(np.concatenate([src_embeddings, tgt_embeddings], 0))
    return pca_model.transform(src_embeddings), pca_model.transform(tgt_embeddings)


random_candidates_dict = None

def random_candidate_sampling(src_embeddings, tgt_embeddings, src_ids, args):

    random.seed(args.seed)
    src_size = src_embeddings.shape[0]
    candidate_size = int(args.candidate_size)
    
    # Load data features from cache or datas et file
    cached_dir = f"./cached_data/{args.dataset}"
    if not os.path.exists(cached_dir):
        os.makedirs(cached_dir)
    cached_file = os.path.join(cached_dir, f"random_candidates_{args.max_sample_size}")
    # load processed dataset or process the original dataset

    global random_candidates_dict

    if random_candidates_dict is None:
        if os.path.exists(cached_file):
            logging.info("Loading random candidates from cached file %s", cached_file)
            random_candidates_dict = torch.load(cached_file)
        else:
            random_candidates_dict = dict()
    
    candidate_key = f"{args.seed}_{candidate_size}"
    if candidate_key in random_candidates_dict:
        sampled_candidate_ids = random_candidates_dict[candidate_key]
    else:
        
        equal_matrix = (src_ids[:, np.newaxis].repeat(src_size, 1) == src_ids[np.newaxis, :].repeat(src_size, 0)).astype(np.float32)
        neg_coords = np.where(equal_matrix==0)
        
        neg_ids = []
        cur_row = -1
        sampled_candidate_ids = []
        neg_size = candidate_size - 1
        num_sampling = 0
        for i in range(neg_coords[0].shape[0]):
            if neg_coords[0][i] != cur_row:
                cur_row = neg_coords[0][i]
                if len(neg_ids) > 0:
                    sampled_candidate_ids.extend([num_sampling] + random.sample(neg_ids[-1], neg_size))
                    num_sampling += 1
                neg_ids.append([])
                
            neg_ids[-1].append(neg_coords[1][i])
        sampled_candidate_ids.extend([num_sampling] + random.sample(neg_ids[-1], neg_size))

        random_candidates_dict[candidate_key] = sampled_candidate_ids
        logging.info("Saving candidate ids to %s", cached_file)
        torch.save(random_candidates_dict, cached_file)
    
    sampled_candidate_embeddings = tgt_embeddings[sampled_candidate_ids].reshape(src_size, candidate_size, -1)


    return src_embeddings[:, np.newaxis, :].repeat(candidate_size, 1), sampled_candidate_embeddings