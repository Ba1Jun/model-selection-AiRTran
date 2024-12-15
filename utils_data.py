import torch
import random
import copy
import re
import os
import json
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer


DO_LOWER_CASE = {
    "bert-base-uncased": True,
    "bert-base-cased": False,
    "roberta-base": False,
    "dmis-lab/biobert-base-cased-v1.1": False,
    "google/electra-base-discriminator": True,
    "princeton-nlp/unsup-simcse-bert-base-uncased": True,
    "princeton-nlp/sup-simcse-bert-base-uncased": True,
    "openai-gpt": True,
    "facebook/bart-base": False,
    "allenai/scibert_scivocab_cased": False,
    "allenai/scibert_scivocab_uncased": True,
    "nghuyong/ernie-2.0-base-en": True,
    "albert-base-v2": True,
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext": True,
    "michiyasunaga/BioLinkBERT-base": True,
    "distilbert-base-cased": False,
    "distilbert-base-uncased": True,
    "distilroberta-base": False,
    "distilgpt2": False,
    "distilbert-base-multilingual-cased": False,


    "sentence-transformers/sentence-t5-base": False,
    "Muennighoff/SGPT-125M-mean-nli": False,
    "facebook/contriever": True,
    "intfloat/e5-base": True,
    "BAAI/llm-embedder": True,
    "BAAI/bge-base-en": True,
    "sentence-transformers/gtr-t5-base": False,
    "thenlper/gte-base": True,
    "avsolatorio/GIST-Embedding-v0": True,
    "microsoft/mpnet-base": True,

    "IEITYuan/Yuan2-2B-hf": False,
    "facebook/opt-1.3b": False,
    "Qwen/Qwen1.5-1.8B": False,
    "deepseek-ai/deepseek-coder-1.3b-base": False,
    "EleutherAI/pythia-1.4b": False,
    "microsoft/phi-1_5": False,
    "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T": False,

    "bigscience/bloom-1b7": False,
    "tiiuae/falcon-rw-1b": False,
    "Muennighoff/SGPT-1.3B-mean-nli": False,
    "HuggingFaceTB/cosmo-1b": False,
    "EleutherAI/gpt-neo-1.3B": False,
    "cognitivecomputations/TinyDolphin-2.8-1.1b": False,

    "stabilityai/stable-code-3b": False,
    "FreedomIntelligence/Apollo-2B": False,
    "google/gemma-2b": False,
    "microsoft/phi-2": False,
    "OEvortex/EMO-2B": False,
    "pansophic/rocket-3B": False,
    "stabilityai/stablelm-3b-4e1t": False,

    "google/gemma-7b": False,
    "THUDM/chatglm3-6b": False,
    "meta-llama/Meta-Llama-3-8B": False,
    "Qwen/Qwen1.5-7B": False,
    "mistralai/Mistral-7B-v0.3": False,
}

SENTENCE_TRANSFORMERS_MODELS = [
    "sentence-transformers/sentence-t5-base",
    "sentence-transformers/gtr-t5-base",
    "Muennighoff/SGPT-125M-mean-nli",
    "intfloat/e5-base",
    "thenlper/gte-base",
    "avsolatorio/GIST-Embedding-v0",
]

LLM_MODELS = {
    "tiiuae/falcon-rw-1b": 2048,
    "Muennighoff/SGPT-1.3B-mean-nli": 2048,
    "HuggingFaceTB/cosmo-1b": 2048,
    "EleutherAI/gpt-neo-1.3B": 2048,
    "cognitivecomputations/TinyDolphin-2.8-1.1b": 2048,

    "IEITYuan/Yuan2-2B-hf": 2048,
    "facebook/opt-1.3b": 2048,
    "Qwen/Qwen1.5-1.8B": 2048,
    "bigscience/bloom-1b7": 2048,
    "deepseek-ai/deepseek-coder-1.3b-base": 2048,
    "EleutherAI/pythia-1.4b": 2048,
    "microsoft/phi-1_5": 2048,
    "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T": 2048,

    "stabilityai/stable-code-3b": 2560,
    "FreedomIntelligence/Apollo-2B": 2048,
    "google/gemma-2b": 2048,
    "microsoft/phi-2": 2560,
    "OEvortex/EMO-2B": 2048,
    "pansophic/rocket-3B": 2560,
    "stabilityai/stablelm-3b-4e1t": 2560,

    "google/gemma-7b": 3072,
    "THUDM/chatglm3-6b": 4096,
    "meta-llama/Meta-Llama-3-8B": 4096,
    "Qwen/Qwen1.5-7B": 4096,
    "mistralai/Mistral-7B-v0.3": 4096,
}

MODEL_WITHOUT_PAD = [
    "openai-gpt", 
    "distilgpt2", 
    "Qwen/Qwen-1_8B", 
    "IEITYuan/Yuan2-2B-hf",
    "EleutherAI/pythia-1.4b",
    "microsoft/phi-1",
    "microsoft/phi-1_5",
    "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    "microsoft/phi-2",
    "pansophic/rocket-3B",
    "stabilityai/stablelm-3b-4e1t",
    "stabilityai/stable-code-3b",
    "tiiuae/falcon-rw-1b",
    "HuggingFaceTB/cosmo-1b",
    "EleutherAI/gpt-neo-1.3B",
    "meta-llama/Meta-Llama-3-8B",
    "mistralai/Mistral-7B-v0.3",
]


def max_length(dataset: str):
    if dataset.startswith("bioasq"):
        return 24, 168
    elif dataset == "scifact":
        return 36, 512
    elif dataset == "mrpc":
        return 36, 36
    elif dataset == "mutual":
        return 196, 36
    elif dataset == "nq": # 23, 100(max); 11, 100(90%)
        return 24, 168
    elif dataset == "squad": # 15, 45
        return 24, 64


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, 
                                                       cache_dir=args.cache_dir,
                                                       trust_remote_code=True, 
                                                       do_lower_case=DO_LOWER_CASE[args.model_name_or_path])
        if args.model_name_or_path in MODEL_WITHOUT_PAD:
            self.pad_token_id = 0
        else:
            self.pad_token_id = self.tokenizer.pad_token_id
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
    
    def tokenize(self, texts, text_type):
        tokens = []
        input_ids = []
        for text in tqdm(texts, desc=f'[tokenize {text_type}]', leave=True):
            tokens.append(self.tokenizer.tokenize(str(text)))
            input_ids.append(self.tokenizer.convert_tokens_to_ids(tokens[-1]))
        return tokens, input_ids
    
    def padding(self, input_ids, max_len, backward=False):
        _input_ids = list(input_ids)
        for i, item in enumerate(_input_ids):
            if max_len == -1:
                _input_ids[i] = ([self.cls_token_id] if self.cls_token_id is not None else []) + item[-510:] if backward else item[:510] + ([self.sep_token_id] if self.sep_token_id is not None else [])
            else:
                _input_ids[i] = ([self.cls_token_id] if self.cls_token_id is not None else []) + item[-max_len+2:] if backward else item[:max_len-2] + ([self.sep_token_id] if self.sep_token_id is not None else [])
        # import pdb; pdb.set_trace()
        max_len = max([len(s) for s in _input_ids])
        input_ids = np.array([item + [self.pad_token_id] * (max_len - len(item)) for item in _input_ids], dtype=np.int32)
        attention_mask = np.array([[1] * len(item) + [0] * (max_len-len(item)) for item in _input_ids], dtype=np.int32)
        input_ids = torch.LongTensor(input_ids)
        attention_mask = torch.LongTensor(attention_mask)

        return input_ids.to(self.device), attention_mask.to(self.device)


class ReQADataset(BaseDataset):
    def __init__(self, args, split='train', data_type=None):
        super(ReQADataset, self).__init__(args)
        self.split = split
        self.data_type = data_type
        self.dataset = args.dataset
        self.data_folder = f"./data/{args.dataset}"
        self.max_question_len, self.max_answer_len = max_length(self.dataset)
        self.process()

    def process(self):
        # Load data features from cache or datas et file
        cached_dir = f"./cached_data/{self.dataset}"
        if not os.path.exists(cached_dir):
            os.makedirs(cached_dir)
        plm_name = [s for s in self.args.model_name_or_path.split('/') if s !=''][-1]
        cached_dataset_file = os.path.join(cached_dir, f'{self.split}_{plm_name}')

        # load processed dataset or process the original dataset
        if os.path.exists(cached_dataset_file):# and not self.args.overwrite_cache:
            logging.info("Loading dataset from cached file %s", cached_dataset_file)
            data_dict = torch.load(cached_dataset_file)
            self.question_input_ids = data_dict["questions"]
            self.answer_input_ids = data_dict["answers"]
            if self.split == 'test':
                self.ground_truths = data_dict['ground_truths']
            else:
                self.question_ids = data_dict['question_ids']
        else:
            logging.info("Creating instances from dataset file at %s", self.data_folder)
            json_file = self.data_folder + f'/{self.split}.json'
            with open(json_file, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)

            if self.split == 'test':
                questions = raw_data['questions']
                answers = raw_data['candidates']
                self.ground_truths = raw_data['ground_truths']
            else:
                questions = raw_data['questions']
                answers = raw_data['answers']
                self.question_ids = raw_data['question_ids']
            
            question_tokens, self.question_input_ids = self.tokenize(questions, 'question')
            answer_tokens, self.answer_input_ids = self.tokenize(answers, 'answer  ')
                
            logging.info(f"question: {questions[0]}")
            logging.info(f'question tokens: {question_tokens[0]}')
            logging.info(f'question input ids: {self.question_input_ids[0]}')
            logging.info('')
            logging.info(f"answer: {answers[0]}")
            logging.info(f'answer tokens: {answer_tokens[0]}')
            logging.info(f'answer input ids: {self.answer_input_ids[0]}')
            logging.info('')
            # save data
            if self.split == 'test':
                saved_data = {
                    'questions': self.question_input_ids,
                    'answers': self.answer_input_ids,
                    'ground_truths': self.ground_truths
                }
            else:
                saved_data = {
                    'questions': self.question_input_ids,
                    'answers': self.answer_input_ids,
                    'question_ids': self.question_ids
                }
                
            logging.info("Saving processed dataset to %s", cached_dataset_file)
            torch.save(saved_data, cached_dataset_file)
        
        
    def __len__(self):
        if self.split == "test" and self.data_type == "candidate":
            return len(self.answer_input_ids)
        else:
            return len(self.question_input_ids)

    def __getitem__(self, idx):
        if self.split == 'test':
            if self.data_type == "question":
                return (self.args, self.question_input_ids[idx], self.ground_truths[idx])
            elif self.data_type == "candidate":
                return (self.args, self.answer_input_ids[idx])
        else:
            return (self.args, 
                    self.question_input_ids[idx],
                    self.answer_input_ids[idx],
                    self.question_ids[idx])
            
    
    def collate_fn(self, raw_batch):
        args = raw_batch[-1][0]
        batch = dict()
        if self.split == 'test':
            if self.data_type == "question":
                max_len = self.max_question_len if self.args.model_name_or_path in LLM_MODELS else -1
                _, input_ids, ground_truth = list(zip(*raw_batch))
                batch['ground_truth'] = ground_truth
                batch['input_ids'], batch['attention_mask'] = self.padding(input_ids, max_len, backward=True if self.dataset=="mutual" else False)
            elif self.data_type == "candidate":
                max_len = self.max_answer_len if self.args.model_name_or_path in LLM_MODELS else -1
                _, input_ids = list(zip(*raw_batch))
                batch['input_ids'], batch['attention_mask'] = self.padding(input_ids, max_len, backward=False)
        else:
            _, question_input_ids, answer_input_ids, question_ids = list(zip(*raw_batch))
            batch['src_input_ids'], batch['src_attention_mask'] = self.padding(question_input_ids, self.max_question_len, backward=True if self.dataset=="mutual" else False)
            batch['tgt_input_ids'], batch['tgt_attention_mask'] = self.padding(answer_input_ids, self.max_answer_len, backward=False)
            batch['src_ids'] = torch.LongTensor(list(question_ids)).to(self.device)
            
        return batch