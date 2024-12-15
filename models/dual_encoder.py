import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, BitsAndBytesConfig
from utils_data import SENTENCE_TRANSFORMERS_MODELS, LLM_MODELS
from sentence_transformers import models
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training


class RankModel(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.is_ST_Model = False
        self.is_LLM = False
        if args.model_name_or_path in LLM_MODELS:
            self.is_LLM = True
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4"
            )
            self.encoder = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, 
                                                                cache_dir=args.cache_dir,
                                                                trust_remote_code=True,
                                                                quantization_config=quantization_config
                                                                )
            
            self.encoder = prepare_model_for_kbit_training(self.encoder)

            peft_config = LoraConfig(
                r=8,
                lora_alpha=16,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION,
                target_modules=["q_proj", "v_proj"],
            )
            self.encoder = get_peft_model(self.encoder, peft_config)
            logging.info(self.encoder.print_trainable_parameters())

        elif args.model_name_or_path in SENTENCE_TRANSFORMERS_MODELS:
            self.is_ST_Model = True
            self.encoder = models.Transformer(args.model_name_or_path, cache_dir=args.cache_dir)
        else:
            self.encoder = AutoModel.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)


        self.args = args
        self.matching_func = args.matching_func
        self.temperature = args.temperature if hasattr(args, 'temperature') else 1
        self.ce_loss = nn.CrossEntropyLoss(reduction='mean')
        self.softmax = nn.Softmax(dim=1)
    

    def pooler(self, last_hidden_state, attention_mask):
        if self.args.pooler == 'cls':
            pooler_output = last_hidden_state[:, 0]
        if self.args.pooler == 'mean':
            if self.args.model_name_or_path in ["THUDM/chatglm3-6b"]:
                last_hidden_state = last_hidden_state.transpose(0, 1)
            pooler_output = torch.sum(last_hidden_state * attention_mask.unsqueeze(2), dim=1)
            pooler_output /= torch.sum(attention_mask, dim=1, keepdim=True)
        return pooler_output


    def sentence_encoding(self, input_ids, attention_mask):
        if self.is_LLM:
            encoder_outputs = self.encoder(input_ids=input_ids, 
                                            attention_mask=attention_mask,
                                            output_hidden_states=True,
                                            return_dict=True)
            hidden_states = encoder_outputs['hidden_states']
            if isinstance(hidden_states, tuple):
                last_hidden_state = hidden_states[-1]
            else:
                last_hidden_state = hidden_states
            # import pdb; pdb.set_trace()
        elif self.is_ST_Model:
            encoder_outputs = self.encoder({"input_ids": input_ids, 
                                            "attention_mask": attention_mask})
            last_hidden_state = encoder_outputs['token_embeddings']
        else:
            encoder_outputs = self.encoder(input_ids=input_ids, 
                                            attention_mask=attention_mask,
                                            output_hidden_states=True,
                                            return_dict=True)
            last_hidden_state = encoder_outputs['last_hidden_state']
        sentence_embeddings = self.pooler(last_hidden_state, attention_mask)
        
        return sentence_embeddings
    
    def matching(self, src_embeddings, tgt_embeddings, src_ids=None, tau_scale=True):

        if self.matching_func == "cos":
            src_embeddings = F.normalize(src_embeddings, dim=-1)
            tgt_embeddings = F.normalize(tgt_embeddings, dim=-1)
        predict_logits = src_embeddings.mm(tgt_embeddings.t())
        if tau_scale:
            predict_logits /= self.temperature

        if src_ids is not None:
            batch_size = src_embeddings.shape[0]
            logit_mask = (src_ids.unsqueeze(1).repeat(1, batch_size) == src_ids.unsqueeze(0).repeat(batch_size, 1)).float() - torch.eye(batch_size).to(src_ids.device)
            predict_logits -= logit_mask * 100000000

        return predict_logits

    def forward(self, src_input_ids, src_attention_mask, tgt_input_ids, tgt_attention_mask, src_ids=None):
        # cls:[bs, d_model]
        src_embeddings = self.sentence_encoding(src_input_ids, src_attention_mask)
        tgt_embeddings = self.sentence_encoding(tgt_input_ids, tgt_attention_mask)
 
        if self.training:
            # [bs, bs]
            predict_logits = self.matching(src_embeddings, tgt_embeddings, src_ids, True)
            # loss
            labels = torch.arange(0, predict_logits.shape[0]).to(src_input_ids.device)
            predict_loss = self.ce_loss(predict_logits, labels)
                
            predict_result = torch.argmax(predict_logits, dim=1)
            acc = labels == predict_result
            acc = (acc.int().sum() / (predict_logits.shape[0] * 1.0)).item()

            return predict_loss, acc
        else:
            return src_embeddings, tgt_embeddings, src_ids