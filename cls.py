import numpy as np
from transformers import AutoTokenizer, AutoConfig
import torch

from tqdm import tqdm
import argparse
import os
import sys
import json
from process import get_loader, SQuAD_dataset


class Train_API():
    
    def __init__(self, args) -> None:
        self.input_file = args.input
        self.output_file = args.output
        self.device = torch.device("cpu") if args.cuda == -1 else torch.device(f"cuda:{args.cuda}")

        self.batch_size = args.batch_size
        if args.model == 'bert':
            self.model_name = f'bert-{args.model_size}-uncased'
        elif args.model == 'deberta':
            self.model_name = 'microsoft/deberta-xlarge'

        config = AutoConfig.from_pretrained(
            'SQuAD-test/pretrained/deberta-xlarge-config'
        )
        config.dropout =args.dropout
        tokenizer = AutoTokenizer.from_pretrained(
            'SQuAD-test/pretrained/deberta-xlarge-tokenizer'
        )
        
        self.model = torch.load('SQuAD-test/checkpoints/model_2.pt', map_location=self.device)        

        dataset = SQuAD_dataset(args.input)
        self.eval_example = dataset.examples
        self.eval_dataset = dataset.feat
        self.eval_loader = get_loader(dataset)

        self.batch_size = self.batch_size * torch.cuda.device_count()
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.adam_epsilon = 1e-8
        self.weight_decay = 0
        self.post_process_function = dataset.post_process_function


    def predict(self, output_file):
        self.model.to(self.device)
        self.model.eval()
        pbar = tqdm(total=len(self.eval_loader))
        with torch.no_grad():
            start, end = [],[]
            for batch_idx, batch in enumerate(self.eval_loader):
                batch = {k:v.to(self.device) for k,v in batch.items()}
                output = self.model(**batch)
                start_logits, end_logits = output.start_logits, output.end_logits
                start.append(start_logits)
                end.append(end_logits)
                pbar.update(1)
            start_logits = np.array(torch.cat(start).cpu())
            end_logits = np.array(torch.cat(end).cpu())
        # np.save('start.npy', start_logits)
        # np.save('end.npy', end_logits)
        # start_logits = np.load('start.npy')
        # end_logits = np.load('end.npy')
        preds = self.post_process_function(self.eval_example, self.eval_dataset, (start_logits, end_logits))
        out_file = open(output_file, 'w')
        # predictions = dict((p["id"], p["prediction_text"]) for p in preds[0])
        predictions = dict((p["id"], p["prediction_text"]) for p in preds)
        json.dump(predictions, out_file)


def construct_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--pre_seq_len', type=int, default=8)
    parser.add_argument('--model', type=str, choices=['bert', 'deberta'], default='deberta')
    parser.add_argument('--model_size', type=str, choices=['base', 'large'], default='base')
    parser.add_argument('--method', type=str, choices=['finetune', 'prefix'], default='prefix')
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--do_train', action='store_true', dest='do_train')
    parser.add_argument('--input', type=str, default='SQuAD-test/data/dev-v2.0.json')
    parser.add_argument('--output', type=str, default='SQuAD-test/out/prediction.json')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = construct_args()
    train_api = Train_API(args)
    train_api.predict(args.output)
