import json
from collections import defaultdict
from datasets.arrow_dataset import Dataset
import torch
from torch.utils.data.sampler import SequentialSampler
from torch.utils.data import DataLoader, dataloader
from transformers import default_data_collator
from transformers import AutoTokenizer, EvalPrediction

from utils_qa import postprocess_qa_predictions



class SQuAD_dataset(Dataset):
    def __init__(self, path_to_dev) -> None:
        
        self.raw_data = self.get_raw(path_to_dev)
        self.column_data = self.to_column(raw_data=self.raw_data) 
        self.tokenized_data = self.tokenize(self.column_data)
        self.feat = self.get_feature()
        self.examples = self.get_example()
        self.tokenized_data.pop('example_id')
        self.tokenized_data.pop('offset_mapping')
        self.input_ids = self.tokenized_data['input_ids']
        self.token_type_ids = self.tokenized_data['token_type_ids']
        self.attention_mask = self.tokenized_data['attention_mask']

    def get_raw(self, path_to_dev) -> list:
        raw = []
        f = open(path_to_dev, 'r')
        squad = json.load(f)
        for example in squad["data"]:
            title = example.get("title", "")
            for paragraph in example["paragraphs"]:
                context = paragraph["context"]  # do not strip leading blank spaces GH-2585
                for qa in paragraph["qas"]:
                    question = qa["question"]
                    id_ = qa["id"]

                    answer_starts = [answer["answer_start"] for answer in qa["answers"]]
                    answers = [answer["text"] for answer in qa["answers"]]

                    # Features currently used are "context", "question", and "answers".
                    # Others are extracted here for the ease of future expansions.
                    data = {
                        "title": title,
                        "context": context,
                        "question": question,
                        "id": id_,
                        "answers": {
                            "answer_start": answer_starts,
                            "text": answers,
                        },
                    }
                    raw.append(data)
        return raw
        print('finished loading')

    def to_column(self, raw_data:list)->defaultdict(list):
        column_data = {}
        column_data['id'] = []
        column_data['context'] = []
        column_data['title'] = []
        column_data['question'] = []

        for raw in raw_data:
            column_data['id'].append(raw['id'])
            column_data['title'].append(raw['title'])
            column_data['context'].append(raw['context'])
            column_data['question'].append(raw['question'])
        return column_data

    def tokenize(self, column_data) -> dict:
        column_data['question'] = [q.lstrip() for q in column_data['question']]
        pad_on_right = True
        max_seq_len = 384
        tokenizer = AutoTokenizer.from_pretrained('SQuAD-test/pretrained/deberta-xlarge-tokenizer')
        tokenized = tokenizer(
            column_data['question' if pad_on_right else 'context'],
            column_data['context' if pad_on_right else 'question'],
            truncation='only_second' if pad_on_right else 'only_first',
            max_length=max_seq_len,
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_mapping = tokenized.pop("overflow_to_sample_mapping")
        tokenized["example_id"] = []

        for i in range(len(tokenized["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized["example_id"].append(column_data["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized["offset_mapping"][i])
            ]
        # print(tokenized.keys())
        return tokenized

    def get_feature(self):
        features = []
        keys = self.tokenized_data.keys()
        for i in range(len(self.tokenized_data['input_ids'])):
            feature = {}
            for key in keys:
                feature[key] = self.tokenized_data[key][i]
            features.append(feature)
        return features

    def get_example(self):
        examples = []
        keys = self.column_data.keys()
        for i in range(len(self.column_data['context'])):
            example = {}
            for key in keys:
                example[key] = self.column_data[key][i]
            examples.append(example)
        return examples

    def __len__(self):
        return len(self.token_type_ids)

    def __getitem__(self, key):
        return {
            'input_ids': self.input_ids[key],
            'token_type_ids': self.token_type_ids[key],
            'attention_mask': self.attention_mask[key],
        }

    def post_process_function(self, examples, features, predictions, stage='eval'):
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            version_2_with_negative=True,
            n_best_size=20,
            max_answer_length=30,
            null_score_diff_threshold=0.0,
            output_dir='output',
            prefix=stage,
        )
        if True: # squad_v2
            formatted_predictions = [
                {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
            ]
        else:
            formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

        return formatted_predictions


def get_loader(tokenized_dataset):
    batch_size = 8
    generator = torch.Generator()
    generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
    sampler = SequentialSampler(tokenized_dataset)

    data_collator = default_data_collator

    return DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=data_collator,
        drop_last=False,
        num_workers=0,
        pin_memory=True,
    )


