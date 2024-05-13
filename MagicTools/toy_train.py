import torch
import wandb

from magic_utils import PadSequence
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from datasets import load_dataset, load_metric
from train_function import TrainUtils

import argparse


wandb.login(key='b3451a268e7b638ac4d8789aa1e8046da81710c5')


metric = load_metric('glue',"mnli")

class MyTrainUtils(TrainUtils):
    def inference(self,model, tokenizer, batch,do_sample):
        logits = model(
            input_ids=batch['input_ids'].to(model.device),
            attention_mask=batch['attention_mask'].to(model.device)
        ).logits.detach()
        records = []
        for index, logit in zip(batch['index'], logits):
            predict_label = torch.topk(logit, k=1)[1].item()
            record = {
                'index': index,
                'predict_label': predict_label
            }
            records.append(record)
        return records

    def loss_function(self,model, batch):
        loss = model(
            input_ids=batch['input_ids'].to(model.device),
            attention_mask=batch['attention_mask'].to(model.device),
            labels=batch['label']
        ).loss
        return loss


    def compute_score(self,records):
        return metric.compute(
            predictions=list(map(lambda x: x['predict_label'], records)),
            references=list(map(lambda x: x['label'], records))
        )['accuracy']


    def construct_instance(self,data,tokenizer,is_train, is_chinese):
        return data

    def process_inputs(self,tokenizer,instance,is_train):
        inputs = tokenizer(instance['sentence'], truncation=True, add_special_tokens=True)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        index = instance['index']
        instance = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': instance['label'],
            'index': index
        }
        return instance

    def collate_fn(self,batch):
        input_ids, attention_mask, labels, indexs = [list() for i in range(4)]
        for inst in batch:
            input_ids.append(inst['input_ids'])
            attention_mask.append(inst['attention_mask'])
            indexs.append(inst['index'])
            labels.append(inst['label'])

        max_len = max([len(x) for x in input_ids])
        input_ids = PadSequence(input_ids, max_len=max_len, pad_token_id=tokenizer.pad_token_id)
        attention_mask = PadSequence(attention_mask, max_len=max_len, pad_token_id=0)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        batch = {'input_ids': input_ids, 'attention_mask': attention_mask, 'label': labels, 'index': indexs}
        return batch


    def get_arguments(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--train_file', type=str, required=True)
        parser.add_argument('--dev_file', type=str, required=True)
        parser.add_argument('--cache_dir', type=str, default='./cache')
        parser.add_argument('--use_cache', action='store_true')
        parser.add_argument('--batch_size', type=int, default=64)
        parser.add_argument('--lr', type=float, default=1e-5)
        parser.add_argument('--weight_decay', type=float, default=1e-5)
        parser.add_argument('--adam_epsilon', type=float, default=1e-5)
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--seed', type=int, default=123)
        parser.add_argument('--epochs', type=int, default=50)
        parser.add_argument('--warmup_rate', type=float, default=0.1)
        parser.add_argument('--resume', action='store_true')
        parser.add_argument('--log_dir', type=str, required=True)
        parser.add_argument('--project_name', type=str, default='MagicToolsTest')
        parser.add_argument('--run_name', type=str, default='SST2')
        parser.add_argument('--optimize_direction', type=str, default='max')

        args = parser.parse_args()
        return args



    def get_model(self):
        model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased")
        return model

    def get_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
        return tokenizer


if __name__ == '__main__':
    train_utils = MyTrainUtils()
    config = train_utils.get_arguments()
    tokenizer = train_utils.get_tokenizer()
    train_utils.train(config=config)