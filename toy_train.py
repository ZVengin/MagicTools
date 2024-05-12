import torch
import json
import wandb
import torch.nn.parallel.DistributedDataParallel as DDP
from model import MagicModel
from dataset import get_dataloader
from magic_utils import PadSequence
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from datasets import load_dataset, load_metric
from torch.distributed import init_process_group, destroy_process_group

import argparse
import os
import random
import numpy as np

wandb.login(key='b3451a268e7b638ac4d8789aa1e8046da81710c5')


metric = load_metric('glue',"mnli")

def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def get_dataset(dataset_dir):
    actual_task = "mnli"
    data = load_dataset("glue", actual_task)
    index = 0
    for split in ["train", "dev", "test"]:
        dataset_path = os.path.join(dataset_dir,f'{split}.json')
        for inst in data[split]:
            inst['index'] = index
            index += 1
        with open(dataset_path,'w') as f:
            json.dump(data[split], f)


def process_instance(tokenizer,instance,is_train):
    inputs = tokenizer(instance['sentence'], truncation=True, add_special_tokens=True)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    index = instance['index']
    instance = {
        'input_ids': input_ids,
        'attention_mask':attention_mask,
        'label':instance['label'],
        'index':index
    }
    return instance


def collate_batch(batch):
    input_ids, attention_mask, labels, indexs = [list() for i in range(3)]
    for inst in batch:
        input_ids.append(inst['input_ids'])
        attention_mask.append(inst['attention_mask'])
        indexs.append(inst['index'])
        labels.append(inst['label'])

    max_len = max([len(x) for x in input_ids])
    input_ids = PadSequence(input_ids, max_len=max_len, pad_token_id=tokenizer.pad_token_id)
    attention_mask = PadSequence(attention_mask,max_len=max_len, pad_token_id=0)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    batch = {'input_ids': input_ids, 'attention_mask': attention_mask, 'label':labels,'index':indexs}
    return batch



def evaluate(model, device, test_loader):
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total

    return accuracy


def get_loss(model, batch):
    loss = model._model(
        input_ids=batch['input_ids'].to(model._local_rank),
        attention_mask=batch['attention_mask'].to(model._local_rank),
        labels=batch['label']
    ).loss
    return loss


def get_predictions(model, tokenizer, batch,do_sample):
    logits = model._model(
        input_ids= batch['input_ids'].to(model._local_rank),
        attention_mask=batch['attention_mask'].to(model._local_rank)
    ).logits.detach()
    records = []
    for index,logit in zip(batch['index'], logits):
        predict_label = torch.topk(logit, k=1)[1].item()
        record = {
            'index': index,
            'predict_label': predict_label
        }
        records.append(record)
    return records

def compute_accuracy(records):
    return metric.compute(
        predictions=list(map(lambda x:x['predict_label'], records)),
        references=list(map(lambda x:x['label'],records))
    )


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--dev_file', type=str, required=True)
    parser.add_argument('--cache_dir', type=str, default='./cache')
    parser.add_argument('--use_cache', action='store_true')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float,default=1e-5)
    parser.add_argument('--adam_epsilon', type=float, default=1e-5)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--warmup_rate', type=float, default=0.1)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--log_dir', type=str, required=True)

    args = parser.parse_args()
    return args




def train(config):
    set_random_seeds(config.seed)
    # rank and world_size will be automatically set by torchrun
    init_process_group(backend='nccl')
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    device_id = int(os.environ['LOCAL_RANK'])
    global_rank = int(os.environ['RANK'])

    if global_rank == 0:
        wandb_run = wandb.init(
            project=f"MagicToolTest",
            name=f"MNLI",
            config=config
        )

    model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased")
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
    model.to("cuda:{}".format(device_id))
    model = DDP(model, device_ids=[device_id])

    magic_model = MagicModel(
        model,
        tokenizer,
        loss_function=get_loss,
        inference=get_predictions,
        compute_score=compute_accuracy,
        process_outs=lambda tokenizer,outs,batch:outs,
        init_eval_score= 0,
        optimize_direction='max',
        distributed=True,
        local_rank=device_id,
        global_rank=global_rank)

    train_loader = get_dataloader(
        dataset_file=config.train_file,
        format='json',
        tokenizer=tokenizer,
        construct_instance=lambda x:x,
        process_inputs=process_instance,
        sample_weight=None,
        is_train=True,
        use_cache=False,
        cache_dir=config.cache_dir,
        batch_size=config.batch_size,
        collate_fn=collate_batch,
        num_workers=config.num_workers,
        distributed=True
    )

    if global_rank == 0:
        val_loader = get_dataloader(
            dataset_file=config.dev_file,
            format='json',
            tokenizer=tokenizer,
            construct_instance=lambda x: x,
            process_inputs=process_instance,
            sample_weight=None,
            is_train=False,
            use_cache=False,
            cache_dir=config.cache_dir,
            batch_size=config.batch_size,
            collate_fn=collate_batch,
            num_workers=config.num_workers,
            distributed=False
        )

    epoch_steps = len(train_loader)
    total_steps = epoch_steps*config.epochs
    warmup_steps = total_steps*config.warmup_rate
    magic_model.get_optimizer(
        lr = config.lr,
        training_steps=total_steps,
        warmup_steps=warmup_steps,
        weight_decay=config.weight_decay,
        adam_epsilon=config.adam_epsilon)

    magic_model.load_data('train', train_loader)
    magic_model.load_data('test', val_loader)

    model_path = os.path.join(config.log_dir, 'best_model.pth')
    if config.resume:
        magic_model.resume(model_path)

    for epoch in range(magic_model._epoch,config.epochs):
        magic_model.train(epoch)
        if global_rank == 0:
            records = magic_model.test()
            scores = magic_model.compute_score(records)
            wandb.log({'dev_acc':scores['accuracy']})
            if scores['accuracy'] >= magic_model._best_eval_score:
                magic_model._best_eval = scores['accuracy']
                magic_model.save_model(model_path=model_path)

destroy_process_group()

if __name__ == '__main__':
    config = get_arguments()
    train(config=config)
