import json

import torch
import wandb
import logging
import os
import random
import gc
import numpy as np
from .model import MagicModel
from .dataset import get_dataloader
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

class TrainUtils:
    def __init__(self, mode):
        if mode == 'train':
            self.config = self.get_train_arguments()
        else:
            self.config = self.get_test_arguments()
        self.model = None
        self.tokenizer = None

    def get_model(self):
        pass

    def get_tokenizer(self):
        pass

    def get_train_arguments(self):
        pass

    def get_test_arguments(self):
        pass

    def loss_function(self,model, batch):
        pass

    def inference(self,model, tokenizer, batch,do_sample):
        pass

    def compute_score(self,records):
        pass

    def process_inputs(self,tokenizer,instance,is_train):
        pass

    def process_outs(self, tokenizer, outs):
        pass

    def construct_instance(self,data,tokenizer,is_train, is_chinese):
        pass

    def collate_fn(self,batch):
        pass

    def get_optimizer(self,model, lr, total_steps, warmup_steps,weight_decay, adam_epsilon):
        model.get_optimizer(
            lr=lr,
            training_steps=total_steps,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            adam_epsilon=adam_epsilon)


    def set_random_seed(self,random_seed):
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(random_seed)
        random.seed(random_seed)


    def train(self):
        # rank and world_size will be automatically set by torchrun
        init_process_group(backend='nccl')
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        self.set_random_seed(self.config.seed)

        device_id = int(os.environ['LOCAL_RANK'])
        global_rank = int(os.environ['RANK'])
        print('local rank:{} | global rank:{}'.format(device_id, global_rank))

        if global_rank == 0:
            os.makedirs(self.config.log_dir, exist_ok=True)

            wandb_run = wandb.init(
                project=self.config.project_name,
                name=self.config.run_name,
                config=self.config
            )

        self.model = self.get_model()
        self.tokenizer = self.get_tokenizer()

        self.model.to("cuda:{}".format(device_id))
        self.model = DDP(self.model, device_ids=[device_id],find_unused_parameters=True)

        magic_model = MagicModel(
            self.model,
            self.tokenizer,
            loss_function=self.loss_function,
            inference=self.inference,
            compute_score=self.compute_score,
            process_outs=self.process_outs,
            init_eval_score=self.config.init_eval_score,
            optimize_direction=self.config.optimize_direction,
            distributed=True,
            local_rank=device_id,
            global_rank=global_rank)

        train_loader = get_dataloader(
            dataset_file=self.config.train_file,
            format='json',
            tokenizer=self.tokenizer,
            construct_instance=self.construct_instance,
            process_inputs=self.process_inputs,
            sample_weight=None,
            is_train=True,
            use_cache=self.config.use_cache,
            cache_dir=self.config.cache_dir,
            batch_size=self.config.batch_size,
            collate_fn=self.collate_fn,
            is_chinese=self.config.is_chinese,
            num_workers=self.config.num_workers,
            distributed=True,
            epoch_based=self.config.epoch_based
        )

        val_loader = get_dataloader(
            dataset_file=self.config.dev_file,
            format='json',
            tokenizer=self.tokenizer,
            construct_instance=self.construct_instance,
            process_inputs=self.process_inputs,
            sample_weight=None,
            is_train=False,
            use_cache=self.config.use_cache,
            cache_dir=self.config.cache_dir,
            batch_size=self.config.batch_size,
            collate_fn=self.collate_fn,
            is_chinese=self.config.is_chinese,
            num_workers=self.config.num_workers,
            distributed=True,
            epoch_based=False
        )
        magic_model.load_data('train', train_loader)
        magic_model.load_data('test', val_loader)

        epoch_steps = len(train_loader)
        total_steps = epoch_steps * self.config.epochs
        warmup_steps = total_steps * self.config.warmup_rate

        self.get_optimizer(magic_model,self.config.lr,total_steps,warmup_steps,self.config.weight_decay,self.config.adam_epsilon)


        model_path = os.path.join(self.config.log_dir, 'best_model.pth')
        if self.config.resume:
            resume_model_path = os.path.join(self.config.resume_dir, 'best_model.pth')
            magic_model.resume(resume_model_path, self.config.only_load_model)

        for epoch in range(magic_model._epoch, self.config.epochs):
            if self.config.epoch_based:
                magic_model._dataset['train'].dataset.set_epoch(epoch)
            magic_model.train_epoch(epoch, accumulated_size=self.config.accumulated_size)
            records = magic_model.test()
            if global_rank == 0:
                logger.info(f'==>>>record:{len(records)},data:{len(magic_model._dataset["test"].dataset.data)}')
                score = magic_model.compute_score(records)
                wandb.log({'dev_score': score})
                assert self.config.save_strategy in ['epoch','best_model'], 'save_strategy must be one of ["epoch","best_model"]'
                if self.config.save_strategy == 'epoch':
                    magic_model.save_model(model_path=model_path)
                    logger.info('==>>>model is saved at {}'.format(model_path))
                else:
                    if (self.config.optimize_direction == 'max' and score >= magic_model._best_eval_score
                    ) or (self.config.optimize_direction == 'min' and score <= magic_model._best_eval_score):
                        logger.info('==>>>best score:{}/eval score:{}'.format(magic_model._best_eval_score, score))
                        logger.info('==>>>best model is saved at {}'.format(model_path))
                        magic_model._best_eval_score = score
                        magic_model.save_model(model_path=model_path)


        del magic_model
        gc.collect()
        torch.cuda.empty_cache()
        destroy_process_group()


    def test(self):
        # rank and world_size will be automatically set by torchrun
        init_process_group(backend='nccl')
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        self.set_random_seed(self.config.seed)

        device_id = int(os.environ['LOCAL_RANK'])
        global_rank = int(os.environ['RANK'])
        print('local rank:{} | global rank:{}'.format(device_id, global_rank))

        if global_rank == 0:
            wandb_run = wandb.init(
                project=self.config.project_name,
                name=self.config.run_name,
                config=self.config
            )

        self.model = self.get_model()
        self.tokenizer = self.get_tokenizer()

        self.model.to("cuda:{}".format(device_id))
        self.model = DDP(self.model, device_ids=[device_id], find_unused_parameters=True)

        magic_model = MagicModel(
            self.model,
            self.tokenizer,
            inference=self.inference,
            compute_score=self.compute_score,
            process_outs=self.process_outs,
            distributed=True,
            local_rank=device_id,
            global_rank=global_rank)

        val_loader = get_dataloader(
            dataset_file=self.config.test_file,
            format='json',
            tokenizer=self.tokenizer,
            construct_instance=self.construct_instance,
            process_inputs=self.process_inputs,
            sample_weight=None,
            is_train=False,
            use_cache=self.config.use_cache,
            cache_dir=self.config.cache_dir,
            batch_size=self.config.batch_size,
            collate_fn=self.collate_fn,
            is_chinese=self.config.is_chinese,
            num_workers=self.config.num_workers,
            distributed=True,
            epoch_based=self.config.epoch_based
        )

        magic_model.load_data('test', val_loader)

        model_path = os.path.join(self.config.log_dir, 'best_model.pth')
        magic_model.resume(model_path, self.config.only_load_model)
        records = magic_model.test()

        if global_rank == 0:
            logger.info(f'==>>>record:{len(records)},data:{len(magic_model._dataset["test"].dataset.data)}')
            score = magic_model.compute_score(records)
            wandb.log({'test_score': score})
            eval_out_dir = os.path.join(self.config.log_dir, self.config.eval_out)
            os.makedirs(eval_out_dir, exist_ok=True)
            with open(os.path.join(eval_out_dir,'predictions.json'), 'w') as f:
                json.dump(records,f,indent=2)

            with open(os.path.join(eval_out_dir,'scores.json'),'w') as f:
                json.dump({'acc':'{:.4f}'.format(score)},f,indent=2)

        del magic_model
        gc.collect()
        torch.cuda.empty_cache()
        destroy_process_group()