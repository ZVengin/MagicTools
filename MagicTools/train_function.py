import torch
import wandb
import os
import random
import numpy as np
from .model import MagicModel
from .dataset import get_dataloader
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP


class TrainUtils:
    def get_model(self):
        pass

    def get_tokenizer(self):
        pass

    def get_arguments(self):
        pass

    def loss_function(self,model, batch):
        pass

    def inference(self,model, tokenizer, batch,do_sample):
        pass

    def compute_score(self,records):
        pass

    def process_inputs(self,tokenizer,instance,is_train):
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
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(random_seed)
        random.seed(random_seed)


    def train(self,config):
        self.set_random_seed(config.seed)
        # rank and world_size will be automatically set by torchrun
        init_process_group(backend='nccl')
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

        device_id = int(os.environ['LOCAL_RANK'])
        global_rank = int(os.environ['RANK'])
        print('local rank:{} | global rank:{}'.format(device_id, global_rank))

        if global_rank == 0:
            os.makedirs(config.log_dir, exist_ok=True)

            wandb_run = wandb.init(
                project=config.project_name,
                name=config.run_name,
                config=config
            )

        model = self.get_model()
        tokenizer = self.get_tokenizer()

        model.to("cuda:{}".format(device_id))
        model = DDP(model, device_ids=[device_id],find_unused_parameters=True)

        magic_model = MagicModel(
            model,
            tokenizer,
            loss_function=self.loss_function,
            inference=self.inference,
            compute_score=self.compute_score,
            process_outs=lambda tokenizer, outs, batch: outs,
            init_eval_score=config.init_eval_score,
            optimize_direction=config.optimize_direction,
            distributed=True,
            local_rank=device_id,
            global_rank=global_rank)

        train_loader = get_dataloader(
            dataset_file=config.train_file,
            format='json',
            tokenizer=tokenizer,
            construct_instance=self.construct_instance,
            process_inputs=self.process_inputs,
            sample_weight=None,
            is_train=True,
            use_cache=False,
            cache_dir=config.cache_dir,
            batch_size=config.batch_size,
            collate_fn=self.collate_fn,
            num_workers=config.num_workers,
            distributed=True
        )

        val_loader = get_dataloader(
            dataset_file=config.dev_file,
            format='json',
            tokenizer=tokenizer,
            construct_instance=self.construct_instance,
            process_inputs=self.process_inputs,
            sample_weight=None,
            is_train=False,
            use_cache=False,
            cache_dir=config.cache_dir,
            batch_size=config.batch_size,
            collate_fn=self.collate_fn,
            num_workers=config.num_workers,
            distributed=True
        )
        magic_model.load_data('train', train_loader)
        magic_model.load_data('test', val_loader)

        epoch_steps = len(train_loader)
        total_steps = epoch_steps * config.epochs
        warmup_steps = total_steps * config.warmup_rate

        self.get_optimizer(magic_model,config.lr,total_steps,warmup_steps,config.weight_decay,config.adam_epsilon)


        model_path = os.path.join(config.log_dir, 'best_model.pth')
        if config.resume:
            magic_model.resume(model_path)

        for epoch in range(magic_model._epoch, config.epochs):
            magic_model.train_epoch(epoch, accumulated_size=config.accumulated_size)
            records = magic_model.test()
            if global_rank == 0:
                score = magic_model.compute_score(records)
                wandb.log({'dev_score': score})
                if (config.optimize_direction == 'max' and score >= magic_model._best_eval_score
                ) or (config.optimize_direction == 'min' and score <= magic_model._best_eval_score):
                    magic_model._best_eval = score
                    magic_model.save_model(model_path=model_path)

        destroy_process_group()