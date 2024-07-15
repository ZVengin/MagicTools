# MagicTools: A Training Framework Using DistributedDataParallel(DDP)
This is a repository for training the language models 
using the DistributedDataParallel framework.
This repository aims to provide a general training script
to train the models and alleviate the burden of re-writing
the training scripts.

__Please kindly star this repository if this repository 
is helpful to you!!!__

### Tutorial
- install the package
```shell
git clone https://github.com/ZVengin/MagicTools.git
cd MagicTools
pip install .
```

- how to use the package
  - inherit the `TrainUtils` class and implement the following
  functions. The example of how to implement these functions 
  is included in the `toy_train.py` file.
```python
from MagicTools import TrainUtils
class MyTrainUtils(TrainUtils):
    def get_model(self):
        # how to load the model
        return model

    def get_tokenizer(self):
        # how to load tokenizer
        return tokenizer

    def get_train_arguments(self):
        # define the commandline arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('--train_file',...)
        ...
        args = parser.parse_args()
        return args
    
    def get_test_arguments(self):
        # define the commandline arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('--test_file',...)
        ...
        args = parser.parse_args()
        return args

    def loss_function(self,model, batch):
        # define how to compute the loss.
        # the batch is a batch inputs returned by
        # the collate_fn.
        # e.g., 
        # input_ids = batch['input_ids'].to(model.device)
        # attention_mask = batch['attention_mask'].to(model.device)
        # labels = batch['label'].to(model.device)
        # loss = model(input_ids,attention_mask,labels).loss
        return loss

    def inference(self,model, tokenizer, batch,do_sample=False):
        # define how to predict the results during the evaluation period
        # the returned results should be the format like
        # {
        #   'index': index,
        #   'predict': predict_result,
        # }
        # the index should be the same as that in the batch
        return records

    def compute_score(self,records):
        # define how to compute the scores based on the
        # predict results
        # Note that, each inference result has been augmented 
        # with its corresponding dataset record.
        # so each record will contain the information in 
        # the original dataset
        return score

    def process_inputs(self,tokenizer,instance,is_train):
        # define how to process the each record of dataset
        # into the inputs of the model
        return inputs
        

    def construct_instance(self,data,tokenizer,is_train, is_chinese):
        # define how to preprocess the dataset
        # the processing results are a list data records
        return records

    def collate_fn(self,batch):
        # define how to combine a group of inputs into a batch
        return batch
```

- training script
```python
if __name__ == '__main__':
    train_utils = MyTrainUtils(mode='train')
    train_utils.train()
```

- test script
```python
if __name__ == '__main__':
    train_utils = MytrainUtils(mode='test')
    train_utils.test()
```

- running the training script using the torchrun on
multiple GPUs, e.g. running the `toy_train.py` on two GPUs
on a single machine. To run the `toy_train.py`, the wandb login
key needs to be set in the `toy_train.py`.
```shell
 torchrun --standalone --nproc_per_node=2 toy_train.py --train_file ./../dataset/train.json --dev_file ./../dataset/validation.json --log_dir ./../logs --batch_size 64
 ```

- when running the scripts on multiple nodes using the `SLURM` schedule system, 
we need specify the host ip address and the `NCCL_SOCKET_IFNAME` . To get the host ip address
and the `NCCL_SOCKET_IFNAME` automatically, we could run the following commands in 
the bash script of job submission, e.g. `run_toy_train.sh` .
```shell
# get the host IP address
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
echo nodes:${nodes}
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

## get the NCCL_SOCKET_IFNAME
IPOUT=$(ifconfig | grep 'enp')
NCCL_SOCKET_IFNAME=$( cut -d ':' -f 1 <<< ${IPOUT})
echo NCCL_SOCKET_IFNAME:$NCCL_SOCKET_IFNAME
```
then we could use the `torchrun` to run the training script with the following command
```shell
torchrun \
        --nproc_per_node=4 \
        --nnodes=2 \
        --rdzv_id $RANDOM \
        --rdzv_backend c10d \
        --rdzv_endpoint $head_node_ip:29500 \
        toy_train.py \
        --train_file ./../dataset/train.json \
        --dev_file ./../dataset/validation.json \
        --log_dir ./../logs \
        --batch_size $BATCH_SIZE
```
the `rdzv_id` is the id for the distributed training job, 
it is automatically set by the `torchrun`.
the `rdzv_backend` is the backend for process communication.
the `rdzv_endpoint` is the ip address and port of the host
process.
the `nproc_per_node` is the process number for each node,
usually one process for each GPU of a node.
the `nnodes` is the node number we used to run the job.

In the `SLURM` system, the job could be submitted with the 
following command
```shell
sbatch \
--partition=v \
--nodes=2 \
--gres=gpu:4  \
--mem=80G   \
-t 48:00:00 \
-o exp.out \
-J job_name \
--export=ALL,BATCH_SIZE=16,...\
run_toy_train.sh
```