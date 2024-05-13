# MagicTools
This is a repository for training the language models 
using the DistributedDataParallel framework.
This repository aims to provide a general training script
to train the models and alleviate the burden of re-writing
the training scripts.

### Tutorial
- install the package
```angular2html
pip install https://github.com/ZVengin/MagicTools.git
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

    def get_arguments(self):
        # define the commandline arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('--train_file',...)
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
    train_utils = MyTrainUtils()
    config = train_utils.get_arguments()
    tokenizer = train_utils.get_tokenizer()
    train_utils.train(config)
```

- running the training script using the torchrun on
multiple GPUs, e.g. running the `toy_train.py` on two GPUs
on a single machine. To run the `toy_train.py`, the wandb login
key needs to be set in the `toy_train.py`.
```shell
torchrun --standalone --nproc_per_node=2 toy_train.py --train_file ./dataset/train.json --dev_file ./dataset/validation.json --log_dir ./logs --batch_size 64
```