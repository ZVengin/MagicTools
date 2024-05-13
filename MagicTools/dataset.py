import json,os
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
class MagicDataset(Dataset):
  def __init__(self,
               dataset_path,
               tokenizer,
               format='csv', 
               construct_instance=lambda x,y:x,
               process_inputs = lambda x,y,z:x,
               is_train=True,
               is_chinese=False,
               use_cache=False,
               cache_dir=None,
               sample_weight=None
              ):
    """
    params:
      dataset_path: the path to the dataset file

      format: the format of dataset. Currently, 
              it supports csv, jsonl, json, and txt.
              
              jsonl: the records are saved in different 
              lines andeach line is a record. e.g. {'text':'I like it', 'label':'positive'}
              
              json: the records are saved as a list 
              and each element is a record. e.g. {'text':'I like it', 'label':'positive'}
              
              txt: the records are saved in different lines
              and each line is a record. The first line is 
              the head, e.g, 'text \t label \n'. From the second 
              line, each line is a record, e.g., 'I like it \t positive \n'

      construct_instance: the function takes a data record as inputs and return
              an instance for model inputs
    """
    self.construct_instance = construct_instance
    self.process_inputs = process_inputs
    self.tokenizer = tokenizer
    self.is_train = is_train
    self.is_chinese = is_chinese
    self.use_cache=use_cache
    self.cache_dir=cache_dir
    self.sample_weight=sample_weight
    if format=='csv':
      data = pd.read_csv(dataset_path)
      self.data = data.to_dict('records')
    elif format=='jsonl':
      with open(dataset_path,'r') as f:
        self.data = []
        for line in f:
          if line.strip():
            self.data.append(json.loads(line))
    elif format=='json':
      with open(dataset_path,'r') as f:
        self.data = json.load(f)
    elif format=='txt':
      with open(dataset_path,'r') as f:
        data = f.read().split('\n')
        heads = [h.strip() for h in data[0].split('\t')]
        self.data = [{k:v for k,v in zip(heads,line.split('\t'))}
                     for line in data[1:]]

    dataset_name = os.path.basename(dataset_path).split('.')[0]
    if self.use_cache and self.cache_dir != None and os.path.exists(os.path.join(self.cache_dir,dataset_name)):
        with open(os.path.join(self.cache_dir,dataset_name),'r') as f:
            self.data = json.load(f)
    else:
        self.data = self.construct_instance(self.data,self.tokenizer,self.is_train, self.is_chinese)
        if self.cache_dir != None:
            os.makedirs(self.cache_dir,exist_ok=True)
            with open(os.path.join(self.cache_dir,dataset_name),'w') as f:
                json.dump(self.data,f)
    if self.sample_weight is not None:
        self.weight = self.sample_weight(self.data)
    else:
        self.weight = None
    

  def __getitem__(self,idx):
    instance = self.data[idx]
    instance = self.process_inputs( self.tokenizer, instance, self.is_train)
    return instance

  def __len__(self):
    return len(self.data)


def get_dataloader(
    dataset_file,
    format,
    tokenizer,
    construct_instance,
    process_inputs,
    is_train,
    use_cache,
    cache_dir,
    batch_size,
    collate_fn,
    num_workers,
    sample_weight=None,
    distributed=False
):
    dataset = MagicDataset(
        dataset_file, 
        format=format, 
        tokenizer = tokenizer,
        construct_instance=construct_instance,
        process_inputs = process_inputs,
        is_train=is_train,
        use_cache=use_cache,
        cache_dir=cache_dir,
        sample_weight=sample_weight
    )
    if distributed:
        sampler = DistributedSampler(dataset)
    else:
        sampler = None
    dataloader=DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle = True if sampler is None else False,
        num_workers=num_workers,
        sampler = sampler,
        pin_memory =True
    )
    return dataloader