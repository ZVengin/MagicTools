import logging
import argparse
from collections import namedtuple


logging.basicConfig(level=logging.DEBUG,format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)



param = namedtuple('param',['name','type','default','help','action','choices'])

def AcceleratorPrint(accelerator_print_func,message,file):
    accelerator_print_func(message,file=file)
    file.flush()

def Param(name,type=None,default=None,help=None,action=None, choices=None):
  return param(name=name,type=type,default=default,help=help,action=action, choices=choices)

def MagicParse(paras,vals=None):
  parser = argparse.ArgumentParser()
  for para in paras:
    assert para.name is not None or para.name.strip()!='', 'The parameter name is empty string or None type'
    if para.name.startswith('--'):
      if para.action is not None:
        parser.add_argument(para.name,
                            action=para.action,
                            help=para.help)
      elif para.choices is not None:
        parser.add_argument(para.name,
                            choices=para.choices,
                            type=para.type,
                            default=para.default,
                            help=para.help)
      else:
        assert para.default is not None, f'Default value for {para.name} is missing.'
        assert para.type is not None, f'Type value for {para.type} is missing.'

        parser.add_argument(para.name,
                        type=para.type,
                        default=para.default,
                        help=para.help)
    else:
      if para.choices is not None:
        parser.add_argument(para.name,
                            choices=para.choices,
                            type=para.type,
                            default=para.default,
                            help=para.help)
      else:
        assert para.type is not None, f'Type value for {para.type} is missing.'
        parser.add_argument(para.name,
                            type=para.type,
                            help=para.help)
  args = parser.parse_args(args=vals)
  return args

def GetLoss(model,tokenizer,batch):
  prompts = []
  prompt_template='Write a story following the given prompt. prompt:{} story:{}'
  for sample in batch:
    prompt = prompt_template.format(sample['prompt'],sample['story'])
    prompts.append(prompt)
  model_inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors='pt')
  model_inputs  = {k:v.to(model.decoder.embed_tokens.weight.get_device()) 
                    for k,v in model_inputs.items()}
  model_inputs['labels'] = model_inputs['input_ids'].to(model.lm_head.weight.get_device())
  model_outputs = model(**model_inputs)
  return model_outputs.loss


def PadSequence(seqs, max_len, pad_token_id):
    pad_seqs = []

    for seq in seqs:
        pad_seq = seq + [pad_token_id] * (max_len - len(seq))
        pad_seqs.append(pad_seq)
    return pad_seqs