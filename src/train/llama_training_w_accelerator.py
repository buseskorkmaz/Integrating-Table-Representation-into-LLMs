import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../'))
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    DataCollatorForSeq2Seq,
)
from accelerate import Accelerator
import json
import wandb
from torch.utils.data import DataLoader
import torch
from functools import partial
from tqdm.auto import tqdm
from collections import deque
import numpy as np
from typing import Callable, Union, Any
from flatten_dict import flatten, unflatten
import argparse

file_path = os.path.dirname(os.path.realpath(__file__))

def map_pytree(f: Callable[[Union[np.ndarray, torch.Tensor]], Any], 
               item: Any):
    if isinstance(item, dict):
        return {k: map_pytree(f, v) for k, v in item.items()}
    elif isinstance(item, list) or isinstance(item, set) or isinstance(item, tuple):
        return [map_pytree(f, v) for v in item]
    elif isinstance(item, np.ndarray) or isinstance(item, torch.Tensor):
        return f(item)
    else:
        return item

def to(item: Any, device: torch.device):
    return map_pytree(lambda x: torch.tensor(x).to(device), item)


class DistributeCombineLogs:
    count_tag = '__count__'

    def __init__(self, accelerator, use_wandb=False):
        self.totals = {}
        self.accelerator = accelerator
        self.use_wandb = use_wandb

    def convert_key(self, k):
        return (self.count_tag,) + k

    def key_is_count(self, k):
        return k[0] == self.count_tag

    def log(self, log_label, **additional_items):
        self.accelerator.wait_for_everyone()
        # Adding the log label to the gathered logs
        total_logs = self.gather_logs(log_label=log_label, **additional_items)
        if self.accelerator.is_main_process:
            if self.use_wandb:
                wandb.log({log_label: total_logs})
            print({log_label: total_logs})
        self.accelerator.wait_for_everyone()
        return total_logs

    def gather_logs(self, log_label, **additional_items):
        str_totals = {json.dumps(list(k)): v for k, v in self.totals.items()}
        combined_totals = self.accelerator.gather(str_totals)
        combined_totals = {tuple(json.loads(k)): v.sum().item() for k, v in combined_totals.items()}
        final_logs = {}
        for k, v in combined_totals.items():
            if not self.key_is_count(k):
                if combined_totals[self.convert_key(k)] == 0:
                    final_logs[k] = v * float('inf')
                else:
                    final_logs[k] = v / combined_totals[self.convert_key(k)]
        final_logs = unflatten(final_logs)
        final_logs = {**final_logs, **additional_items}
        return final_logs

    def accum_logs(self, logs):
        logs = flatten(logs)
        for k, v in logs.items():
            if isinstance(v, tuple) and len(v) == 2:
                item, n = v
            else:
                item = v
                n = 1  # Default count value

            new_item = torch.tensor([item]).float().to(self.accelerator.device)
            count_item = torch.tensor([n]).float().to(self.accelerator.device)

            if k in self.totals:
                self.totals[k] += new_item * count_item
                self.totals[self.convert_key(k)] += count_item
            else:
                self.totals[k] = new_item * count_item
                self.totals[self.convert_key(k)] = count_item

    def reset_logs(self):
        self.totals = {}


def label_logs(logs, label):
    return {label: logs}


def main(model_name):

    accelerator = Accelerator()
    # Print out key configuration properties
    print("Device:", accelerator.device)
    print("Distributed Type:", accelerator.distributed_type)
    print("Local process index:", accelerator.local_process_index)
    print("Number of processes:", accelerator.num_processes)
    print("Is main process:", accelerator.is_main_process)
    print("Is local main process:", accelerator.is_local_main_process)
    print("Use FP16:", accelerator.use_fp16)

    train_cfg = {
        "save_checkpoint_dir": os.path.join(file_path, f'../../models/{model_name}-table-1e-6_new_tokens'),
        "use_wandb" : "true",
        "wandb_project" : "table_understanding",
        "dataloader_workers" : 1,
        "bsize" : 1,
        "eval_bsize": 1,
        "lr": 1e-6,
        "grad_accum_steps": 1,
        "log_every": 128,
        "save_every": 4096,
        "eval_every": 512,
        "eval_batches": 10,
        "loss": {},
        "max_checkpoints": 1,
        "epochs": 5,
    }

    if not os.path.exists(train_cfg['save_checkpoint_dir']):
        try:
            os.makedirs(train_cfg['save_checkpoint_dir'])
        except:
            print("Couldn't create the checkpoint dir, probably it already exists")
            pass
    with open(os.path.join(train_cfg['save_checkpoint_dir'], 'config.json'), 'w') as f:
        json.dump(train_cfg, f)

    if train_cfg['use_wandb']:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            wandb.init(project=train_cfg['wandb_project'], config=train_cfg)
        accelerator.wait_for_everyone()

    # Load dataset from disk
    dataset_path = os.path.join(file_path, '../../data/llama_compliant_hf_train_large_spec_tokens')
    table_dataset = load_from_disk(dataset_path)
    dataset_eval = table_dataset.select(range(30000, 32000))
    dataset_train = table_dataset.select(range(20000))

    # Load tokenizer and model
    model_id = f"meta-llama/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    special_tokens_dict = {'additional_special_tokens': ['<R>','<C>','<CAP>', '[EMPTY]', '[BOLD]']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.resize_token_embeddings(len(tokenizer))

    data_collator = DataCollatorForSeq2Seq(
        tokenizer, 
        model=model, 
        label_pad_token_id=tokenizer.pad_token_id, 
        pad_to_multiple_of=2
    )

    train_data_loader_kwargs = {'num_workers': train_cfg['dataloader_workers'], 
                                'batch_size': train_cfg['bsize'], 
                                'collate_fn': data_collator,
                                'shuffle': False}
    eval_data_loader_kwargs = {'num_workers': train_cfg['dataloader_workers'], 
                                'batch_size': train_cfg['eval_bsize'], 
                                'collate_fn': data_collator,
                                'shuffle': False}

    data_loader = DataLoader(dataset_train, **train_data_loader_kwargs)
    eval_data_loader = DataLoader(dataset_eval, **eval_data_loader_kwargs)
    model.to(accelerator.device)
    model.train()
    model = accelerator.prepare(model)

    if hasattr(model, 'param_groups'):
        params = [{'params': frozenset().union(*list(map(lambda x: x.parameters(), p))), **f(train_cfg)} for p, f in model.param_groups]
    else:
        params = model.parameters()

    optim = torch.optim.AdamW(params, lr=train_cfg['lr'])
    optim, data_loader, eval_data_loader = accelerator.prepare(optim, data_loader, eval_data_loader)

    train_logs = DistributeCombineLogs(accelerator, use_wandb=train_cfg['use_wandb'])
    eval_logs = DistributeCombineLogs(accelerator, use_wandb=train_cfg['use_wandb'])
    step = 0
    best_loss = float('inf')
    saved_checkpoints = deque([])
    for epoch in tqdm(range(train_cfg['epochs']), disable=not accelerator.is_local_main_process):
        for items in tqdm(data_loader, disable=not accelerator.is_local_main_process):
            items = to(items, accelerator.device)
            input_ids = items["input_ids"]
            attention_mask = items["attention_mask"]
            labels = items["labels"]

            # Forward pass
            outputs = accelerator.unwrap_model(model)(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            # print("Training loss: ", loss, ", epoch", epoch)
            logs = {'loss': loss.item(), 'iteration': step, 'epoch': epoch}
            accelerator.backward(loss / train_cfg['grad_accum_steps'])
            train_logs.accum_logs(logs)
            if (step + 1) % train_cfg['grad_accum_steps'] == 0:
                optim.step()
                optim.zero_grad()
            if (step + 1) % train_cfg['log_every'] == 0:
                train_logs.log('train', iteration=step, epoch=epoch)
            if (step + 1) % train_cfg['grad_accum_steps'] == 0:
                train_logs.reset_logs()
            if (step + 1) % train_cfg['eval_every'] == 0:
                model.eval()
                eval_logs.reset_logs()
                with torch.no_grad():
                    for i, eval_items in enumerate(eval_data_loader):
                        eval_items = to(eval_items, accelerator.device)
                        if i >= train_cfg['eval_batches']:
                            break
                        input_ids = eval_items["input_ids"]
                        attention_mask = eval_items["attention_mask"]
                        labels = eval_items["labels"]

                        outputs = accelerator.unwrap_model(model)(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                        loss = outputs.loss
                        logs = {'loss': loss.item(), 'iteration': step, 'epoch': epoch}
                        # print("Eval loss: ", loss, ", epoch: ", epoch, " iteration: ", step)
                        eval_logs.accum_logs(logs)
                eval_label = 'eval'
                eval_total_logs = eval_logs.log('eval', iteration=step, epoch=epoch)
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    if eval_total_logs['loss'] < best_loss:
                        print('new best eval loss! Saving ...')
                        if not os.path.exists(train_cfg['save_checkpoint_dir']):
                            os.makedirs(train_cfg['save_checkpoint_dir'])
                        torch.save(accelerator.unwrap_model(model).state_dict(),
                                    os.path.join(train_cfg['save_checkpoint_dir'], 'model.pkl'))
                        torch.save(optim.state_dict(), os.path.join(train_cfg['save_checkpoint_dir'], 'optim.pkl'))
                        print('saved.')
                        best_loss = eval_total_logs['loss']
                accelerator.wait_for_everyone()
                model.train()
            if train_cfg['save_every'] is not None and (step + 1) % train_cfg['save_every'] == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    print('saving checkpoint...')
                    if not os.path.exists(train_cfg['save_checkpoint_dir']):
                        os.makedirs(train_cfg['save_checkpoint_dir'])
                    if (train_cfg['max_checkpoints'] is not None) and (len(saved_checkpoints) >= train_cfg['max_checkpoints']):
                        os.system('rm -rf %s' % (saved_checkpoints.popleft()))
                    torch.save(accelerator.unwrap_model(model).state_dict(),
                                os.path.join(train_cfg['save_checkpoint_dir'], 'model_%d.pkl' % (step)))
                    saved_checkpoints.append(os.path.join(train_cfg['save_checkpoint_dir'], 'model_%d.pkl' % (step)))
                    print('saved.')
                accelerator.wait_for_everyone()
            step += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training of llama2 model')
    parser.add_argument('--model_name', type=str, help='Model name: Huggingface id of llama2 family')
    args = parser.parse_args()
    main(args.model_name)
