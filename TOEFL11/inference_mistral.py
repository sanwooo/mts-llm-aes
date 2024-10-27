import re
import os
import argparse
import numpy as np
import pandas as pd
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, GenerationConfig, LlamaTokenizer

torch.manual_seed(42)


class Mistral():
    def __init__(self, config):
        self.df = config.df # columns : [essay_id, prompt_id, essay, score, msg_system, msg_user_instruction, msg_user_essay]
        self.model = config.model
        self.tokenizer = config.tokenizer
        self.generation_config = config.generation_config
        self.save_path = config.save_path
        self.batch_size = config.batch_size
    
    def request(self, list_of_prompt):
        N = len(list_of_prompt)
        list_of_completion = []
        for i in range(0, N, self.batch_size):
            prompt_batch = list_of_prompt[i:i+self.batch_size]
            tokenized_batch = self.tokenizer(
                prompt_batch,
                padding=True,
                truncation=True,
                return_tensors='pt',
            )
            input_ids = tokenized_batch['input_ids'].cuda()
            attention_mask = tokenized_batch['attention_mask'].cuda()
            result = self.model.generate(
                        input_ids = input_ids,
                        attention_mask = attention_mask,
                        generation_config = self.generation_config,
                    )
            decoded_batch = self.tokenizer.batch_decode(result[:, input_ids.shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            list_of_completion += decoded_batch
        
        return list_of_completion


    def sub(self, pattern, repl, string):
        # escape backslash
        repl = str(repl).replace('\\', '\\\\')
        result = string.replace(pattern, repl)
        # result = re.sub(pattern, repl, string)
        return result

    def inference(self):
        pass
    

class MistralVanilla(Mistral):
    def __init__(self, config):
        Mistral.__init__(self, config)

    def compose_prompt(self, msg_system, msg_user_instruction):
        messages = [
            {'role': 'user', 'content':msg_system + '\n' + msg_user_instruction}
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        return prompt
    
    def inference(self):
        groups = []
        for prompt_id, group in self.df.groupby('prompt_id'):
            print("-----------prompt {}------------".format(prompt_id))
            group.loc[:, 'prompt'] = group.apply(lambda x: self.compose_prompt(x['msg_system'], x['msg_user_instruction']), axis=1)
            list_of_msg_assistant = self.request(group['prompt'].tolist())
            group['msg_assistant'] = list_of_msg_assistant
            groups.append(group)
        
        result = pd.concat(groups, axis=0)
        result.to_excel(self.save_path, index=False)
        print("inference result saved successfully.")
        return
    
class MistralMTS(Mistral):
    def __init__(self, config):
        Mistral.__init__(self, config)
    
    def compose_prompt_retrieval(self, msg_system, msg_user_retrieval):
        messages = [
            {'role': 'user', 'content': msg_system + '\n' + msg_user_retrieval},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        return prompt
    
    def compose_prompt_score(self, msg_system, msg_user_retrieval, msg_assistant_retrieval, msg_user_score):
        messages = [
            {'role': 'user', 'content': msg_system + '\n' + msg_user_retrieval},
            {'role': 'assistant', 'content': msg_assistant_retrieval},
            {'role': 'user', 'content': msg_user_score},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        return prompt
    
    def inference(self):
        groups = []
        for prompt_id, group in self.df.groupby('prompt_id'):
            print("-----------prompt {}------------".format(prompt_id))
            for trait_idx in [1, 2, 3, 4]:
                print("-----------trait {}------------".format(trait_idx))
                group.loc[:, f'prompt_retrieval_{trait_idx}'] = group.apply(lambda x: self.compose_prompt_retrieval(x[f'msg_system_{trait_idx}'], x[f'msg_user_retrieval_{trait_idx}']), axis=1)
                group.loc[:, f'msg_assistant_retrieval_{trait_idx}'] = self.request(group[f'prompt_retrieval_{trait_idx}'].tolist())
                group.loc[:, f'prompt_score_{trait_idx}'] = group.apply(lambda x: self.compose_prompt_score(x[f'msg_system_{trait_idx}'], x[f'msg_user_retrieval_{trait_idx}'], x[f'msg_assistant_retrieval_{trait_idx}'], x[f'msg_user_score_{trait_idx}']), axis=1)
                group.loc[:, f'msg_assistant_score_{trait_idx}'] = self.request(group[f'prompt_score_{trait_idx}'].tolist())
            groups.append(group)
            
        result = pd.concat(groups, axis=0)
        result.to_excel(self.save_path, index=False)
        print("inference result saved successfully.")
        return
    
def main():
    parser = argparse.ArgumentParser(description="inference Mistral2")
    parser.add_argument('--method', type=str, choices=['vanilla', 'mts'], help='method to use')
    parser.add_argument('--model_path_prefix', type=str, default="", help='path to LLM.')
    parser.add_argument('--model_name', type=str, help='the name of LLM')
    parser.add_argument("--gpu_id", type=int, default=0, help='gpu id')
    parser.add_argument("--batch_size", type=int, default=8, help='batch size for inference.')
    args = parser.parse_args()
    print(args._get_kwargs())
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    dataset_path = f"./TOEFL11/resource/df_test_{args.method}.xlsx"

    class CONFIG:
        df = pd.read_excel(dataset_path)
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.model_path_prefix, args.model_name))
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(os.path.join(args.model_path_prefix, args.model_name), device_map='cuda', torch_dtype=torch.float16).eval()
        save_path = f"./TOEFL11/resource/df_test_{args.model_name}_{args.method}.xlsx"
        generation_config = GenerationConfig(
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.1,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            repetition_penalty=1.1,
        )
        batch_size = args.batch_size
    config = CONFIG()

    if args.method == 'vanilla':
        generator = MistralVanilla(config)
    elif args.method == 'mts':
        generator = MistralMTS(config)

    generator.inference()

    return

if __name__ == '__main__':
    main()