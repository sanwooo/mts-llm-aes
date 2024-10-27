import numpy as np
import pandas as pd
import re
import os
import argparse

class DataProcessor():
    def __init__(self, args):
        self.dataset_path = args.dataset_path
        self.template_path = args.template_path
        self.dataset = pd.read_excel(args.dataset_path)
        self.template = pd.read_excel(args.template_path)
        self.df = self.dataset.merge(self.template, on='prompt_id')
        self.save_path = args.save_path

    def sub(self, pattern, repl, string):
        # escape backslash
        repl = str(repl).replace('\\', '\\\\')
        result = re.sub(pattern, repl, string)
        return result
    
    def save(self, df):
        # save dataframe
        df.to_excel(self.save_path, index=False)
        return

class DataProcessorVanilla(DataProcessor):
    def __init__(self, args):
        DataProcessor.__init__(self, args)
        
    def fill_msg_user_instruction_template(self, msg_user_instruction_template, prompt, essay):
        msg_user_instruction_template = self.sub('@prompt', prompt.strip(), msg_user_instruction_template)
        msg_user_instruction_template = self.sub('@essay', essay.strip(), msg_user_instruction_template)
        return msg_user_instruction_template
    
    def batch_fill_msg_user_instruction_template(self, df):
        df.loc[:, 'msg_user_instruction'] = df.apply(lambda x: self.fill_msg_user_instruction_template(x['msg_user_instruction_template'], x['prompt'], x['essay']), axis=1)
        return df
    
    def compose_test_dataset(self):
        df_test = self.df
        df_test = self.batch_fill_msg_user_instruction_template(df_test)
        self.df_test = df_test
        self.save(self.df_test)
        print("compose test dataset: success.")
        return

class DataProcessorMTS(DataProcessor):
    def __init__(self, args):
        DataProcessor.__init__(self, args)

    def fill_msg_system_template(self, msg_system_template, trait, trait_description):
        msg_system_template = self.sub('@trait', trait, msg_system_template)
        msg_system_template = self.sub('@description', trait_description, msg_system_template)
        return msg_system_template
    
    def fill_msg_user_retrieval_template(self, msg_user_retrieval_template, prompt, essay, trait):
        msg_user_retrieval_template = self.sub('@prompt', prompt, msg_user_retrieval_template)
        msg_user_retrieval_template = self.sub('@essay', essay, msg_user_retrieval_template)
        msg_user_retrieval_template = self.sub('@trait', trait, msg_user_retrieval_template)
        return msg_user_retrieval_template
    
    def fill_msg_user_score_template(self, msg_user_score_template, trait, trait_rubric):
        msg_user_score_template = self.sub('@trait', trait, msg_user_score_template)
        msg_user_score_template = self.sub('@rubric', trait_rubric, msg_user_score_template)
        return msg_user_score_template
    
    def compose_test_dataset(self):
        df_test = self.df

        data = []
        for i, row in df_test.iterrows():
            datum = row.to_dict()
            prompt, essay = row['prompt'], row['essay']
            msg_system_template = row['msg_system_template']
            msg_user_retrieval_template = row['msg_user_retrieval_template']
            msg_user_score_template  = row['msg_user_score_template']
            for trait_idx in [1,2,3,4]:
                trait, trait_description, trait_rubric = row[f'trait_{trait_idx}'], row[f'description_{trait_idx}'], row[f'rubric_{trait_idx}']
                msg_system = self.fill_msg_system_template(msg_system_template, trait, trait_description)
                msg_user_retrieval = self.fill_msg_user_retrieval_template(msg_user_retrieval_template, prompt, essay, trait)
                msg_user_score = self.fill_msg_user_score_template(msg_user_score_template, trait, trait_rubric)
                datum.update({
                    f'msg_system_{trait_idx}': msg_system,
                    f'msg_user_retrieval_{trait_idx}': msg_user_retrieval, 
                    f'msg_user_score_{trait_idx}': msg_user_score, 
                })
            data.append(datum)
        
        df_test = pd.DataFrame(data=data)
        self.df_test = df_test
        self.save(self.df_test)
        print("compose test dataset: success.")
        return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, choices=['vanilla', 'mts'], help='method to use')
    args = parser.parse_args()

    args.dataset_path = './TOEFL11/resource/dataset.xlsx'
    args.template_path = f'./TOEFL11/resource/template_{args.method}.xlsx'
    args.save_path = f'./TOEFL11/resource/df_test_{args.method}.xlsx'

    if args.method == 'vanilla':
        data_processor = DataProcessorVanilla(args)
    elif args.method == 'mts':
        data_processor = DataProcessorMTS(args)
    data_processor.compose_test_dataset()
    return

if __name__ == '__main__':
    main()
