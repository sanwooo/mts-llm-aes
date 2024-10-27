import numpy as np
import pandas as pd
import os
import re

def read_essay(essay_path, dir_path= './TOEFL11/resource/ETS_Corpus_of_Non-Native_Written_English/data/text/responses/original'):
    path = os.path.join(dir_path, essay_path)
    with open(path, 'r') as f:
        essay = f.read().strip()
    return essay

if __name__ == '__main__':
    
    df = pd.read_csv('./TOEFL11/resource/ETS_Corpus_of_Non-Native_Written_English/data/text/index-test.csv', header=None, names=['path', 'prompt_id', 'proficiency_level'])
    df['essay'] = df['path'].map(lambda x: read_essay(x))
    df['prompt_id'] = df['prompt_id'].map(lambda x: int(re.sub('P(\d)', lambda x: x.groups()[0], x)))
    df = df.rename_axis('essay_id').reset_index()

    df_language = pd.read_csv('./TOEFL11/resource/ETS_Corpus_of_Non-Native_Written_English/data/text/index.csv')
    df = df.merge(df_language[['Filename','Language']], how='inner', left_on='path', right_on='Filename')
    df.drop(['Filename'], axis=1, inplace=True)
    df.rename({'Language': 'language'}, axis=1, inplace=True)

    df.to_excel('./TOEFL11/resource/dataset.xlsx', index=False)


    # add prompt column to template_vanilla.xlsx and template_mts.xlsx 
    prompts = []
    for prompt_id in [1, 2, 3, 4, 5, 6, 7, 8]:
        with open(os.path.join('./TOEFL11/resource/ETS_Corpus_of_Non-Native_Written_English/data/text/prompts', f'P{prompt_id}.txt'), 'r') as f:
            prompt = f.read().strip()
        prompts.append(prompt)

    template_vanilla = pd.read_excel('./TOEFL11/resource/template_vanilla.xlsx')
    template_vanilla['prompt'] = prompts
    template_mts = pd.read_excel('./TOEFL11/resource/template_mts.xlsx')
    template_mts['prompt'] = prompts

    template_vanilla.to_excel('./TOEFL11/resource/template_vanilla.xlsx')
    template_mts.to_excel('./TOEFL11/resource/template_mts.xlsx')