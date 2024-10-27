import os, re
import numpy as np
import pandas as pd

url_replacer = '<url>'

def replace_url(text):
        replaced_text = re.sub('(http[s]?://)?((www)\.)?([a-zA-Z0-9]+)\.{1}((com)(\.(cn))?|(org))', url_replacer, text)
        return replaced_text

def reformat_essay(essay: str):
    essay = essay.strip()
    essay = re.sub(r'@(PERSON|ORGANIZATION|LOCATION|DATE|TIME|MONEY|PERCENT|MONTH|EMAIL|NUM|CAPS|DR|CITY|STATE)\d+', lambda x: f'{{{x.groups()[0]}}}', essay)
    essay = replace_url(essay)
    if "..." in essay:
        essay = re.sub(r'\.{3,}(\s+\.{3,})*', '...', essay)
        # print essay
    if "??" in essay:
        essay = re.sub(r'\?{2,}(\s+\?{2,})*', '?', essay)
        # print essay
    if "!!" in essay:
        essay = re.sub(r'\!{2,}(\s+\!{2,})*', '!', essay)
    return essay

if __name__ == '__main__':


    df = pd.read_excel('./ASAP/resource/training_set_rel3.xlsx')
    # the essay score for id 10534 is blank, and we get its corresponding score from training_set_rel3.tsv file
    df.loc[df['essay_id']==10534, 'domain1_score'] = 3

    # for prompt 7, the overall score specified in the "domain1_score" column is given incorrectly (see the description of essay set #7 in the original dataset for more details.)
    # in essence, the points for "Ideas" (i.e., "rater1_trait1" and "rater2_trait1" columns in "./ASAP/resource/training_set_rel3.xlsx") should be doubled according to their scoring rubric.
    # hence we correct 'domain1_score' for prompt 7 accordingly, as follows.
    df_p7 = df.loc[df['essay_set'] == 7]
    corrected_scores = (2*df_p7['rater1_trait1']+df_p7['rater1_trait2']+df_p7['rater1_trait3']+df_p7['rater1_trait4'])+(2*df_p7['rater2_trait1']+df_p7['rater2_trait2']+df_p7['rater2_trait3']+df_p7['rater2_trait4'])
    df.loc[df['essay_set']==7, 'domain1_score'] = corrected_scores

    processed_df = pd.DataFrame({
        'essay_id': df['essay_id'],
        'prompt_id': df['essay_set'],
        'essay': df['essay'].map(lambda x: reformat_essay(x)),
        'score': df['domain1_score'].astype(int),
    })

    processed_df.to_excel('./ASAP/resource/dataset.xlsx', index=False)