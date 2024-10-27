import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
import re
import argparse

class Evaluator():
    def __init__(self, config):
        self.df = pd.read_excel(config.df_path) #  columns [essay_id, prompt_id, essay, score, msg_system, msg_user_instruction, msg_assistant]
        self.save_path = config.save_path
        self.agg_mode = config.agg_mode
        self.clip_outliers = config.clip_outliers
        self.map_mode = config.map_mode
        self.score_min = 1
        self.score_max = 5
        print(len(self.df))

    def parse_msg_assistant(self, msg_assistant):
        pass

    def get_traits(self):
        pass

    def check_score(self, score):
        score_min = 0
        score_max = 10
        return score <= score_max and score >= score_min
    
    def aggregate_traits(self, mode):
        if mode == 'mean':
            def agg_fn(group):
                def temp(*traits):
                    return np.mean(traits)
                group['score_agg'] = group.apply(lambda x: temp(x.trait1, x.trait2, x.trait3, x.trait4), axis=1)
                return group
            
        elif mode == 'std_wsum': # weight for each trait is determined by (std_i / sigma_j{1,2,3,4} std_j)
            def agg_fn(group):
                std = {}
                for step_idx in [1,2,3,4]:
                    std[step_idx] = group[f'trait{step_idx}'].std()
                std_denominator = np.sum(list(std.values()))
                std_weights = {k: v / std_denominator for k, v in std.items()}
                std_wsum = std_weights[1]*group['trait1']+std_weights[2]*group['trait2']+std_weights[3]*group['trait3'] +std_weights[4]*group['trait4']
                group['score_agg'] = std_wsum
                return group
            
        grouped = self.df.groupby('prompt_id', group_keys=False)
        self.df = grouped.apply(agg_fn)
        return
    
    def map_proficiency_level(self, score):
        # low is for essays scoring between 1.0 and 2.0, medium is for 2.5 to 3.5, and high is for 4.0 to 5.0.
        # Essays with a combined score of 0 are invalid responses and were not included in TOEFL11.
        
        # thresholds = np.linspace(start=self.score_min, stop=self.score_max, num=4)
        thresholds = [1, 2.25, 3.75, 5]
        # thresholds = [0, 1.67, 3.34, 5]
        if score < thresholds[1]: 
            return 'low'
        elif score < thresholds[2]:
            return 'medium'
        else:
            return 'high'
    
    def map_score(self, mode):
        if mode == 'standard':
            def map_fn(group):
                score_min, score_max = self.score_min, self.score_max
                score_agg = group['score_agg']
                score_scaled_0_1 = (score_agg - 0) / (10-0)
                score_scaled_target = score_scaled_0_1 * (score_max - score_min) + score_min
                group['score_map'] = score_scaled_target
                group['proficiency_level_pred'] = [np.nan if np.isnan(x) else self.map_proficiency_level(x) for x in score_scaled_target]
                return group
            
        elif mode == 'minmax':
            def map_fn(group):
                score_min, score_max = self.score_min, self.score_max
                score_agg = group['score_agg']

                if self.clip_outliers == True:
                    # round outliers
                    q1 = score_agg.quantile(0.25)
                    q3 = score_agg.quantile(0.75)
                    iqr = q3 - q1
                    iqr_width = 1.5
                    score_agg = pd.Series(np.where(score_agg < (q1-iqr*iqr_width), q1-iqr*iqr_width, score_agg))
                    score_agg = pd.Series(np.where(score_agg > (q3+iqr*iqr_width), q3+iqr*iqr_width, score_agg))


                score_scaled_0_1 = (score_agg - score_agg.min()) / (score_agg.max() - score_agg.min())
                score_scaled_target = score_scaled_0_1 * (score_max - score_min) + score_min
                group['score_map'] = score_scaled_target
                group['proficiency_level_pred'] = [np.nan if np.isnan(x) else self.map_proficiency_level(x) for x in score_scaled_target]
                return group
        
        grouped = self.df.groupby('prompt_id', group_keys=False)
        self.df = grouped.apply(map_fn)
        return

    def cal_qwk(self, y1_name, y2_name):
        # calculate QWK for each prompt
        df = self.df.copy(deep=True)
        N = len(df)
        df = df.dropna(subset=['proficiency_level_pred'], axis=0)
        print("dropped {} instances due to NaN value in proficiency_level_pred column.".format(N-len(df)))
        def temp(group):
            y1 = group[y1_name]
            y2 = group[y2_name]
            qwk = cohen_kappa_score(y1, y2, weights='quadratic', labels=['low','medium','high'])
            return round(qwk, 3)
        grouped = df.groupby('prompt_id')
        qwks = grouped.apply(temp).to_dict()
        qwks['average'] = round(np.mean(list(qwks.values())), 3)
        return qwks

class EvaluatorVanilla(Evaluator):
    def __init__(self, config):
        Evaluator.__init__(self, config)

    def parse_msg_assistant(self, msg_assistant):
        pattern = r'(Score:|score of|as)[\n ]*\"? *([Ll]ow|[Mm]edium|[Hh]igh) *\"?|([Ll]ow|[Mm]edium|[Hh]igh) score'
        match = re.search(pattern, msg_assistant)
        if match is None:
            return np.nan
        else:
            score = re.search(r'[Ll]ow|[Mm]edium|[Hh]igh', match.group(0)).group(0).lower()
            return score
    
    def get_score_pred_column(self, column_name='proficiency_level_pred'):
        self.df[column_name] = self.df.apply(lambda x: self.parse_msg_assistant(x['msg_assistant']), axis=1)
        return
    
    def evaluate(self, save=False):
        self.get_score_pred_column(column_name='proficiency_level_pred')
        if save:
            self.df.to_excel(self.save_path, index=False)
            print("result saved.")
        qwks = self.cal_qwk(y1_name='proficiency_level', y2_name='proficiency_level_pred')
        return qwks

class EvaluatorMTS(Evaluator):
    def __init__(self, config):
        Evaluator.__init__(self, config)

    def parse_msg_assistant(self, msg_assistant):
        pattern = r'[sS]core: \d+\.?\d*|score of \d+\.?\d*/?\d*\.?\d*|\d+\.?\d* out of \d+\.?\d*|as a \d+\.?\d*|at \d+\.?\d*|Score: <score>\d+\.?\d*</score>|Score: <7>\d+\.?\d*</score>'
        match = re.search(pattern, str(msg_assistant))
        if match == None:
            return np.nan
        else:
            score = float(re.search(r'\d+\.?\d*', match.group(0)).group(0))
            if self.check_score(score):
                return score
            else:
                return np.nan
            
    def get_traits(self):
        data = [] 
        for i, instance in self.df.iterrows():
            datum = instance.to_dict()
            for step_idx in [1,2,3,4]:
                msg_assistant = instance[f'msg_assistant_score_{step_idx}']
                score = self.parse_msg_assistant(msg_assistant)
                datum.update({f'trait{step_idx}': score})
            data.append(datum)
        
        df = pd.DataFrame(data=data)
        self.df = df
        return
    
    def evaluate(self, save=False):
        self.get_traits()
        self.aggregate_traits(mode=self.agg_mode)
        self.map_score(mode=self.map_mode)
        if save:
            self.df.to_excel(self.save_path, index=False)
            print("result saved.")
        qwks = self.cal_qwk(y1_name='proficiency_level', y2_name='proficiency_level_pred')
        return qwks

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, choices=['vanilla', 'mts'], help='method to use')
    parser.add_argument('--model_name', type=str, help='the name of LLM')
    parser.add_argument('--save', action='store_true', default=False, help='whether to save evaluation result.')
    
    config = parser.parse_args()
    config.df_path = f"./TOEFL11/resource/df_test_{config.model_name}_{config.method}.xlsx"
    config.save_path = f"./TOEFL11/resource/df_test_{config.model_name}_{config.method}.xlsx"
    config.agg_mode = 'mean'
    config.map_mode = 'minmax'
    config.clip_outliers = True

    if config.method == 'vanilla':
        evaluator = EvaluatorVanilla(config)
    elif config.method == 'mts':
        evaluator= EvaluatorMTS(config)

    print(f'method: {config.method}    model: {config.model_name}')
    qwks = evaluator.evaluate(save=config.save)
    print(qwks)

if __name__ == '__main__':
    main()