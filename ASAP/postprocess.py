import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.manifold import MDS
from scipy import stats
import re

class Evaluator():
    def __init__(self, config):
        self.df = pd.read_excel(config.df_path) #  columns [essay_id, prompt_id, essay, score, msg_system, msg_user_instruction, msg_assistant]
        self.score_range = {
            1: (2, 12),
            2: (1, 6),
            3: (0, 3),
            4: (0, 3),
            5: (0, 4),
            6: (0, 4),
            7: (0, 30),
            8: (0, 60),
        }
        self.save_path = config.save_path
        self.agg_mode = config.agg_mode
        self.clip_outliers = config.clip_outliers
        self.map_mode = config.map_mode
        self.model_name = config.model_name

    def parse_msg_assistant(self, msg_assistant):
        pass

    def get_traits(self):
        pass

    def check_score(self, score, prompt_id=None):
        if prompt_id == None:
            score_min = 0
            score_max = 10
        else:
            score_min = self.score_range[prompt_id][0]
            score_max = self.score_range[prompt_id][1]
        return score <= score_max and score >= score_min
    
    def aggregate_traits(self, mode):
        if mode == 'mean':
            def agg_fn(group):
                def temp(*traits):
                    return np.mean(traits)
                group['score_agg'] = group.apply(lambda x: temp(x.trait1, x.trait2, x.trait3, x.trait4), axis=1)
                return group
            
        grouped = self.df.groupby('prompt_id', group_keys=False)
        self.df = grouped.apply(agg_fn)
        return
    
    def map_score(self, mode):
        if mode == 'standard':
            def map_fn(group):
                prompt_id = group.name
                score_min, score_max = self.score_range[prompt_id][0], self.score_range[prompt_id][1]
                score_agg = group['score_agg']
                score_scaled_0_1 = (score_agg - 0) / (10-0)
                score_scaled_target = score_scaled_0_1 * (score_max - score_min) + score_min
                group['score_map'] = score_scaled_target
                group['score_pred'] = [np.nan if np.isnan(x) else np.round(x).astype(int) for x in score_scaled_target]
                return group
            
        elif mode == 'minmax':
            def map_fn(group):
                prompt_id = group.name
                score_min, score_max = self.score_range[prompt_id][0], self.score_range[prompt_id][1]
                score_agg = group['score_agg']

                # round outliers
                if self.clip_outliers == True:
                    q1 = score_agg.quantile(0.25)
                    q3 = score_agg.quantile(0.75)
                    iqr = q3 - q1
                    iqr_width = 1.5
                    score_agg = pd.Series(np.where(score_agg < (q1-iqr*iqr_width), q1-iqr*iqr_width, score_agg))
                    score_agg = pd.Series(np.where(score_agg > (q3+iqr*iqr_width), q3+iqr*iqr_width, score_agg))


                score_scaled_0_1 = (score_agg - score_agg.min()) / (score_agg.max() - score_agg.min())
                score_scaled_target = score_scaled_0_1 * (score_max - score_min) + score_min
                group['score_map'] = score_scaled_target
                group['score_pred'] = [np.nan if np.isnan(x) else np.round(x).astype(int) for x in score_scaled_target]
                return group
        
        grouped = self.df.groupby('prompt_id', group_keys=False)
        self.df = grouped.apply(map_fn)
        return

    def cal_qwk(self, y1_name, y2_name):
        # calculate QWK for each prompt
        df = self.df.copy(deep=True)
        N = len(df)
        df = df.dropna(subset=['score_pred'], axis=0)
        print("dropped {} instances due to NaN value in score_pred column.".format(N-len(df)))
        def temp(group):
            prompt_id = group.name
            score_min = self.score_range[prompt_id][0]
            score_max = self.score_range[prompt_id][1]
            y1 = group[y1_name]
            y2 = group[y2_name]
            qwk = cohen_kappa_score(y1, y2, weights='quadratic', labels=np.arange(score_min, score_max+1))
            return round(qwk, 3)
        grouped = df.groupby('prompt_id')
        qwks = grouped.apply(temp).to_dict()
        qwks['average'] = round(np.mean(list(qwks.values())), 3)
        return qwks
    
    def quadratic_weighted_kappa(self, y1, y2, prompt_id):
        s_min, s_max = self.score_range[prompt_id][0], self.score_range[prompt_id][1]
        qwk = cohen_kappa_score(y1, y2, weights='quadratic', labels=np.arange(s_min, s_max+1))
        return round(qwk, 3)
    
    def kendall_tau(self, y1, y2):
        res = stats.kendalltau(y1, y2)
        return round(res.statistic, 3)

    def cal_metrics(self, y1_name, y2_name):
        df = self.df.copy(deep=True)
        N = len(df)
        df = df.dropna(subset=['score_pred'], axis=0)
        print("dropped {} instances due to NaN value in score_pred column.".format(N-len(df)))

        data = []
        prompt_ids = [1,2,3,4,5,6,7,8]
        for pid in prompt_ids:
            y1 = df.loc[df['prompt_id'] == pid, y1_name]
            y2 = df.loc[df['prompt_id'] == pid, y2_name]
            #metrics
            qwk = self.quadratic_weighted_kappa(y1, y2, pid)
            kendall_tau = self.kendall_tau(y1, y2)
            data.append({
                'prompt_id': pid,
                'qwk': qwk,
                'kendall_tau': kendall_tau,
            })
        
        eval_report = pd.DataFrame(data=data)
        return eval_report

class EvaluatorVanilla(Evaluator):
    def __init__(self, config):
        Evaluator.__init__(self, config)

    def parse_msg_assistant(self, msg_assistant, prompt_id):
        # these patterns should be adjusted according to the generation results of LLMs.
        pattern_1 = r'Score:\n* *(\d+\.?\d*)/(\d+\.?\d*)|(\d+\.?\d*) out of (\d+\.?\d*)|a score of (\d+\.?\d*)/(\d+\.?\d*)'
        pattern_2 = r'Score:\n* *\d+\.?\d*|<score>\d+\.?\d*</score>'
        if re.search(pattern_1, msg_assistant):
            answer_spans = [float(x) for x in re.search(pattern_1, msg_assistant).groups() if re.match('\d+', str(x))]
            numerator, denominator = answer_spans[0], answer_spans[1]
            s_min, s_max = self.score_range[prompt_id][0], self.score_range[prompt_id][1]
            pred_score = np.round((numerator/denominator) * (s_max-s_min) + s_min).astype(int)
            if self.check_score(pred_score, prompt_id):
                return pred_score
            else:
                return np.nan
            
        elif re.search(pattern_2, msg_assistant):
            pred_score = np.round(float(re.search(r'\d+', re.search(pattern_2, msg_assistant).group(0)).group(0))).astype(int)
            if self.check_score(pred_score, prompt_id):
                return pred_score
            else:
                return np.nan

        return np.nan 
    
    def get_score_pred_column(self, column_name='score_pred'):
        self.df[column_name] = self.df.apply(lambda x: self.parse_msg_assistant(x['msg_assistant'], x['prompt_id']), axis=1)
        return
    
    def evaluate(self, save=False):
        self.get_score_pred_column(column_name='score_pred')
        if save:
            self.df.to_excel(self.save_path, index=False)
            print("result saved.")
        qwks = self.cal_qwk(y1_name='score', y2_name='score_pred')
        return qwks

class EvaluatorMTS(Evaluator):
    def __init__(self, config):
        Evaluator.__init__(self, config)

    def parse_msg_assistant(self, msg_assistant):
        # this pattern should be adjusted according to the generation result of LLMs.
        pattern = r'[sS]core: \d+\.?\d*|score of \d+\.?\d*/?\d*\.?\d*|\d+\.?\d* out of \d+\.?\d*|as a \d+\.?\d*|at \d+\.?\d*|Score: <score>\d+\.?\d*</score>'
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
        qwks = self.cal_qwk(y1_name='score', y2_name='score_pred')
        return qwks
    
    def evaluate(self, save=False):
        self.get_traits()
        self.aggregate_traits(mode=self.agg_mode)
        self.map_score(mode=self.map_mode)
        if save:
            self.df.to_excel(self.save_path, index=False)
            print("result saved.")
        qwks = self.cal_qwk(y1_name='score', y2_name='score_pred')
        return qwks

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, choices=['vanilla', 'mts'], help='method to use')
    parser.add_argument('--model_name', type=str, help='the name of LLM')
    parser.add_argument('--save', action='store_true', default=False, help='whether to save evaluation result.')
    
    config = parser.parse_args()
    config.df_path = f"./ASAP/resource/df_test_{config.model_name}_{config.method}.xlsx"
    config.save_path = f"./ASAP/resource/df_test_{config.model_name}_{config.method}.xlsx"
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