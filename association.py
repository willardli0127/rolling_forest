import numpy as np
import pandas as pd

from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules


def get_valid_features(df, min_support=1):
    valid_features =[]
    for feature in sorted(df.columns):
        cnts = df[feature].dropna()
        valid_cnt, valid_val = len(cnts), len(cnts.drop_duplicates())
        if (valid_cnt > min_support) and (valid_val > 1):
            valid_features.append(feature)
        
        print(feature, valid_cnt, valid_val)
    
    return valid_features


def rules_filter(rules_df, kept_antecedents=[], dropped_antecedents=[], kept_consequents=[], dropped_consequents=[], res_file='rules.csv'):
    kept_antecedents, dropped_antecedents, kept_consequents, dropped_consequents, res_index = frozenset(kept_antecedents), frozenset(dropped_antecedents), frozenset(kept_consequents), frozenset(dropped_consequents), []
    for index, antecedent, consequent in zip(rules_df.index, rules_df['antecedents'], rules_df['consequents']):
        if kept_antecedents.issubset(antecedent) and dropped_antecedents.isdisjoint(antecedent) and kept_consequents.issubset(consequent) and dropped_consequents.isdisjoint(consequent):
            res_index.append(index)
            
    valid_rules_df = rules_df[rules_df.index.isin(res_index)]
    if valid_rules_df.shape[0] > 0:
        valid_rules_df.sort_values(['lift', 'confidence', 'support'], ascending=False, inplace=True)
        valid_rules_df.to_csv(res_file, index=False)
        print(valid_rules_df)
    else:
        print(None)


drop_list = ['id', 'batch_id', 'timestamp', 'first_call_date', 'money', 'human', 'duration',
            #  '点击激活按钮', '激活成功', '激活被拒', '点击去借钱', '首贷一元', '首贷成功', '首贷被拒',
             'FAQ:上门', 'FAQ:什么保险', 'FAQ:什么快递', 'FAQ:会员卡（桔子）', 'FAQ:关于企业主贷', 'FAQ:小米与小米金融的关系', 'FAQ:忘记了', 'FAQ:投诉']


dir = '~/Downloads/Data/Telemarketing/'
data_csv = 'telemarketing_train_1209.csv'


df = pd.read_csv((dir + data_csv), na_values='None')
valid_fetures = get_valid_features(df, 228717)
df = df[valid_fetures][df.timestamp > 0]
df['first_call_date'] = pd.to_datetime(df['timestamp'].values, unit='ms')#.tz_localize('UTC').tz_convert('Asia/Shanghai')
df['first_call_date'] = df['first_call_date'].dt.date
df.fillna(0, inplace=True)

df.drop(columns=drop_list, inplace=True)

frequent_itemsets = fpgrowth(df, min_support=0.0001, use_colnames=True)
print(frequent_itemsets)
rules = association_rules(frequent_itemsets, min_threshold=0.1)
rules_filter(rules, kept_consequents=['首贷成功'])
