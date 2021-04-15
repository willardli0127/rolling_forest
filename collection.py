import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dateutil.relativedelta import relativedelta
from econml.metalearners import TLearner
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import auc, mean_absolute_error, r2_score


class RF:
    def __init__(self, model, df, completed_start, completed_end, updating_start, updating_end, datetime_name, target_name, features_list=None, drop_list=None, n_estimator=1):
        self.completed_start, self.completed_end, self.updating_start, self.updating_end, self.datetime_name, self.target_name, self.n_estimator = completed_start, completed_end, updating_start, updating_end, datetime_name, target_name, n_estimator
        self._model = model
        self.target_hat = target_prep_list(self.target_name, 'hat')
        if features_list is not None:
            X, y = X_y_t_split(df[(df[self.datetime_name] >= self.completed_start) & (df[self.datetime_name] <= self.completed_end)], self.target_name, features_list=features_list)
        elif drop_list is not None:
            X, y = X_y_t_split(df[(df[self.datetime_name] >= self.completed_start) & (df[self.datetime_name] <= self.completed_end)], self.target_name, drop_list=drop_list)
        
        self.features = X.columns
        if self._model == 'classifier':
            self.model = RandomForestClassifier(n_estimators=((self.completed_end-self.completed_start).days+1)*self.n_estimator, random_state=0, n_jobs=-1, warm_start=True, oob_score=True).fit(X.values, y.values)
        elif self._model == 'regressor':
            self.model = RandomForestRegressor(n_estimators=((self.completed_end-self.completed_start).days+1)*self.n_estimator, random_state=0, n_jobs=-1, warm_start=True, oob_score=True).fit(X.values, y.values)
#         print(X.shape, self.model.oob_score_, self.model.n_estimators, self.model.estimators_)
#         self._feature_importance()
    
        self.model.n_estimators += ((self.updating_end - self.updating_start).days + 1) * self.n_estimator
        X, y = X_y_t_split(df[(df[self.datetime_name] >= self.updating_start) & (df[self.datetime_name] <= self.updating_end)], self.target_name, features_list=self.features)
        self.model.fit(X.values, y.values)
#         print(X.shape, self.model.oob_score_, self.model.n_estimators, self.model.estimators_)
        self._feature_importance()


    def update(self, df, completed_end, updating_start, updating_end):
        self.completed_end, self.updating_start, self.updating_end = completed_end, updating_start, updating_end
        
        updating_len = (self.updating_end - self.updating_start).days + 1
        self.model.estimators_ = self.model.estimators_[:-updating_len*self.n_estimator]
        self.model.n_estimators -= updating_len * self.n_estimator
#         print(self.model.oob_score_, self.model.n_estimators, self.model.estimators_)
        self.model.n_estimators += self.n_estimator

        X, y = X_y_t_split(df[df[self.datetime_name] == self.completed_end], self.target_name, features_list=self.features)    
        self.model.fit(X.values, y.values)
#         print(X.shape, self.model.oob_score_, self.model.n_estimators, self.model.estimators_)
#         self._feature_importance()

        self.model.n_estimators += updating_len * self.n_estimator
        X, y = X_y_t_split(df[(df[self.datetime_name] >= self.updating_start) & (df[self.datetime_name] <= self.updating_end)], self.target_name, features_list=self.features)
        self.model.fit(X.values, y.values)
#         print(X.shape, self.model.oob_score_, self.model.n_estimators, self.model.estimators_)
        self._feature_importance()


    def pred(self, df):
        if self._model == 'classifier':
            yhat = self.model.predict_proba(df[self.features].values)[:, 1]
        elif self._model == 'regressor':
            yhat = self.model.predict(df[self.features].values)
        
        target_len = len(self.target_name)
        if target_len == 1:
            df[self.target_hat] = yhat
        elif target_len > 1:
            for (i, target_hat) in enumerate(self.target_hat):
                df[target_hat] = yhat[:, i]           
            
        return df.sort_values(self.ordered_keys, ascending=False)


    def _feature_importance(self):
        self.rfi = pd.DataFrame({'feature': self.features, 'importance': self.model.feature_importances_})
        self.rfi = self.rfi[self.rfi.importance > 0].sort_values('importance', ascending=False)
        self.ordered_keys = self.target_hat + self.rfi['feature'].to_list()
        # print(self.rfi)


class TRF:
    def __init__(self, model, df, completed_start, completed_end, updating_start, updating_end, datetime_name, target_name, treatment_name, treatment_set, features_list=None, drop_list=None, n_estimator=1):
        self.completed_start, self.completed_end, self.updating_start, self.updating_end, self.datetime_name, self.target_name, self.treatment_name, self.treatment_set, self.n_estimator = completed_start, completed_end, updating_start, updating_end, datetime_name, target_name, treatment_name, treatment_set, n_estimator
        self._model = model
        self.target_effect = target_prep_list(self.target_name, 'effect')

        if features_list is not None:
            X, y, t = X_y_t_split(self._complete_treatment(df[(df[self.datetime_name] >= self.completed_start) & (df[self.datetime_name] <= self.completed_end)]), self.target_name, features_list=features_list, treatment_name=self.treatment_name)
        elif drop_list is not None:
            X, y, t = X_y_t_split(self._complete_treatment(df[(df[self.datetime_name] >= self.completed_start) & (df[self.datetime_name] <= self.completed_end)]), self.target_name, drop_list=drop_list, treatment_name=self.treatment_name)
        self.features = X.columns

        if self._model == 'classifier':
            self.model = TLearner(models=RandomForestClassifier(n_estimators=((self.completed_end-self.completed_start).days+1)*self.n_estimator, random_state=0, n_jobs=-1, warm_start=True, oob_score=True)).fit(y.values, t.values, X.values)
        elif self._model == 'regressor':
            self.model = TLearner(models=RandomForestRegressor(n_estimators=((self.completed_end-self.completed_start).days+1)*self.n_estimator, random_state=0, n_jobs=-1, warm_start=True, oob_score=True)).fit(y.values, t.values, X.values)

        for model in self.model.models:
            model.n_estimators += ((self.updating_end - self.updating_start).days + 1) * self.n_estimator

        X, y, t = X_y_t_split(self._complete_treatment(df[(df[self.datetime_name] >= self.updating_start) & (df[self.datetime_name] <= self.updating_end)]), self.target_name, features_list=self.features, treatment_name=self.treatment_name)
        self.model.fit(y.values, t.values, X.values)
    
    
    def update(self, df, completed_end, updating_start, updating_end):
        self.completed_end, self.updating_start, self.updating_end = completed_end, updating_start, updating_end
        updating_len = (self.updating_end - self.updating_start).days + 1
        
        for model in self.model.models:
            model.estimators_ = model.estimators_[:-updating_len*self.n_estimator]
            model.n_estimators -= updating_len * self.n_estimator
            model.n_estimators += self.n_estimator

        X, y, t = X_y_t_split(self._complete_treatment(df[df[self.datetime_name] == self.completed_end]), self.target_name, features_list=self.features, treatment_name=self.treatment_name)    
        self.model.fit(y.values, t.values, X.values)

        for model in self.model.models:
            model.n_estimators += updating_len * self.n_estimator
            
        X, y, t= X_y_t_split(self._complete_treatment(df[(df[self.datetime_name] >= self.updating_start) & (df[self.datetime_name] <= self.updating_end)]), self.target_name, features_list=self.features, treatment_name=self.treatment_name)
        self.model.fit(y.values, t.values, X.values)
        
    
    def effect(self, df):
        if self._model == 'classifier':
            yeffect = self.model.effect(df[self.features].values)[:, 1]
        elif self._model == 'regressor':
            yeffect = self.model.effect(df[self.features].values)

        target_len = len(self.target_name)
        if target_len == 1:
            df[self.target_effect] = yeffect
        elif target_len > 1:
            for (i, target_effect) in enumerate(self.target_effect):
                df[target_effect] = yeffect[:, i]
            
        return df.sort_values(self.target_effect, ascending=False)
    
    
    def _complete_treatment(self, df):
        treatment_diff = self.treatment_set.difference(set(df[self.treatment_name]))
        if len(treatment_diff) > 0:
            for treatment in treatment_diff:
                blank_row = pd.DataFrame(np.zeros((1, len(df.columns))), columns=df.columns)
                blank_row[self.treatment_name] = treatment
                df = df.append(blank_row)
        
        return df


def get_valid_features(df, min_support=1, detail=0):
    valid_features, na_features, features = [], [], {}
    for feature in sorted(df.columns):
        if df[feature].hasnans == True:
            na_features.append(feature)
                    
        valid_cnt, valid_val = df[feature].dropna().shape[0], df[feature].drop_duplicates().sort_values()
        valid_len, kvs = len(valid_val), {}
        
        print(feature, valid_cnt, valid_len, end=' ')
        if (valid_cnt > min_support) and (valid_len > 1):
            valid_features.append(feature)
            
            if detail == 0:
                print()
            elif detail == 1:
                for key in valid_val:
                    if np.isnan(key):
                        kvs[key] = df[df[feature].isna() == True].shape[0]
                    else: 
                        kvs[key] = df[df[feature] == key].shape[0]
                
                features[feature] = kvs
                print(kvs)
        else:
            print()
            
    print(na_features)

    return valid_features, na_features


def X_y_t_split(df, target_name, features_list=None, drop_list=None, treatment_name=None):
    if features_list is not None:
        X = df[features_list]
    elif drop_list is not None:
        X = df.drop(columns=drop_list)
    
    y = df[target_name]
    
    if treatment_name is None:
        return X, y
    elif treatment_name is not None:
        return X, y, df[treatment_name]


def target_prep_list(target_list, suffix):
    if len(target_list) == 1:
        return [target_list[0] + '_' + suffix]
    else:
        target_hat_list = []
        for target in target_list:
            target_hat_list.append(target + '_' + suffix)
    
        return target_hat_list


def dispatch_topk(df, p, k):
    df = df.head(k)
    df['dispatching_date'] = p.date()
    
    return df


def paym_preprocess(paym_csv):
    paym_df = pd.read_csv(paym_csv, 
                          usecols=[
                              'case_id',
                              'repaid_date', 
                              'repaid_money'
                          ],
                          parse_dates=['repaid_date']).groupby(['case_id', 'repaid_date']).sum().reset_index()
    print(paym_df.shape)

    return paym_df


def comm_preprocess(comm_csv):
    comm_df = pd.read_csv(comm_csv, parse_dates=['call_day']).drop(columns='subgroup')
    print(comm_df.shape)
 
    return comm_df


def df_preprocess(case_csv, paym_csv, acc_features):
    df = pd.read_csv(case_csv,
                     usecols=[
                         'id',
                         'current_batch',
                         'sex',
                         'role_id',
                         'user_name',
                         'create_start_time',
                         'overdue_stage',
                         'principal',
                         'return_time',
                         'settle',
                         'hive_dt'
                     ],
                     parse_dates=['create_start_time', 'return_time', 'hive_dt'])
    df = df[(df.create_start_time >= '2020-11-01') & (df.current_batch.isna() == False)]
    
    df['batch'] = df['current_batch'].str.extract('[B](\d)')[0].astype('int')
    df = df[df.batch < 3]

    df['month_batch'] = df['current_batch'].str[5: 7].astype('int')

    df['artificial'] = 0
    df['artificial'][(df.role_id >= 50) & (df.role_id <= 70)] = 1
    
    df['user_name'].fillna('robot', inplace=True)
    print(df.shape)
 
    df = df.merge(paym_preprocess(paym_csv), how='left', left_on=['id', 'hive_dt'], right_on=['case_id', 'repaid_date']).drop(columns=['case_id'])
    print(df.shape)

    df = df.merge(comm_preprocess(comm_csv), how='left', left_on=['id', 'hive_dt'], right_on=['case_id', 'call_day']).drop(columns=['case_id', 'call_day']).sort_values('hive_dt')
    print(df.shape)

    df['month'] = df['hive_dt'].dt.month
    
    df['repaid_money'].fillna(0, inplace=True)
    df['month_payment'] = df.groupby(['id', 'month'])['repaid_money'].cumsum()
    
    df_group = df.groupby(['id'])
    df['total_payment'] = df_group['repaid_money'].cumsum()
    df['payment_cumsum'] = df['total_payment'] - df['repaid_money']

    df['last_valid_defalter_call_date'] = None
    df['last_valid_defalter_call_date'][(df.manual_self_valid_connect_times > 0) | (df.robot_self_valid_connect_times > 0)] = df['hive_dt'][(df.manual_self_valid_connect_times > 0) | (df.robot_self_valid_connect_times > 0)]
    
    df.fillna(df_group.ffill(), inplace=True)

    df['last_valid_defalter_call_date'] = df_group['last_valid_defalter_call_date'].shift(1)
    df['repaid_date'] = df_group['repaid_date'].shift(1)

    df['overdue_day'] = df['return_time'].dt.day
    df['record_day'] = df['hive_dt'].dt.day

    df['overdue_days'] = (df['hive_dt'] - df['return_time']).dt.days

    df['last_valid_defalter_call_days'] = None
    df['last_valid_defalter_call_days'][df.last_valid_defalter_call_date != -1] = (df['hive_dt'][df.last_valid_defalter_call_date != -1] - df['last_valid_defalter_call_date'][df.last_valid_defalter_call_date != -1]).dt.days
    df['last_valid_defalter_call_days'].fillna(-1, inplace=True)

    df['last_payment_days'] = None
    df['last_payment_days'][df.repaid_date != -1] = (df['hive_dt'][df.repaid_date != -1] - df['repaid_date'][df.repaid_date != -1]).dt.days
    df['last_payment_days'].fillna(-1, inplace=True)

    df['overdue_day_diff'] = abs(df['record_day'] - df['overdue_day'])
    df['overdue_day_diff'][df.overdue_day_diff > (31 - df.overdue_day_diff)] = 31 - df['overdue_day_diff']

    for feature in acc_features:
        df[feature] = df_group[feature].shift(1).fillna(0).astype('int')
        df[feature + '_acc'] = df_group[feature].cumsum().astype('int')

    print(df.shape)

    return df


dir = '~/Downloads/Data/Collection/83/'
case_csv = dir + 'case_2011_2101.csv'
paym_csv = dir + 'paym_2011_2101.csv'
comm_csv = dir + 'comm_2011_2101.csv'

features = [
    'sex',
    'principal',
#     'robot_self_call_times',
#     'manual_self_call_times',
    'robot_self_valid_connect_times',
    'manual_self_valid_connect_times',
    'robot_self_valid_connect_length',
    'manual_self_valid_connect_length',
#     'robot_other_call_times',
#     'manual_other_call_times',
    'robot_other_valid_connect_times',
    'manual_other_valid_connect_times',
    'robot_other_valid_connect_length',
    'manual_other_valid_connect_length',
#     'connect_times',
    'self_helper_day_count',
    'self_silence_day_count',
    'self_hang_day_count',
    'self_voice_mail_day_count',
    'not_self_day_count',
    'other_helper_day_count',
    'other_silence_day_count',
    'other_hang_day_count',
    'other_voice_mail_day_count',
    'promise_day_count',
    'delay_day_count',
    'reduction_day_count',
    'need_human_day_count',
    'wait_salary_day_count',
    'no_money_day_count',
    'joint_debt_day_count',
    'has_plan_day_count',
    'cant_pay_day_count',
    'refute_day_count',
    'payment_cumsum',
    'overdue_day',
    'record_day',
    'overdue_days',
    'last_valid_defalter_call_days',
    'last_payment_days',
    'overdue_day_diff',
#     'robot_self_call_times_acc',
#     'manual_self_call_times_acc',
    'robot_self_valid_connect_times_acc',
    'manual_self_valid_connect_times_acc',
    'robot_self_valid_connect_length_acc',
    'manual_self_valid_connect_length_acc',
#     'robot_other_call_times_acc',
#     'manual_other_call_times_acc',
    'robot_other_valid_connect_times_acc',
    'manual_other_valid_connect_times_acc',
    'robot_other_valid_connect_length_acc',
    'manual_other_valid_connect_length_acc',
#     'connect_times_acc',
    'self_helper_day_count_acc',
    'self_silence_day_count_acc',
    'self_hang_day_count_acc',
    'self_voice_mail_day_count_acc',
    'not_self_day_count_acc',
    'other_helper_day_count_acc',
    'other_silence_day_count_acc',
    'other_hang_day_count_acc',
    'other_voice_mail_day_count_acc',
    'promise_day_count_acc',
    'delay_day_count_acc',
    'reduction_day_count_acc',
    'need_human_day_count_acc',
    'wait_salary_day_count_acc',
    'no_money_day_count_acc',
    'joint_debt_day_count_acc',
    'has_plan_day_count_acc',
    'cant_pay_day_count_acc',
    'refute_day_count_acc'
]

target = ['total_payment', 'month_payment', 'repaid_money']

groups_2012 = {
    0: ['2020-11-B1+', '2020-11-B1+WEAK', '2020-11-B2+', '2020-11-B2+WEAK'],
    1: ['2020-12-B1+', '2020-12-B1+WEAK'],
    2: ['2020-12-B2+', '2020-12-B2+WEAK']
}

groups_2101 = {
    0: ['2020-11-B1+', '2020-11-B1+WEAK', '2020-11-B2+', '2020-11-B2+WEAK', '2020-12-B1+', '2020-12-B1+WEAK', '2020-12-B2+', '2020-12-B2+WEAK'],
    1: ['2021-01-B1+', '2021-01-B1+WEAK'],
    2: ['2021-01-B2+', '2021-01-B2+WEAK']
}


df = df_preprocess(case_csv, paym_csv, features[2:34-5])
valid_features, na_features = get_valid_features(df)

target = ['total_payment', 'month_payment', 'repaid_money']
target_hat = target_prep_list(target, 'hat')
target_effect = target_prep_list(target, 'effect')

treatment_set = set(df['artificial'])

# rfr, rfi = dict.fromkeys([1, 2], None), dict.fromkeys([1, 2], None)
rfr_model = dict.fromkeys([1, 2], None)
trfr_model = dict.fromkeys([1, 2], None)

df[target_hat] = [0] * len(target)
df['dispatching_date'], df['dispatching_rank'], df['batch_rank'], df['daily_rank'] = None, None, None, None, 

p = pd.to_datetime('2020-12-01')
while p <= pd.to_datetime('2021-01-31'):
    date_df = df[df.hive_dt == p]
    print(p, date_df.shape[0])
    for batch in [1, 2]:
        batch_df = date_df[date_df.batch == batch]
        print(batch, end=' ')
        
        if p.day == 1:
            completed_start, completed_end, updating_start, updating_end = p - relativedelta(months=1), p - pd.to_timedelta(8, 'D'), p - pd.to_timedelta(7, 'D'), p - pd.to_timedelta(1, 'D')
            rfr_model[batch] = RF('regressor', df[df.batch == batch], completed_start, completed_end, updating_start, updating_end, 'hive_dt', target, features)
            trfr_model[batch] = TRF('regressor', df[df.batch == batch], completed_start, completed_end, updating_start, updating_end, 'hive_dt', target, 'artificial', treatment_set, features)

        else:
            rfr_model[batch].update(df[df.batch == batch], completed_end, updating_start, updating_end)
            trfr_model[batch].update(df[df.batch == batch], completed_end, updating_start, updating_end)
        
        batch_df = rfr_model[batch].pred(batch_df)
        batch_df = trfr_model[batch].effect(batch_df)
        
        for (target_name, target_hat_name) in zip(target, target_hat):
            print(mean_absolute_error(batch_df[target_name], batch_df[target_hat_name]), r2_score(batch_df[target_name], batch_df[target_hat_name]), end=' ')
        
        print()
        
        batch_df['batch_rank'] = batch_df.groupby('current_batch').cumcount()   
        batch_df['daily_rank'] = batch_df.groupby('batch').cumcount()
        batch_df['dispatching_rank'][batch_df.dispatching_date.isna() == True] = batch_df[batch_df.dispatching_date.isna() == True].groupby('batch').cumcount()
        
        batch_df = batch_df[target_hat + ['batch_rank', 'daily_rank', 'dispatching_rank']]

        df.update(batch_df)
        
        
    date_df = df[(df.hive_dt == p) & (df.dispatching_date.isna() == True) & (df.overdue_stage <= 1)]
    
    if p.month == 12:
        groups = groups_2012
    elif p.month == 1:
        groups = groups_2101
        
    for group in groups.items():
        if group[0] == 0:
            dispatching_df = date_df[date_df.current_batch.isin(group[1])].sort_values(target_hat, ascending=False)
            if p.day == 1:
                dispatching_df = dispatch_topk(dispatching_df, p, 25*400)
            elif p.day > 1:
                dispatching_df = dispatch_topk(dispatching_df, p, 25*13)
            
            print(group, dispatching_df.shape[0])
            
        elif (group[0] == 1) and (p.day >= 10):
            dispatching_df = date_df[date_df.current_batch.isin(group[1])].sort_values('dispatching_rank')
            if p.day == 10:
                dispatching_df = dispatch_topk(dispatching_df, p, 25*150)
            elif p.day > 10:
                dispatching_df = dispatch_topk(dispatching_df, p, 25*18)
                
            print(group, dispatching_df.shape[0])
        
        elif (group[0] == 2) and (p.day >= 10):
            dispatching_df = date_df[date_df.current_batch.isin(group[1])].sort_values('dispatching_rank')
            if p.day == 10:
                dispatching_df = dispatch_topk(dispatching_df, p, 17*150)
            elif p.day > 10:
                dispatching_df = dispatch_topk(dispatching_df, p, 17*20)
            
            print(group, dispatching_df.shape[0])
        
        df.update(dispatching_df)
    
    df['dispatching_date'][(df.hive_dt >= p) & (df.hive_dt.dt.month == p.month)] = df[(df.hive_dt >= p) & (df.hive_dt.dt.month == p.month)].groupby('id')['dispatching_date'].ffill()
    df['dispatching_rank'][(df.hive_dt >= p) & (df.hive_dt.dt.month == p.month) & (df.dispatching_date.isna() == False)] = df[(df.hive_dt >= p) & (df.hive_dt.dt.month == p.month) & (df.dispatching_date.isna() == False)].groupby('id')['dispatching_rank'].ffill()
    
    completed_end, updating_start, updating_end = completed_end + pd.to_timedelta(1, 'D'), updating_start + pd.to_timedelta(1, 'D'), updating_end + pd.to_timedelta(1, 'D')
    p = p + pd.to_timedelta(1, 'D')
    
df.to_csv('df.csv', index=False)


# df = pd.read_csv('df.csv', parse_dates=['create_start_time', 'return_time', 'hive_dt', 'last_valid_defalter_call_date', 'dispatching_date'])
# df_group = df[df.dispatching_date.isna() == False].groupby(['id', 'month'])
# df['month_manual_self_valid_connect_times'] = df_group['manual_self_valid_connect_times'].cumsum().astype('int')
# df['month_manual_other_valid_connect_times'] = df_group['manual_other_valid_connect_times'].cumsum().astype('int')
# df['dispathced_valid_connect_times'] = df['month_manual_self_valid_connect_times'] + df['month_manual_other_valid_connect_times']
# df['dispathced_valid_connect_payment'] = df_group['repaid_money'].cumsum()

# p = pd.to_datetime('2020-12-01')
# while p <= pd.to_datetime('2021-01-31'):
#     date_df = df[df.hive_dt <= p]
#     print(p, date_df.shape[0])
    
#     if p == pd.to_datetime('2020-12-01'):
#         groups = groups_2012       
#     elif p == pd.to_datetime('2021-01-01'):
#         groups = groups_2101
    
#     if p.day == 1:
#         group_cumsum = {}
#         for item in groups.items():
#             group_cumsum[item[0]] = [0, 0, 0]#, 0, 0]
        
#     for group in groups.items():
#         group_df = date_df[(date_df.current_batch.isin(group[1]) == True) & (date_df.month == p.month)].groupby(['id', 'month']).last().sort_values(['dispatching_rank', 'dispatching_date']).reset_index()
        
#         sum_dispatched_date = group_df['repaid_money'][group_df.dispatching_date.isna() == False].sum()
#         group_cumsum[group[0]][0] = group_cumsum[group[0]][0] + sum_dispatched_date
        
#         sum_robot_date = group_df['repaid_money'][group_df.dispatching_date.isna() == True].sum()
#         group_cumsum[group[0]][1] = group_cumsum[group[0]][1] + sum_robot_date
        
#         sum_dispatched_connected_date = group_df['repaid_money'][group_df.dispathced_valid_connect_times > 0].sum()
#         group_cumsum[group[0]][2] = group_cumsum[group[0]][2] + sum_dispatched_connected_date

#         # sum_dispatched_uplift = group_df['repaid_money'][(group_df.dispatching_date.isna() == False) & ((group_df.role_id == 80) | (group_df.role_id == -1))].sum()
#         # group_cumsum[group[0]][3] = group_cumsum[group[0]][3] + sum_dispatched_uplift
        
#         # sum_robot_uplift = group_df['repaid_money'][(group_df.dispatching_date.isna() == True) & ((group_df.role_id >= 50) & (group_df.role_id <= 70))].sum()
#         # group_cumsum[group[0]][4] = group_cumsum[group[0]][4] + sum_robot_uplift
                        
#         print(group[0], group_df.shape[0], group_df[group_df.dispatching_date == p].shape[0], group_df[group_df.dispatching_date.isna() == False].shape[0], sum_dispatched_date, group_cumsum[group[0]][0], sum_robot_date, group_cumsum[group[0]][1], sum_dispatched_connected_date, group_cumsum[group[0]][2])#, 
#         #sum_dispatched_uplift, group_cumsum[group[0]][3], sum_robot_uplift, group_cumsum[group[0]][4], group_cumsum[group[0]][3]- group_cumsum[group[0]][4])
        
#         if p.day == 31:
#             y = group_df['month_payment'].cumsum().values
#             y_len = len(y)
#             x = np.arange(y_len)
#             plt.plot(x, y)
#             plt.plot([len(group_df['id'][(group_df.month == p.month) & (group_df.dispatching_date.isna() == False)].unique())] * y_len, y)
#             plt.plot(x, np.linspace(0, y[-1], y_len))
#             y_ = np.gradient(y)
#             plt.plot(x, y_)
#             plt.show()
            
#             print(auc(np.linspace(0, 1, y_len), (y / y[-1])))
        
#     p = p + pd.to_timedelta(1, 'D')