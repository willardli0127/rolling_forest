import numpy as np
import pandas as pd

from dateutil.relativedelta import relativedelta
from econml.metalearners import TLearner
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
# from xgboost import XGBRegressor, XGBClassifier


def checkedout_ongoing_split(df, latest_date, updating_len):
    completed_date = latest_date - pd.to_timedelta(updating_len, 'D')
    checkedout_df, ongoing_df = df[df.first_call_date <= completed_date], df[df.first_call_date > completed_date]

    return checkedout_df, ongoing_df


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


def robot_artificial_stats(df, converted_query):
    converted = df.query(converted_query)
    robot_df, artificial_df = df[df.human == 0], df[df.human == 1]
    converted_robot, converted_artificial = converted[converted.human == 0], converted[converted.human == 1]
    
    print("num: %d, converted: %d (%.6f)" 
          % (df.shape[0], converted.shape[0], converted.shape[0]/df.shape[0]))
    if (robot_df.shape[0] != 0) & (converted_robot.shape[0] != 0):
        print("robot: %d (%.6f)" 
              % (converted_robot.shape[0], converted_robot.shape[0]/robot_df.shape[0]))
    if (artificial_df.shape[0] != 0) & (converted_artificial.shape[0] != 0):
        print("artificial: %d (%.6f)" 
              % (converted_artificial.shape[0], converted_artificial.shape[0]/artificial_df.shape[0]))


def checkedout_ongoing_stats(df, latest_date, updating_len):
    checkedout_df, general_df = checkedout_ongoing_split(df, latest_date, updating_len)
    if checkedout_df.shape[0] > 0:
        print("checked out: ", end='')
        robot_artificial_stats(checkedout_df, converted_query)

    if general_df.shape[0] > 0:
        print("ongoing: ", end='')
        robot_artificial_stats(general_df, converted_query)


def postive_converted_stats(df, postive_query, converted_query):
    postive = df.query(postive_query)
    converted = df.query(converted_query)
    postive_converted = postive.query(converted_query)

    print("total: %d, potive checks: %d, converted: %d, converted_postive: %d;"
          % (df.shape[0], postive.shape[0], converted.shape[0], postive_converted.shape[0]))
    print("postive rate: %.6f, conversion rate: %.6f, converted_postive rate: %.6f;" 
          % (postive.shape[0]/df.shape[0], converted.shape[0]/df.shape[0], postive_converted.shape[0]/df.shape[0]))
    print("converted_postive/postive: %.6f, converted_postive/converted: %.6f."
          % (postive_converted.shape[0]/postive.shape[0], postive_converted.shape[0]/converted.shape[0]))


def baseline_eval(df, dispatched_len, postive_query, converted_query):
    df.sort_values(['duration', 'message', 'appointment', 'is_not_silence'], ascending=False, inplace=True)
    dispatched_eval(df, dispatched_len, postive_query, converted_query)


def repair_cols_diff(pred_df, features_list):
    pred_df = pred_df[features_list.intersection(pred_df.columns)]
    pred_df[features_list.difference(pred_df.columns)] = 0
    
    pred_df.columns = features_list

    return pred_df


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


class RFT:
    def __init__(self, model, df, completed_start, completed_end, updating_start, updating_end, datetime_name, target_name, treatment_name, treatment_set, features_list=None, drop_list=None, n_estimator=1):
        self.completed_start, self.completed_end, self.updating_start, self.updating_end, self.datetime_name, self.target_name, self.treatment_name, self.treatment_set, self.n_estimator = completed_start, completed_end, updating_start, updating_end, datetime_name, target_name, treatment_name, treatment_set, n_estimator
        self._model = model
        self.target_effect = target_prep_list(self.target_name, 'effect')
        
        self.features = None
        self.model = dict.fromkeys(treatment_set, None)
        for treat in self.treatment_set:
            self.model[treat] = RF(self._model, df[df[self.treatment_name] == treat], self.completed_start, self.completed_end, self.updating_start, self.updating_end, self.datetime_name, self.target_name, features_list, drop_list, self.n_estimator)
            
            if self.features is None:
                self.features = self.model[treat].features
    
    
    def update(self, df, completed_end, updating_start, updating_end):
        self.completed_end, self.updating_start, self.updating_end = completed_end, updating_start, updating_end
        
        for treat in self.treatment_set:
            self.model[treat].update(df[df[self.treatment_name]== treat])

    
    def predict(self, df):
        for treat in self.treatment_set:
            if self._model == 'classifier':
                df['treatment_hat_' + str(treat)] = self.model[treat].model.predict_proba(df[self.features].values)[:, 1]
            elif self._model == 'regressor':
                df['treatment_hat_' + str(treat)] = self.model[treat].model.predict(df[self.features].values)

        return df


def stats_process(df, latest_date, updating_len, dispatching_dict, general_query, check_query, interest_list):
    for batch, dispatching_rate in list(dispatching_dict.items()):
        batch_df = df[df.batch_id == batch]
        if (len(batch_df['首贷成功'].unique()) < 2):
            dispatching_dict.pop(batch)
            continue

        print(batch)
        print("total: %d, dispatching rate: %.6f" % (batch_df.shape[0], dispatching_rate), end=', ')

        dispatched_batch_robot, dispatched_batch_artificial = batch_df[batch_df.human == 0], batch_df[batch_df.human == 1]
        print("robot: %d, artificial: %d" % (dispatched_batch_robot.shape[0], dispatched_batch_artificial.shape[0]))
    
        general_batch = batch_df.query(general_query)
        if general_batch.shape[0] != 0:
            print("general:")
            checkedout_ongoing_stats(general_batch, latest_date, updating_len)

        checks_batch = batch_df.query(check_query)
        if checks_batch.shape[0] != 0:
            print("checks:")
            checkedout_ongoing_stats(checks_batch, latest_date, updating_len)

        for check in interest_list:
            check_batch = checks_batch[checks_batch[check] == 1]
            if check_batch.shape[0] == 0:
                continue
        
            print(check + ': ')
            checkedout_ongoing_stats(check_batch, latest_date, updating_len)

        print()


def eval_process(model, df, X, yhat_name, feature_importance):
    df[yhat_name] = model.predict_proba(X.values)[:, 1]
    df.sort_values(feature_importance, ascending=False, inplace=True)
    
    return df


def daily_process(df, dispatching_dict, postive_query, converted_query, drop_list, pred_df=None):
    for batch, dispatching_rate in list(dispatching_dict.items()):
        batch_df = df[df.batch_id == batch].fillna(0)
        dispatched_batch_len = int(batch_df.shape[0] * dispatching_rate)
        if dispatched_batch_len == 0:
            continue

        postive_batch = batch_df.query(postive_query)
        print("%s: total: %d, postive: %d, dispatching rate: %.4f, dispatched: %d" 
              % (batch, batch_df.shape[0], postive_batch.shape[0], dispatching_rate, dispatched_batch_len))

        baseline_eval(batch_df, dispatched_batch_len, postive_query, converted_query)

    rfc = RF('classifier', batch_df, pd.to_datetime('2021-02-27'), pd.to_datetime('2021-03-02'), pd.to_datetime('2021-03-03'), pd.to_datetime('2021-03-12'), 'first_call_date', ['首贷成功'], drop_list=drop_list, n_estimator=10)
    # batch_df_ = batch_df
    batch_df = rfc.pred(batch_df)
    dispatched_eval(batch_df, dispatched_batch_len, postive_query, converted_query)
    batch_df.to_csv('1.csv', index=False)
    rfc.rfi.to_csv('2.csv', index=False)

    rfct = RFT('classifier', batch_df, pd.to_datetime('2021-02-27'), pd.to_datetime('2021-03-02'), pd.to_datetime('2021-03-03'), pd.to_datetime('2021-03-12'), 'first_call_date', ['首贷成功'], treatment_name='human', treatment_set=set(batch_df['human']), features_list=rfc.features, n_estimator=10)
    batch_df = rfct.predict(batch_df)
    batch_df['effect'] = batch_df['treatment_hat_1'] - batch_df['treatment_hat_0']
    print(rfct)
    batch_df.sort_values(['effect', 'treatment_hat_1'], ascending=False, inplace=True)
    dispatched_eval(batch_df, dispatched_batch_len, postive_query, converted_query)
    batch_df.to_csv('3.csv', index=False)

    pred_df= batch_df[batch_df.first_call_date == pd.to_datetime('2021-03-13')]
    pred_df = rfct.predict(pred_df)
    pred_df.sort_values(['effect', 'treatment_hat_1'] 
                        #+ rfct.rfct[1].rfi['feature'].to_list()
                        , ascending=False, inplace=True)
    pred_df.to_csv('4.csv', index=False)


def dispatched_eval(df, dispatched_df_len, postive_query, converted_query):
    df = df.iloc[:dispatched_df_len]
    converted_df = df.query(converted_query)
    if converted_df.shape[0] == 0:
        return
    
    postive_df = df.query(postive_query)
    postive_converted_df = converted_df.query(postive_query)

    print("postive: %d, converted: %d, postive_converted: %d;" 
          % (postive_df.shape[0], converted_df.shape[0], postive_converted_df.shape[0]))
    print("postive rate: %.6f, conversion rate: %.6f" 
          % (postive_df.shape[0]/dispatched_df_len, converted_df.shape[0]/dispatched_df_len), end=', ')

    if postive_df.shape[0] == 0:
        print("postive_converted/postive: %.6f" % (0), end=', ')
    else:
        print("postive_converted/postive: %.6f" % (postive_converted_df.shape[0]/postive_df.shape[0]), end=', ')

    print("postive_converted/converted: %.6f." % (postive_converted_df.shape[0]/converted_df.shape[0]))


dir = '~/Downloads/Data/Telemarketing/'
# data_csv = 'telemarketing_train_1209.csv'
data_csv = 'train_data_(2021-03-18).csv'
pred_csv = 'input_features_(2020-12-16).csv'

updating_len = 10
default_dispatching_rate = 0.05

general_query = "(点击激活按钮 == 0) and (激活成功 == 0) and (点击去借钱 == 0) and (首贷一元 == 0) and (激活被拒 == 0) and (首贷被拒 == 0)"
check_query = "(点击激活按钮 == 1) or (激活成功 == 1) or (点击去借钱 == 1) or (首贷一元 == 1) or (激活被拒 == 1) or (首贷被拒 == 1)"
postive_query = "(点击激活按钮 == 1) or (激活成功 == 1) or (点击去借钱 == 1) or (首贷一元 == 1)"
converted_query = "首贷成功 == 1"

interest_list = ['点击激活按钮', '激活成功', '激活被拒', '点击去借钱', '首贷一元', '首贷被拒']
drop_list = ['id', 'batch_id', 'timestamp', 'first_call_date', 'money', 'human',
             '点击激活按钮', '激活成功', '激活被拒', '点击去借钱', '首贷一元', '首贷成功', '首贷被拒',
             ]#'FAQ:上门', 'FAQ:什么保险', 'FAQ:什么快递', 'FAQ:会员卡（桔子）', 'FAQ:关于企业主贷', 'FAQ:小米与小米金融的关系', 'FAQ:忘记了', 'FAQ:投诉']

df = pd.read_csv((dir + data_csv), na_values='None')
# valid_fetures, na_features = get_valid_features(df, 228717, 0)
# df = df[valid_fetures][df.timestamp > 0]

df = df[(df['9月老版'] == 1) & (df['batch_id'] == '金条电销_未_20210207_非中产_ivr')]
valid_fetures, na_features = get_valid_features(df)

df['first_call_date'] = pd.to_datetime(df['timestamp'].values, unit='ms')#.tz_localize('UTC').tz_convert('Asia/Shanghai')
df['first_call_date'] = df['first_call_date'].dt.date
df.fillna(0, inplace=True)

earliest_date, latest_date = df['first_call_date'].min(), df['first_call_date'].max()
print(earliest_date, latest_date)
print(valid_fetures)

dispatching_dict = dict.fromkeys(df['batch_id'].unique().tolist(), default_dispatching_rate)
# dispatching_dict['金条电销_新名单白领_1_15+'] = 0.083
# dispatching_dict['金条电销_新名单蓝领_1_15+'] = 0.072
# dispatching_dict['金条电销_测试3'] = 0.06
dispatching_dict['金条电销_未_20210207_非中产_ivr'] = 0.026

stats_process(df, latest_date, updating_len, dispatching_dict, general_query, check_query, interest_list)
postive_converted_stats(df, postive_query, converted_query)
print()

# pred_df = pd.read_csv(dir + pred_csv)
# pred_df.fillna(0, inplace=True)
daily_process(df, dispatching_dict, postive_query, converted_query, drop_list)#, pred_df)

