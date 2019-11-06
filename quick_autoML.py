import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score

def keep_relavent_columns(df, column_names=None):
    if column_names is None:
        return df
    return df[column_names]

def encode_one_hot(df):
    columnsToEncode = list(df.select_dtypes(include=['category','object']))
    for col in columnsToEncode:
        if len(df[col].unique()) < 50:
            df = pd.concat([df,pd.get_dummies(df[col], prefix=[col])], axis=1)
        df.drop(col,inplace=True,axis=1)
    return df

def normalize_data(df):
    columnsToEncode = list(df.select_dtypes(include=['float','int']))
    for col in columnsToEncode:
        df[col]=(df[col]-df[col].mean())/df[col].std()
    return df

def apply_to_numberic_selective(df):
    columnsToEncode = list(df.select_dtypes(include=['float','int']))
    for col in columnsToEncode:
        df = df.apply(pd.to_numeric, errors='coerce')
        
def process_column_names_xgboost(df):
    regex = re.compile(r"\[|\]|<", re.IGNORECASE)
    df.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in df.columns.values]
    return df

def preprocessing(df, target, column_names=None,bad_columns=None, apply_onehot=True, using_xgboost=True):
    # Avoids processing target feature
    target_df=df[target]
    df.drop(target, inplace=True, axis=1)
    
    df = keep_relavent_columns(df,column_names)
    df = drop_bad_columns(df, bad_columns)
    df = df.dropna()
    
    df = normalize_data(df)
    if apply_onehot:
        df = encode_one_hot(df)
        df = df.astype('float64')
        df = df.apply(pd.to_numeric, errors='coerce')
    else :
        df = apply_to_numberic_selective(df)
    if using_xgboost:
        df = process_column_names_xgboost(df)
    #reads the target feature after processing
    df_merged = df.merge(target_df, how='inner', left_index=True, right_index=True)
    return df_merged
    
def label_feature_split(df, column):
    label=df[[column]].values.ravel()
    feature=df.drop([column], axis=1)
    return feature, label

def split_dataset(df):
    train, test = train_test_split(df, test_size=0.2)
    train, validation = train_test_split(train, test_size=0.125)
    return train, validation, test

def drop_bad_columns(df, columns=None):
    if columns is not None:
        return df.drop(columns, axis=1)
    return df

#This can easily be extended for other metrics, especially for binary labels
def metrics(y_pred, y_test):
    from sklearn.metrics import balanced_accuracy_score, accuracy_score
    print('the accuracy is '+str(accuracy_score(y_pred, y_test)))
    print('the balanced accuracy is '+str(balanced_accuracy_score(y_pred, y_test)))   
    
def run_generic_models(X_train, y_train, X_test, y_test):
    #Using the recomended classifiers
    #https://arxiv.org/abs/1708.05070
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from catboost import CatBoostClassifier
    from xgboost import XGBClassifier
    import lightgbm as lgb
    GBC = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)
    RFC = RandomForestClassifier(n_estimators=500, max_features=0.25, criterion="entropy")
    SVM = SVC(C = 0.01, gamma=0.1, kernel="poly", degree=3, coef0=10.0)
    ETC = ExtraTreesClassifier(n_estimators=1000, max_features="log2", criterion="entropy")
    LR = LogisticRegression(C=1.5, penalty="l1",fit_intercept=True)
    # Models that were not included in the paper not from SKlearn
    XGC = XGBClassifier()
    CBC = CatBoostClassifier(silent=True)
    light_gb = lgb.LGBMClassifier()
    
    models=[(LR, "linear regression"),(ETC, "Extra tree classifier"),(SVM, "support vector classifier"), (RFC, "random forest classifier"), (GBC, "gradient boosted classifier"),
             (XGC, "XGBoost"),(light_gb,"Light GBM"), (CBC, "catboost classifier")]
    #models=[()]
    for model, name in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print('When using '+ name)
        metrics(y_pred,y_test)
    
df = pd.read_csv('Datasets/games.csv')
processed_features_df = preprocessing(df, 'winner', bad_columns='victory_status')

train_df, validation_df, test_df = split_dataset(processed_features_df)
X_train, y_train = label_feature_split(train_df,'winner')
X_validation, y_validation = label_feature_split(validation_df, 'winner')
X_test, y_test = label_feature_split(test_df, 'winner')

run_generic_models(X_train, y_train, X_validation, y_validation)
