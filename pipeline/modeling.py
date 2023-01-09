""" Outcomes Notes:
RQ1 - final result (classification) - predict negative outcomes
RQ2 - final course score (regression)
"""

import logging
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, r2_score, mean_squared_error, \
    confusion_matrix, roc_curve, RocCurveDisplay
import shap
from shap import Explanation

_LOG = logging.getLogger(__name__)


def get_x_y(df, rq):

    drop_ivs = ['code_presentation', 'id_student', 'final_result', 'date_registration', 'date_unregistration',
                'id_assessment', 'date_submitted', 'score', 'dv', 'score']

    drop_VLE_ivs = [ 'n_days_dataplus', 'n_days_dualpane', 'n_days_externalquiz', 'n_days_folder', 'n_days_forumng',
                'n_days_glossary', 'n_days_homepage', 'n_days_htmlactivity', 'n_days_oucollaborate',
                'n_days_oucontent', 'n_days_ouelluminate', 'n_days_ouwiki', 'n_days_page', 'n_days_questionnaire',
                'n_days_quiz', 'n_days_repeatactivity', 'n_days_resource', 'n_days_sharedsubpage', 'n_days_subpage',
                'n_days_url', 'avg_sum_clicks_dataplus', 'avg_sum_clicks_dualpane', 'avg_sum_clicks_externalquiz',
                'avg_sum_clicks_folder', 'avg_sum_clicks_forumng', 'avg_sum_clicks_glossary', 'avg_sum_clicks_homepage',
                'avg_sum_clicks_htmlactivity', 'avg_sum_clicks_oucollaborate', 'avg_sum_clicks_oucontent',
                'avg_sum_clicks_ouelluminate', 'avg_sum_clicks_ouwiki', 'avg_sum_clicks_page',
                'avg_sum_clicks_questionnaire', 'avg_sum_clicks_quiz', 'avg_sum_clicks_repeatactivity',
                'avg_sum_clicks_resource', 'avg_sum_clicks_sharedsubpage', 'avg_sum_clicks_subpage',
                'avg_sum_clicks_url']

    drop_ivs += drop_VLE_ivs

    if rq == 1:
        # withdraw = 3, fail = 1, pass = 2, distinction = 0
        df['dv'] = np.where((df['final_result'] == 3) | (df['final_result'] == 1), 1, 0)
        y = df['dv']

    if rq == 2:
        y = df['score']

    cols = [x for x in df.columns if x not in drop_ivs]

    X = df[cols]

    return X, y


def grid_search(X, y, rq):

    grid = {
        'n_estimators': [400, 500, 600],
        'max_features': ['sqrt', 'log2'],
        'max_depth': [3, 5, 7],
        'min_samples_split': [6, 8, 10, 12],
        'bootstrap': [True, False],
        'random_state': [18]
    }

    if rq == 1:
        metric = 'roc_auc'  # 'accuracy', 'f1'
        est = RandomForestClassifier()

    if rq == 2:
        est = RandomForestRegressor()
        metric = 'r2'  # 'neg_mean_squared_error'

    gs_rf = GridSearchCV(estimator=est,
                         param_grid=grid,
                         cv=5,
                         n_jobs=-1,
                         refit=True,
                         verbose=3,
                         scoring=metric)
    gs_rf.fit(X, y)
    params = gs_rf.best_params_

    return params


def run_model(df, rq, outcome):

    base_wd = os.path.normpath(os.getcwd())

    X, y = get_x_y(df, rq)

    # grid search
    params = grid_search(X, y, rq)
    print(f'optimal parameters: {params}')

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=101)

    # train model with optimal parameters
    if rq == 1:
        mod = RandomForestClassifier(n_estimators=list(params.values())[4],
                                     bootstrap=list(params.values())[0],
                                     max_features=list(params.values())[2],
                                     max_depth=list(params.values())[1],
                                     min_samples_split=list(params.values())[3],
                                     random_state=18)

    if rq == 2:
        mod = RandomForestRegressor(n_estimators=list(params.values())[4],
                                    bootstrap=list(params.values())[0],
                                    max_features=list(params.values())[2],
                                    max_depth=list(params.values())[1],
                                    min_samples_split=list(params.values())[3],
                                    random_state=18)

    mod.fit(X_train, y_train)
    y_pred = mod.predict(X_test)

    # Save to dataframe
    y_df = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred}).join(X_test)
    y_df.to_csv(base_wd + f"\\outputs\\models\\rq{rq}_y_predict.csv")

    results = {}

    # save feature importances
    feats = {}
    for feature, importance in zip(X_train.columns, mod.feature_importances_):
        feats[feature] = importance
    results['importances'] = feats

    if rq == 1:
        results['f1_score'] = f1_score(y_test, y_pred)  # binary targets only
        results['roc_auc'] = roc_auc_score(y_test, y_pred)

        results['confusion_matrix'] = confusion_matrix(y_test, y_pred)
        # NOTE - cm[0][0] = TP, cm[1][1] = TN, cm[0][1] = FP, cm[1][0] = FN

    if rq == 2:
        results['mse'] = mean_squared_error(y_test, y_pred)  # binary targets only
        results['r2'] = r2_score(y_test, y_pred)

        # plt.clf()
        # ax = y_df.plot(kind="scatter", x="total_n_days", y="y_test", color="b", label="real score")
        # y_df.plot(kind="scatter", x="total_n_days", y="y_pred", color="r", label="predicted score", ax=ax)
        # ax.set_xlabel("Total days using VLE")
        # ax.set_ylabel("Assessment score")
        # plt.show()

    _LOG.info(f'Model {rq} results: {results}')
    results_df = pd.DataFrame.from_dict(results.items())
    results_df.to_csv(base_wd + f"\\outputs\\models\\rq{rq}_results.csv")
    _LOG.info(f'Model metrics for {rq} results saved.')

    if rq == 1:
        # Shapley values
        X_sub = shap.utils.sample(X, 1500)
        X_sub['year'] = X_sub['year'] .astype(float)
        explainer = shap.TreeExplainer(mod, X_sub)
        shap_values = explainer.shap_values(X_sub)

        # SHAP waterfall plot - local prediction
        plt.clf()
        shap.initjs()
        row = 1
        shap.waterfall_plot(shap.Explanation(values=shap_values[0][row],
                                             base_values=explainer.expected_value[0],  # 1
                                             data=X_sub.iloc[row],
                                             feature_names=X_sub.columns.tolist())
                            )
        waterfall = plt.gcf()
        plt.tight_layout()
        waterfall.savefig(base_wd + f"\\outputs\\models\\rq{rq}_waterfall_row{row}.png")
        # Notes - f(x) is the model predict_proba value, E[f(x)] is the base value

        # SHAP beeswarm plot
        shap.initjs()
        shap.summary_plot(shap_values=shap_values[1],
                          features=X_sub,
                          feature_names=list(X_sub.columns),
                          max_display=11,
                          show=False
                          )
        beeswarm = plt.gcf()
        beeswarm.savefig(base_wd + f"\\outputs\\models\\rq{rq}_beeswarm.png")

    return results
