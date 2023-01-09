import logging
from datetime import datetime

from pipeline.etl import run_etl
from pipeline.clean_data import clean_data
from pipeline.eda import eda
from pipeline.modeling import run_model

_LOG = logging.getLogger(__name__)


def main():

    logging.basicConfig(level=logging.INFO)

    kickoff = datetime.now()

    _LOG.info('Beginning initial EDA and ETL script.')
    rq1_df, rq2_df = run_etl()
    rq1_df.name, rq2_df.name = 'rq1_df', 'rq2_df'
    _LOG.info('Initial EDA and ETL script complete.\n')

    _LOG.info('Beginning initial data exploration and data cleaning.')
    rq1_df_cleaned, rq1, rq1_df2 = clean_data(rq1_df)
    rq2_df_cleaned, rq2, rq2_df2 = clean_data(rq2_df)
    _LOG.info('Initial data exploration and data cleaning complete.\n')

    _LOG.info('Beginning EDA with cleaned data.')
    eda(rq1_df2, rq1)
    eda(rq2_df2, rq2)
    _LOG.info('EDA with cleaned data complete.\n')

    _LOG.info('Beginning modeling.')

    start_time = datetime.now()

    rq1_results_wd_fail = run_model(rq1_df_cleaned, rq1, 4)

    class_time = datetime.now()
    _LOG.info(f'Classification model duration: {class_time - start_time}')

    rq2_results = run_model(rq2_df_cleaned, rq2, -1)

    reg_time = datetime.now()
    _LOG.info(f'Classification model duration: {class_time - start_time}')
    _LOG.info(f'Regression model duration: {reg_time - class_time}')

    _LOG.info('Modeling complete.')


if __name__ == '__main__':
    main()
