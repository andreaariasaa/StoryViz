import pandas as pd
import numpy as np

lookup = pd.read_excel('Data/survey data look-up table.xlsx', sheet_name='answer_lookup')
main_type = pd.read_excel('Data/survey data look-up table.xlsx', sheet_name='main_table')


# Given a dataframe, question, and result, it will
# filter over the dataframe and return a new DataFrame
# that applies to this result
def narrow_data(current_df, question, result):
    question_type = main_type.loc[main_type['label'] == question]['type'].values[0]
    if question_type == 'c':
        look_key = question
    else:
        look_key = question_type
    new_lookup = lookup[np.logical_and(lookup['label'] == look_key, lookup['text_content'] == result)]
    result_code = new_lookup.iloc[0]['name']

    return current_df[current_df[question] == result_code]
