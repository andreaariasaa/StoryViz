import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textwrap import wrap
import time


# loading reference table to be used in the functions
lookup = pd.read_excel('Data/survey data look-up table.xlsx', sheet_name='answer_lookup')
main_type = pd.read_excel('Data/survey data look-up table.xlsx', sheet_name='main_table')
for type in ['hh','mig_ext', 'mig_int', 'mig_pend']:
    append_type = pd.read_excel('Data/survey data look-up table.xlsx', sheet_name=f'{type}_roster')
    main_type = main_type.append(append_type)

    
def remove_outliers(df,attribute):
    newdf = df[~df.groupby(attribute).transform( lambda x: abs(x-x.mean()) > 1.96*x.std()).values]
    return newdf


# plot categorical data in barplot
def plot_category(df, attribute, split_country=False):
    type = main_type.loc[main_type['label'] == attribute]['type'].values[0]
    title = main_type.loc[main_type['label'] == attribute]['question'].values[0]
    print(title)

    if split_country:
        attribute_list = ['country', attribute]
        df = df[['rsp_id'] + attribute_list].rename(columns={'rsp_id':'count'})
        count = df.groupby(by = attribute_list).count().reset_index()
        count = count.pivot(index='country', columns=attribute, values='count').fillna(0).stack().reset_index().rename(columns={0:'count'})
        country_total = count.groupby(by='country').sum()['count'].reset_index().rename(columns={'count':'total'})
        count = count.merge(country_total, on='country')
        count['pct'] = count['count']/count['total']
        count = count.sort_values(by=['country',attribute], ascending=True).reset_index(drop=True)

    else:
        attribute_list = [attribute]
        df = df[['rsp_id'] + attribute_list].rename(columns={'rsp_id':'count'})
        count = df.groupby(by = attribute_list).count().reset_index()
        count['pct'] = count['count']/count['count'].sum()

    if type == 'c':
        look_key = attribute
    else:
        look_key = type
    lookup_tabel = lookup[lookup['label'] == look_key][['name','text_content']]
    count[attribute] = count[[attribute]].replace(dict(zip(lookup_tabel.name, lookup_tabel.text_content)))

    sns.set_style("whitegrid")
    f, ax = plt.subplots(figsize=(12, 8))

    if split_country:
        barplot = sns.barplot(x=attribute, y="pct", hue="country", data=count, palette="Blues_d", hue_order=['GT','HND','SLV'], order=list(count[attribute].unique()))
        plt.legend(loc='upper right')
    else:
        barplot = sns.barplot(x=attribute, y="count", data=count, color="steelblue", order=list(count[attribute].unique()))
    plt.xticks(rotation = 90)
    ax.set_title("\n".join(wrap(title, 60)))
    if split_country:
        for i,p in zip(count.index, barplot.patches):
            barplot.annotate(str((count.loc[i,'pct']*100).round(1))+'%', xy=(p.get_x()+p.get_width()/4, p.get_height()),
                    xytext=(5, 0), textcoords='offset points', ha="center", va="bottom")
        plt.show()
        display(count.set_index(['country',attribute]).round(2))

    else:
        for i in count.index:
            barplot.annotate(str((count.loc[i,'pct']*100).round(1))+'%', xy=(i,count.loc[i,'count']), ha='center', va='bottom')
        plt.show()
        display(count.set_index(attribute).round(2))



# plot numerical data in histgrams
def plot_distribution(df, attribute, split_country=False):

    title = main_type.loc[main_type['label'] == attribute]['question'].values[0]
    print(' ')
    print(' ')
    print(title)

    sns.set_style("whitegrid")

    if not split_country:
        df = df[[attribute]]
        f, ax = plt.subplots(figsize=(8, 6))
        plt.hist(df[attribute], bins='auto', color='steelblue')
        plt.xlabel(attribute)
        ax.set_title("\n".join(wrap(title, 60)))
        plt.show()
        display(df[attribute].describe().round(2).reset_index())
    else:
        df = df[[attribute,'country']]
        dist = sns.displot(data=df, x=attribute, col='country', height=6, aspect=1)
        plt.show()
        table_display = df[attribute].describe().reset_index()[['index']]
        for country in ['GT','HND','SLV']:
            country_df = pd.DataFrame(df[df['country'] == country][attribute])
            country_display = country_df[attribute].describe().round(2).reset_index().rename(columns={attribute:country})
            table_display = table_display.merge(country_display, on= 'index', how='outer')
        display(table_display.set_index('index'))



# plot multiple choice questions in barplots
def plot_multi_choice(df, attribute, split_country = False):

    if split_country:
        attribute_list = list(lookup[lookup['label'] == attribute]['name_mco'])
        count_df = pd.DataFrame(columns=['country',attribute,'count','pct'])
        # count_df = pd.DataFrame(columns=['country',attribute,'count'])
        total_list = []
        for country in ['GT','HND','SLV']:
            table = pd.DataFrame(df[df['country'] == country])
            total = table[attribute].count()
            count = pd.DataFrame(table[attribute_list].sum(axis=0)).reset_index().rename(columns={'index':attribute,0:'count'})
            count = pd.DataFrame(attribute_list, columns=[attribute]).merge(count, on=attribute, how='left').fillna(0)
            count['country'] = country
            count['pct'] = count['count']/total.sum()
            count_df = count_df.append(count)
            total_list.append(total)

        title = main_type.loc[main_type['label'] == attribute]['question'].values[0] + f" (multiple choice, total response:{total_list})"
    else:
        total = df[attribute].count()
        attribute_list = list(lookup[lookup['label'] == attribute]['name_mco'])
        count_df = pd.DataFrame(df[attribute_list].sum(axis=0)).reset_index().rename(columns={'index':attribute,0:'count'})
        count_df['pct'] = count_df['count']/total.sum()
        title = main_type.loc[main_type['label'] == attribute]['question'].values[0] + f" (multiple choice, total response:{total})"

    lookup_tabel = lookup[lookup['label'] == attribute][['name_mco','text_content']]
    count_df[attribute] = count_df[[attribute]].replace(dict(zip(lookup_tabel.name_mco, lookup_tabel.text_content)))

    print(' ')
    print(' ')
    print(title)

    sns.set_style("whitegrid")
    f, ax = plt.subplots(figsize=(12, 8))
    if split_country:
        count_df = count_df.sort_values(by=['country',attribute], ascending=True).reset_index(drop=True)
        barplot = sns.barplot(x=attribute, y="pct", hue="country", data=count_df, palette="Blues_d", hue_order=['GT','HND','SLV'], order=list(count_df[attribute].unique()))
        plt.legend(loc='upper right')
    else:
        count_df = count_df.sort_values(by=attribute, ascending=True).reset_index(drop=True)
        barplot = sns.barplot(x=attribute, y="count", data=count_df, color="steelblue", order=list(count_df[attribute].unique()))
    plt.xticks(rotation = 90)
    ax.set_title("\n".join(wrap(title, 60)))
    if split_country:
        for i,p in zip(count_df.index, barplot.patches):
            barplot.annotate(str((count_df.loc[i,'pct']*100).round(1))+'%', xy=(p.get_x()+p.get_width()/4, p.get_height()),
                    xytext=(5, 0), textcoords='offset points', ha="center", va="bottom")
        plt.show()
        display(count_df.set_index(['country',attribute]).round(2))
    else:
        for i in count_df.index:
            barplot.annotate(str((count_df.loc[i,'pct']*100).round(1))+'%', xy=(i,count_df.loc[i,'count']), ha='center', va='bottom')
        plt.show()
        display(count_df.set_index(attribute).round(2))



# cooccurence of options in one multiple choice question
def cooccurence_heatmap_1(df, attribute, split_country=False):

    def plot_cooccur(df, attribute):
        attribute1_list = list(lookup[lookup['label'] == attribute]['name_mco'])
        attribute_list = ['rsp_id'] + attribute1_list
        attribute_df = df[attribute_list].set_index('rsp_id').astype(int)

        coocc_df = attribute_df.T.dot(attribute_df)
        lookup_tabel = lookup[lookup['label'] == attribute][['name_mco','text_content']]
        coocc_df = coocc_df.rename(dict(zip(lookup_tabel.name_mco, lookup_tabel.text_content)), axis=0)
        coocc_df = coocc_df.rename(dict(zip(lookup_tabel.name_mco, lookup_tabel.text_content)), axis=1)
        mask = np.zeros_like(coocc_df)
        mask[np.triu_indices_from(mask)] = True
        f, ax = plt.subplots(figsize=(12, 8))
        ax = sns.heatmap(np.log(coocc_df+1), mask=mask, annot=coocc_df, fmt='d', cmap="YlGnBu", linewidths=.5, cbar=False)
        plt.show()

    if split_country:
        for country in ['GT','HND','SLV']:
            country_df = pd.DataFrame(df[df['country'] == country])
            plot_cooccur(country_df, attribute)
    else:
        plot_cooccur(df, attribute)

# multiple choice x multiple choice
def cooccurence_heatmap_2(df, attribute1, attribute2, split_country=False):

    def plot_cooccur(df, attribute1, attribute2):
        attribute1_list = list(lookup[lookup['label'] == attribute1]['name_mco'])
        attribute2_list = list(lookup[lookup['label'] == attribute2]['name_mco'])
        attribute_list = ['rsp_id'] + attribute1_list + attribute2_list
        attribute_df = df[attribute_list].set_index('rsp_id')

        coocc_df = attribute_df.astype(int).T.dot(attribute_df)
        coocc_df = coocc_df.loc[attribute2_list, attribute1_list]
        lookup_tabel = lookup[(lookup['label'] == attribute1) | (lookup['label'] == attribute2)][['name_mco','text_content']]
        coocc_df = coocc_df.rename(dict(zip(lookup_tabel.name_mco, lookup_tabel.text_content)), axis=0)
        coocc_df = coocc_df.rename(dict(zip(lookup_tabel.name_mco, lookup_tabel.text_content)), axis=1)
        f, ax = plt.subplots(figsize=(12, 8))
        ax = sns.heatmap(np.log(coocc_df+1), annot=coocc_df, fmt='d', cmap="YlGnBu", linewidths=.5, cbar=False)
        plt.show()

    if split_country:
        for country in ['GT','HND','SLV']:
            country_df = pd.DataFrame(df[df['country'] == country])
            plot_cooccur(country_df, attribute1, attribute2)
    else:
        plot_cooccur(df, attribute1, attribute2)

# categorical  x categorical
def cooccurence_heatmap_3(df, attribute1, attribute2, split_country=False):

    def plot_cooccur(df, attribute1, attribute2):
        df = df[['rsp_id',attribute1,attribute2]].dropna().set_index('rsp_id').astype('int')
        coocc_df = pd.crosstab(df[attribute1], df[attribute2])
        lookup_tabel1 = lookup[(lookup['label'] == attribute1)][['name','text_content']]
        lookup_tabel2 = lookup[(lookup['label'] == attribute2)][['name','text_content']]
        coocc_df = coocc_df.rename(dict(zip(lookup_tabel1.name, lookup_tabel1.text_content)), axis=1)
        coocc_df = coocc_df.rename(dict(zip(lookup_tabel2.name, lookup_tabel2.text_content)), axis=0)
        f, ax = plt.subplots(figsize=(12, 8))
        ax = sns.heatmap(np.log(coocc_df+1), annot=coocc_df, fmt='d', cmap="YlGnBu", linewidths=.5, cbar=False)
        # ax.set(xlabel='occupation after migration', ylabel='occupation before migration')
        plt.show()

    if split_country:
        for country in ['GT','HND','SLV']:
            country_df = pd.DataFrame(df[df['country'] == country])
            plot_cooccur(country_df, attribute1, attribute2)
    else:
        plot_cooccur(df, attribute1, attribute2)


# Use this function to plot data
def plot_data(df, attribute, split_country = False):
    type = main_type.loc[main_type['label'] == attribute]['type'].values[0]
    if type == 'n':
        plot_distribution(df, attribute, split_country)
    elif type == 'mc':
        plot_multi_choice(df, attribute, split_country)
    else:
        plot_category(df, attribute, split_country)


def lookup_tabel_type(attribute):

    type = main_type.loc[main_type['label'] == attribute]['type'].values[0]
    if type == 'c' or type == 'mc':
        look_key = attribute
    else:
        look_key = type
    lookup_tabel = lookup[lookup['label'] == look_key][['name','text_content']]
    return lookup_tabel


def condition_categorical(df, groupby_feature, condition_feature, condition):
    df_condition = df[[groupby_feature, condition_feature]][df[condition_feature] == condition].rename(columns={condition_feature:'pct'})
    total_number = df_condition.shape[0]
    result_df = (df_condition.groupby(groupby_feature).count()/total_number).reset_index()
    lookup_groupby_feature = lookup_tabel_type(groupby_feature)
    result_df[groupby_feature] = result_df[[groupby_feature]].replace(dict(zip(lookup_groupby_feature.name, lookup_groupby_feature.text_content)))

    lookup_condition_feature = lookup_tabel_type(condition_feature)
    condition_text = lookup_condition_feature[lookup_condition_feature['name'] == condition]['text_content'].values[0]
    title = f"{groupby_feature} where {condition_feature} is {condition_text}"

    sns.set_style("whitegrid")
    f, ax = plt.subplots(figsize=(8, 6))
    barplot = plt.bar(result_df[groupby_feature], result_df['pct'], color='steelblue')
    plt.xticks(rotation = 90)
    ax.set_title("\n".join(wrap(title, 60)))
    for i in result_df.index:
        plt.annotate(str((result_df.loc[i,'pct']*100).round(1))+'%', xy=(i,result_df.loc[i,'pct']), ha='center', va='bottom')

    print (result_df.set_index(groupby_feature).round(2))



def compute_condition(df, groupby_feature, condition_feature, condition, split_country=False):

    if not split_country:
        condition_categorical(df, groupby_feature, condition_feature, condition)
    else:
        slv = df[df['country'] == 'SLV']
        gtm = df[df['country'] == 'GT']
        hnd = df[df['country'] == 'HND']
        for df in [slv, gtm, hnd]:
            condition_categorical(df, groupby_feature, condition_feature, condition)



def categorical_crosstabular(df, groupby_feature, study_feature, stacked=True):
    type_groupby = main_type.loc[main_type['label'] == groupby_feature]['type'].values[0]
    type_study = main_type.loc[main_type['label'] == study_feature]['type'].values[0]

    df_study = df[['rsp_id', groupby_feature, study_feature]]
    groupby_result = df_study.groupby([groupby_feature, study_feature]).count().rename(columns={'rsp_id':'count'})
    groupby_result['pct'] = groupby_result.groupby(level=0).apply(lambda x: x / float(x.sum())).rename(columns={'count':'pct'})['pct']
    groupby_result = groupby_result.reset_index()

    lookup_groupby = lookup_tabel_type(groupby_feature)
    lookup_study = lookup_tabel_type(study_feature)
    groupby_result[groupby_feature] = groupby_result[[groupby_feature]].replace(dict(zip(lookup_groupby.name, lookup_groupby.text_content)))
    groupby_result[study_feature] = groupby_result[[study_feature]].replace(dict(zip(lookup_study.name, lookup_study.text_content)))

    df_result_stacked = groupby_result.reset_index().pivot(index=groupby_feature, columns=study_feature, values='pct')
    df_result = groupby_result

    title = f"The Proportion of {study_feature} of different {groupby_feature}"

    if not stacked:
        f, ax = plt.subplots(figsize=(10,7))
        bar_plot = sns.barplot(data=df_result, x='pct',y=groupby_feature, hue=study_feature, palette="Blues_d", dodge=True)
        bar_plot.legend(fontsize = 7)
    else:
        df_result_stacked.plot.bar(stacked=True, legend=True, figsize=(8,6), colormap='PuBu')
        plt.legend(loc='upper center', bbox_to_anchor=(1.2, 1), ncol=1, frameon=True, shadow=False, prop={'size':8})
    plt.title("\n".join(wrap(title, 60)))




def categorical_crosstabular_compare(df, groupby_feature, study_feature, stacked=True, split_country=False):
    if not split_country:
        categorical_crosstabular(df, groupby_feature, study_feature, stacked)
    else:
        slv = df[df['country'] == 'SLV']
        gtm = df[df['country'] == 'GT']
        hnd = df[df['country'] == 'HND']
        for df in [slv, gtm, hnd]:
            categorical_crosstabular(df, groupby_feature, study_feature, stacked)


def numerical_scatter_plot(df, x_feature, y_feature, hue_feature):
    title = f'The scatterplot of {x_feature} and {y_feature}'
    f, ax = plt.subplots(figsize=(8,6))
    if hue_feature:
        df_study = df[[x_feature, y_feature, hue_feature]]
        sns.scatterplot(data=df_study, x=x_feature, y=y_feature, hue=hue_feature)
    else:
        df_study = df[[x_feature, y_feature]]
        sns.scatterplot(data=df_study, x=x_feature, y=y_feature)
    ax.set_title("\n".join(wrap(title, 60)))



def scatterplot_numeric(df, x_feature, y_feature, hue_feature=False, split_country=False):
        if not split_country:
            numerical_scatter_plot(df, x_feature, y_feature, hue_feature)
        else:
            slv = df[df['country'] == 'SLV']
            gtm = df[df['country'] == 'GT']
            hnd = df[df['country'] == 'HND']
            for df in [slv, gtm, hnd]:
                numerical_scatter_plot(df, x_feature, y_feature, hue_feature)

def cat_num_boxplot(df, cat_feature, num_feature):

    type = main_type.loc[main_type['label'] == cat_feature]['type'].values[0]

    if type == 'mc':
        lookup_tabel = lookup[lookup['label'] == cat_feature][['name_mco','text_content']]
        cat_opt_dict = dict(zip(lookup_tabel.name_mco, lookup_tabel.text_content))
        df_study = pd.DataFrame(columns=[cat_feature, num_feature])

        for options in cat_opt_dict.keys():
            data_option = df[df[options] == 1][[options, num_feature]].dropna().rename(columns={options:cat_feature})
            data_option[cat_feature] = cat_opt_dict[options]
            df_study = df_study.append(data_option, ignore_index=True)
    else:
        df_study = df[[cat_feature, num_feature]].dropna()
        lookup_cat_feature = lookup_tabel_type(cat_feature)
        df_study[cat_feature] = df_study[[cat_feature]].replace(dict(zip(lookup_cat_feature.name, lookup_cat_feature.text_content)))

    title = f'The distribution of {num_feature} of different {cat_feature}'
    sns.set_style("whitegrid")
    f, ax = plt.subplots(figsize=(8, 6))
    ax = sns.boxplot(x=cat_feature, y=num_feature, data=df_study)
    plt.xticks(rotation = 90)
    ax.set_title("\n".join(wrap(title, 60)))

    print (df_study.groupby(by=cat_feature).describe()[[(num_feature,'count'),(num_feature,'mean'),(num_feature,'min'),(num_feature,'max'),(num_feature,'std')]])


def plot_cat_num(df, cat_feature, num_feature, split_country=False):
    if not split_country:
        cat_num_boxplot(df, cat_feature, num_feature)
    else:
        slv = df[df['country'] == 'SLV']
        gtm = df[df['country'] == 'GT']
        hnd = df[df['country'] == 'HND']
        for df in [slv, gtm, hnd]:
            cat_num_boxplot(df, cat_feature, num_feature)
