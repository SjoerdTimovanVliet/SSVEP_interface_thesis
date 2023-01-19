import numpy as np
import scipy.stats as stats
import sys
import pingouin as pg
import pandas as pd
from typing import Union


def create_table_with_significance_using_pvalue(pvalue_matrix: np.ndarray, groups: np.ndarray, alpha_threshold=0.05) -> pd.DataFrame:
    """ Create a table with the significance of the pairwise comparisons

    Args:
        pvalue_matrix (np.ndarray): Pvalue matrix with the pvalues of the pairwise comparisons
        groups (np.ndarray): Array with the groups
        alpha_threshold (float, optional): Threshold for the significance. Defaults to 0.05.

    Returns:
        pd.DataFrame: Table with the significance of the pairwise comparisons
    """
    # create a table with the pairwise comparisons with the rows one group tested against the other group mentioned in the columns
    pairwise_comparisons_table_significance = pd.DataFrame(
        index=groups, columns=groups)

    # loop through the unique groups and check if the pairwise comparison is in the pairwise comparisons table
    for i in range(groups.shape[0]):
        for j in range(groups.shape[0]):

            # extract labels of the groups to be compared
            group_A = groups[i]
            group_B = groups[j]

            # save the significance in the table pairwise_comparisons_table_significance
            if pvalue_matrix[i, j] <= alpha_threshold:
                pairwise_comparisons_table_significance.loc[group_A,
                                                            group_B] = 'Significant'
            else:
                pairwise_comparisons_table_significance.loc[group_A,
                                                            group_B] = 'Not significant'

    return pairwise_comparisons_table_significance


def create_table_with_pairwise_comparisons(pairwise_comparisons: pd.DataFrame, alpha_pairwise_comparisons=0.05) -> Union[pd.DataFrame, pd.DataFrame]:
    """ Create a table with the pairwise comparisons with the rows one group tested against the other group mentioned in the columns

    Args:
        pairwise_comparisons (pd.DataFrame): Table with the pairwise comparisons
        alpha_pairwise_comparisons (float, optional): Alpha value for the pairwise comparisons. Defaults to 0.05.

    Returns:
        Union[pd.DataFrame, pd.DataFrame]: Table with the p-values and table with the significance of the pairwise comparisons
    """
    # extract the unique group labels from the pairwise comparisons table
    unique_groups_A = pairwise_comparisons['A'].unique()
    unique_groups_B = pairwise_comparisons['B'].unique()
    unique_groups = np.unique(np.concatenate(
        (unique_groups_A, unique_groups_B)))

    # extract the p-values from the pairwise comparisons table
    p_values_pairwise_comparisons = pairwise_comparisons['pval']

    # create a table with the pairwise comparisons with the rows one group tested against the other group mentioned in the columns
    pairwise_comparisons_table_p_values = pd.DataFrame(
        index=unique_groups, columns=unique_groups)
    pairwise_comparisons_table_significance = pd.DataFrame(
        index=unique_groups, columns=unique_groups)

    # loop through the unique groups and check if the pairwise comparison is in the pairwise comparisons table
    for i in range(unique_groups.shape[0]):
        for j in range(unique_groups.shape[0]):

            # extract labels of the groups to be compared
            group_A = unique_groups[i]
            group_B = unique_groups[j]

            # check where the group labels are in the pairwise comparisons table
            Check_if_pair_A_B_in_pairwise_comparisons = (
                pairwise_comparisons['A'] == group_A) & (pairwise_comparisons['B'] == group_B)

            # check if any of them are true
            if np.any(Check_if_pair_A_B_in_pairwise_comparisons):
                print("Pairwise comparison between group {} and group {} is in the pairwise comparisons table".format(
                    group_A, group_B))
                # save the p-value in the table
                p_value = float(p_values_pairwise_comparisons[(
                    pairwise_comparisons['A'] == group_A) & (pairwise_comparisons['B'] == group_B)])
                pairwise_comparisons_table_p_values.loc[group_A,
                                                        group_B] = p_value

                # save the significance in the table pairwise_comparisons_table_significance
                if p_value <= alpha_pairwise_comparisons:
                    pairwise_comparisons_table_significance.loc[group_A,
                                                                group_B] = 'Significant'
                else:
                    pairwise_comparisons_table_significance.loc[group_A,
                                                                group_B] = 'Not significant'

            else:
                Check_if_pair_B_A_in_pairwise_comparisons = (
                    pairwise_comparisons['A'] == group_B) & (pairwise_comparisons['B'] == group_A)

                if np.any(Check_if_pair_B_A_in_pairwise_comparisons):
                    print("Pairwise comparison between group {} and group {} is in the pairwise comparisons table".format(
                        group_B, group_A))
                    # save the p-value in the table
                    p_value = float(p_values_pairwise_comparisons[(
                        pairwise_comparisons['A'] == group_B) & (pairwise_comparisons['B'] == group_A)])
                    # save the p-value in the table pairwise_comparisons_table_p_values
                    pairwise_comparisons_table_p_values.loc[group_A,
                                                            group_B] = p_value

                    # save the significance in the table pairwise_comparisons_table_significance
                    if p_value <= alpha_pairwise_comparisons:
                        pairwise_comparisons_table_significance.loc[group_A,
                                                                    group_B] = 'Significant'
                    else:
                        pairwise_comparisons_table_significance.loc[group_A,
                                                                    group_B] = 'Not significant'

                # if the group labels are not in the pairwise comparisons table, put a NaN in the table
                else:
                    print("Pairwise comparison between group {} and group {} is not in the pairwise comparisons table".format(
                        group_A, group_B))
                    # put a NaN in the table
                    pairwise_comparisons_table_p_values.loc[group_A,
                                                            group_B] = np.nan
                    # put an X in the table to indicate that the comparison is not in the pairwise comparisons table
                    pairwise_comparisons_table_significance.loc[group_A,
                                                                group_B] = "X"
                    continue

    return pairwise_comparisons_table_p_values, pairwise_comparisons_table_significance


# load the data
path = '/media/sjoerd/BackUp Drive/Thesis_project/Questionairre results/Questionairre_answers.csv'
data_csv = pd.read_csv(path)

# extract the categories from the data table which is the first column except the values that are NaN
categories = data_csv.iloc[:, 0].dropna()
# unique categories
unique_categories = np.unique(categories)

# extract the groups from the data table which is the second column except the values that are NaN
groups = data_csv.iloc[:, 1].dropna()

# extract the metrics from the data table which is the third column except the values that are NaN
metrics = data_csv.iloc[0:5, 2]

# headers
headers = ['category', 'groups',  ' metric', 'p_value_bartlett', 'Variances equal?', 'ANOVA method', 'df within groups', 'df between groups',
           'p_value ANOVA method', 'Mean different?', 'Type of post hoc test', 'p_value_matrix', 'Significance_matrix']


# create an empty dataframe to store the results of each step of the analysis
results_logger = pd.DataFrame(columns=headers)
for category in unique_categories:
    # create a dictionary to store the results of each step of the analysis
    results_back_log = {}
    # create a list to store the results of each step of the analysis
    row = []
    # save the category in the dictionary
    results_back_log['category'] = category
    # filter the groups that belong to the category
    groups_in_category = groups[categories == category]

    # perform  for each metric a comparison between the groups in the category
    for metric in metrics:

        # extract the values of the metric for each group in the category
        data = {}
        # for each group in the category
        for group in groups_in_category:
            print(f'category: {category}, group: {group}, metric: {metric}')
            # get the index of the group in the groups_in_category
            index_group = np.where(groups_in_category == group)[0][0]
            index_metric = np.where(metrics == metric)[0][0]
            actual_index_in_data_csv = index_group + index_metric
            # extract data of the group in the category for the metric
            data[group] = np.array(
                data_csv.iloc[actual_index_in_data_csv, 3:9]).astype(np.float64)
            print(
                f" The variance of the group {group} in metric {metric} is {np.var(data[group], ddof=1)}")

        # convert the list of groups in the category to an array
        groups_in_category_array = np.array(groups_in_category)

        # perform bartlett's test
        alpha_bartlett = 0.05  # 95% confidence
        # check number of groups
        if len(groups_in_category) == 3:
            # perform Bartlett's test
            print(stats.bartlett(data[groups_in_category_array[0]],
                  data[groups_in_category_array[1]], data[groups_in_category_array[2]]))
            # save the p-value from the test
            p_value_bartlett = stats.bartlett(
                data[groups_in_category_array[0]], data[groups_in_category_array[1]], data[groups_in_category_array[2]])[1]
            # Alpha is the significance level at which we reject the null hypothesis
        elif len(groups_in_category) == 4:
            # perform Bartlett's test
            print(stats.bartlett(data[groups_in_category_array[0]], data[groups_in_category_array[1]],
                  data[groups_in_category_array[2]], data[groups_in_category_array[3]]))
            # save the p-value from the test
            p_value_bartlett = stats.bartlett(data[groups_in_category_array[0]], data[groups_in_category_array[1]],
                                              data[groups_in_category_array[2]], data[groups_in_category_array[3]])[1]

        # if true variances are not equal
        if p_value_bartlett > alpha_bartlett:
            print('The variances are equal')
            variances_equal = 'Yes'
            # if the variances are equal, perform one-way ANOVA
            print("Performing one-way ANOVA")
            anova_method = 'One-way ANOVA'
            # perform one-way ANOVA
            alpha_one_way_anova = 0.05  # 95% confidence
            if len(groups_in_category) == 3:
                # perform one-way ANOVA
                print(stats.f_oneway(data[groups_in_category_array[0]],
                      data[groups_in_category_array[1]], data[groups_in_category_array[2]]))
                # save the p-value from the test
                pvalue_f_oneway = stats.f_oneway(
                    data[groups_in_category_array[0]], data[groups_in_category_array[1]], data[groups_in_category_array[2]])[1]
                # Alpha is the significance level at which we reject the null hypothesis
            elif len(groups_in_category) == 4:
                # perform one-way ANOVA
                print(stats.f_oneway(data[groups_in_category_array[0]], data[groups_in_category_array[1]],
                      data[groups_in_category_array[2]], data[groups_in_category_array[3]]))
                # save the p-value from the test
                pvalue_f_oneway = stats.f_oneway(data[groups_in_category_array[0]], data[groups_in_category_array[1]],
                                                 data[groups_in_category_array[2]], data[groups_in_category_array[3]])

            # save the p-value from the test
            p_value_anova_method = pvalue_f_oneway

            # calculate the degrees of freedom of the f one-way anova
            total_number_of_observations = 0
            for group_i in groups_in_category_array:
                total_number_of_observations += len(data[group_i])
            # calculate all the degrees of freedom of the f one-way anova
            df_between_groups = len(groups_in_category_array) - 1
            df_within_groups = total_number_of_observations - \
                len(groups_in_category_array)
            df_total = total_number_of_observations - 1

            # Alpha is the significance level at which we reject the null hypothesis
            # if p-value is greater than alpha, accept null hypothesis (the means are equal)
            if pvalue_f_oneway > alpha_one_way_anova:
                print('The means are equal')
                mean_different = 'No'
                type_of_post_hoc_test = 'None'
                p_value_matrix = 'None'
                significance_matrix = 'None'

            # if p-value is less than alpha, reject null hypothesis (the means are not equal)
            elif pvalue_f_oneway <= alpha_one_way_anova:
                print('The means are not equal')
                mean_different = 'Yes'
                print("Performing Tukey's test")
                anova_method = 'Tukey\'s test'

                # perform Tukey's test
                if len(groups_in_category) == 3:
                    # perform Tukey's post-hoc test
                    print(stats.tukey_hsd(
                        data[groups_in_category_array[0]], data[groups_in_category_array[1]], data[groups_in_category_array[2]]))
                    # save the p-value from the test
                    results = stats.tukey_hsd(
                        data[groups_in_category_array[0]], data[groups_in_category_array[1]], data[groups_in_category_array[2]])
                    # Alpha is the significance level at which we reject the null hypothesis
                elif len(groups_in_category) == 4:
                    # perform Tukey's post-hoc test
                    print(stats.tukey_hsd(data[groups_in_category_array[0]], data[groups_in_category_array[1]],
                          data[groups_in_category_array[2]], data[groups_in_category_array[3]]))
                    # save the p-value from the test
                    results = stats.tukey_hsd(data[groups_in_category_array[0]], data[groups_in_category_array[1]],
                                              data[groups_in_category_array[2]], data[groups_in_category_array[3]])

                # save the p-value from the test
                pvalue_matrix = results.pvalue
                alpha_tukey = 0.05  # 95% confidence
                tukey_table_significance = create_table_with_significance_using_pvalue(
                    pvalue_matrix, groups_in_category_array, alpha_tukey)
                # save the p-value from the test
                p_value_matrix = pvalue_matrix
                # save the significance matrix
                significance_matrix = tukey_table_significance

        # if the variances are not equal, perform Welch's ANOVA
        elif p_value_bartlett <= alpha_bartlett:
            # log that variances are not equal
            variances_equal = 'No'
            print('The variances are not equal')
            # if the variances are not equal, perform Welch's ANOVA
            print("Performing Welch's ANOVA")
            anova_method = "Welch's ANOVA"

            # check number of groups
            if len(groups_in_category) == 3:
                # create a dataframe with the values of the groups
                df = pd.DataFrame({'value': np.concatenate((data[groups_in_category_array[0]], data[groups_in_category_array[1]], data[groups_in_category_array[2]])), 'group': np.concatenate((np.repeat(
                    groups_in_category[0], len(data[groups_in_category[0]])), np.repeat(groups_in_category[1], len(data[groups_in_category[1]])), np.repeat(groups_in_category_array[2], len(data[groups_in_category_array[2]]))))})
            elif len(groups_in_category) == 4:
                # create a dataframe with the values of the groups
                df = pd.DataFrame({'value': np.concatenate((data[groups_in_category_array[0]], data[groups_in_category_array[1]], data[groups_in_category_array[2]], data[groups_in_category_array[3]])), 'group': np.concatenate((np.repeat(groups_in_category_array[0], len(
                    data[groups_in_category_array[0]])), np.repeat(groups_in_category_array[1], len(data[groups_in_category_array[1]])), np.repeat(groups_in_category_array[2], len(data[groups_in_category_array[2]])), np.repeat(groups_in_category_array[3], len(data[groups_in_category_array[3]]))))})

            # perform Welch's ANOVA
            print(pg.welch_anova(data=df, dv='value', between='group'))
            # PERFORM WELCH'S ANOVA USING PINGOUIN  and save the results in a variable
            results = pg.welch_anova(data=df, dv='value', between='group')

            # extract df between groups from the results
            df_between_groups = results['ddof1'][0]
            # extract the degrees of freedom within groups from the results
            df_within_groups = results['ddof2'][0]
            df_total = df_between_groups + df_within_groups

            # extract the p-value from the results
            p_value_welch_anova = results['p-unc'][0]
            alpha_welch_anova = 0.05  # 95% confidence

            # if the p-value is greater than alpha, accept null hypothesis (the means are equal)
            if p_value_welch_anova > alpha_welch_anova:
                mean_different = 'No'
                print('The mean values are equal')
                type_of_post_hoc_test = 'None'
                p_value_matrix = 'None'
                Significance_matrix = 'None'

            # if the p-value is less than alpha, reject null hypothesis (the means are not equal)
            elif p_value_welch_anova <= alpha_welch_anova:
                mean_different = 'Yes'
                print('The mean values are not equal')
                type_of_post_hoc_test = 'Games-Howell'
                # PERFROM THE pairwise games-howell post hoc test
                print(pg.pairwise_gameshowell(
                    data=df, dv='value', between='group'))

                # create a table with the pairwise comparisons
                pairwise_comparisons = pg.pairwise_gameshowell(
                    data=df, dv='value', between='group')
                alpha_pairwise_comparisons = 0.05  # 95% confidence
                table_with_p_values, p_values_with_significance = create_table_with_pairwise_comparisons(
                    pairwise_comparisons, alpha_pairwise_comparisons)

                # save the data in the logger
                p_value_matrix = table_with_p_values
                significance_matrix = p_values_with_significance

            # save the p-value from the variance test
            p_value_anova_method = p_value_welch_anova

        try:  # round the total degrees of freedom to 2 decimal places
            df_between_groups = round(df_between_groups, 2)
            df_within_groups = round(df_within_groups, 2)
            df_total = round(df_total, 2)

            # create a row with the results
            row = [category, np.array(groups_in_category), metric, p_value_bartlett, variances_equal, anova_method,  df_between_groups,
                   df_within_groups, p_value_anova_method, mean_different, type_of_post_hoc_test, p_value_matrix, significance_matrix]

            # add the row to  results_logger dataframe
            results_logger.loc[len(results_logger)] = row
        except:
            print("Error in the data logger")
            # quit the program
            sys.exit()

# save the results in a csv file
results_logger.to_csv('results_logger_questionnaire.csv', index=False)
