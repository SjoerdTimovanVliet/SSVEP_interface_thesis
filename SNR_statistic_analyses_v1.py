import numpy as np
import scipy.stats as stats
import sys
import os
import pingouin as pg
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union


def create_boxplot(row: np.ndarray, path_to_save: str, data_per_participant_dict: dict):
    """ Create a boxplot for the data

    Args:
        data (np.ndarray): Data to plot in the boxplot
        path_to_save (str): Path to save the figure
    """
    experiment = row[0]
    category = row[1]
    groups_in_category_array = row[2]
    # create a figure
    fig, ax = plt.subplots()

    # create a list of colors for the legend
    colors = ['green', 'purple', 'brown', 'grey', 'olive', 'cyan']
    # create a list for the labels for the legend
    labels = []
    # create a list for the handles for the legend
    handles = []
    # the sequence consits of 4 elements: snr, mean, data_length, data
    for i in range(len(groups_in_category_array)):
        group = groups_in_category_array[i]
        average = row[3+i*4]
        std = row[4+i*4]
        data_length = row[5+i*4]
        data = row[6+i*4]

        # boxplot with median and mean and using different colors for the mean and median also do not show the outliers
        bplot = ax.boxplot(data, positions=[i], widths=0.6, showmeans=True, meanline=True, showfliers=False,
                           patch_artist=True, medianprops=dict(color='blue'), meanprops=dict(color='red'))
        # set the color of the boxplot to be filled the color light blue
        for patch in bplot['boxes']:
            patch.set_facecolor('lightblue')

        # extract the keys of the dictionary which are the participant numbers
        participant_keys = data_per_participant_dict[group].keys()

        # put the keys in numerical order
        participant_keys = sorted(
            participant_keys, key=lambda x: int(x.split('pp')[1]))

        # loop over the participants
        for j, participant in enumerate(participant_keys):
            # calculate the stepsize for the mean lines
            stepsize = 1 / len(data_per_participant_dict[group].keys())
            # extract the data for the participant
            data_participant = data_per_participant_dict[group][participant]
            # combine the data of the participant to a list
            data_participant = list(data_participant)
            # convert the data to a numpy array and flatten it
            data_participant = np.array(data_participant).flatten()
            # calculate the mean of the data for the participant
            participant_mean = np.mean(data_participant)

            # extract the participant number
            # pp5 turns into pp1, etc.
            participant_number = int(participant.split('pp')[1])

            # show the lines of the mean of each participant in the boxplot within each box with an alpha of 0.5 at the correct position with respect to the x axis
            ax.plot([i-0.5 + j*stepsize, i - 0.4 + j*stepsize],
                    [participant_mean, participant_mean], color=colors[j], lw=3)
            if i == 0:
                # create the labels and handles for the legend
                label = f'pp {participant_number} mean'
                # add the label to the list of labels
                labels.append(label)
                handles.append(plt.Line2D([0], [0], color=colors[j], lw=4))

        # put vline after each boxplot
        ax.axvline(x=i+0.5, color='black', lw=1)

    if category == 'pixel_surface':
        ax.set_xlabel('pixel surface [pixels]')
        str_add = 'pixels'
    elif category == 'frequency':
        ax.set_xlabel('frequency [Hz]')
        str_add = 'Hz'
    else:
        ax.set_xlabel(category)
        str_add = ''

    ax.set_ylabel('SNR [dB]')
    # replace the "_" with a space in the category name and the experiment name
    category = category.replace('_', ' ')
    experiment = experiment.replace('_', ' ')

    ax.set_title('Boxplot SNR for ' + category + ' in ' +
                 experiment + ' without outliers')
    # set the x axis ticks to the groups in the category
    ax.set_xticks(np.arange(len(groups_in_category_array)))
    ax.set_xticklabels(groups_in_category_array)
    ax.yaxis.grid(True)

    # create a legend for the mean which is color red and the median which is color blue
    labels.append('mean of all data')
    labels.append('median of all data')

    # create a red line for the mean
    handles.append(plt.Line2D([0], [0], color='red', lw=4))
    # create a blue line for the median
    handles.append(plt.Line2D([0], [0], color='blue', lw=4))

    # create a custom legend with the labels and the colors
    plt.legend(handles, labels, loc='upper right',
               bbox_to_anchor=(1.4, 1.0), ncol=1)

    # save the figure
    plt.savefig(path_to_save + '/boxplot_' + experiment +
                '_' + category + '.png', dpi=300, bbox_inches='tight')


def create_figure(row: list, path_to_save: str):
    """ Create a figure with the snr and mean for each group in the category

    Args:
        row (list): Contains the experiment name, the category, the groups in the category, the snr and mean for each group in the category
        path_to_save (str): Path to save the figure
    """
    experiment = row[0]
    category = row[1]
    groups_in_category_array = row[2]

    snr_array = row[3:3+len(groups_in_category_array)*3]
    # the sequence consits of 3 elements: snr, mean, data_length
    # filter the snr and mean by removing the data_length, every 3rd element
    snr_array = [snr_array[i] for i in range(len(snr_array)) if i % 3 != 2]

    # for each group in the category extract the snr and mean
    snr_array = np.array(snr_array).reshape(len(groups_in_category_array), 2)
    # create a figure
    fig, ax = plt.subplots()
    # plot the snr and mean using a bar plot
    for i in range(len(groups_in_category_array)):
        # plot the snr with on the x axis the category and on the y axis the snr
        ax.bar(i, snr_array[i, 0], align='center',
               alpha=0.5, ecolor='black', capsize=10)
        # plot the std on the bars with respect to the snr mean
        ax.errorbar(i, snr_array[i, 0], yerr=snr_array[i, 1],
                    fmt='o', ecolor='black', capsize=10)

    ax.set_ylabel('SNR')
    if category == 'pixel_surface':
        ax.set_xlabel('pixel surface [pixels]')
        str_add = 'pixels'
    elif category == 'frequency':
        ax.set_xlabel('frequency [Hz]')
        str_add = 'Hz'
    else:
        ax.set_xlabel(category)
        str_add = ''

    # replace the "_" with a space in the category name and the experiment name
    category = category.replace('_', ' ')
    experiment = experiment.replace('_', ' ')

    ax.set_title('SNR for ' + category + ' in ' + experiment)
    # hide the x axis ticks
    ax.set_xticks([])
    ax.yaxis.grid(True)
    # show legend with the bars labeled with the groups in the category and the error bars labeled with the std
    # create the labels for the legend
    labels = []

    for i in range(len(groups_in_category_array)):

        labels.append(str(groups_in_category_array[i]) + f' {str_add} mean')
        labels.append(str(groups_in_category_array[i]) + f' {str_add} std')
    # create the legend and place it outside the figure showing the plots
    ax.legend(labels, loc='center left', bbox_to_anchor=(1, 0.5))
    # make sure the legend is not cut off
    plt.tight_layout()

    # save the figure in the path_to_save
    plt.savefig(path_to_save + '/' + experiment + '_' + category + '.png')


def convert_dataframe_strings_to_list_SNR(df: pd.DataFrame) -> pd.DataFrame:
    """ Convert the list of strings in the dataframe to list of floats

    Args:
        df (pd.DataFrame): Dataframe with the data as list of strings

    Returns:
        pd.DataFrame: Dataframe with the data as list of floats
    """
    # make an empty dataframe that will be filled with the data
    df_new = pd.DataFrame(columns=df.columns)

    # loop over columns
    for j in range(df.shape[0]):
        row = []
        # copy of row j the first 6 columns
        row = list(df.iloc[j, 0:6].copy())

        for i in range(6, df.shape[1]):
            # remove first and last character of the string
            string = df.iloc[j, i][1:-1]
            # separate the string by  space and convert to list
            list_of_strings = string.split(' ')
            # remove '' from the list
            list_of_strings = [x for x in list_of_strings if x != '']
            # convert the list to float
            array_of_floats = [float(x) for x in list_of_strings]
            # save the array in the dataframe

            row.append(array_of_floats)
        # append the row to the dataframe
        df_new.loc[len(df_new)] = row

    return df_new


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

    return pairwise_comparisons_table_p_values, pairwise_comparisons_table_significance


path = r"/media/sjoerd/BackUp Drive/Thesis_project/Data_SNR/SNR_sorted_by_participant.csv"
# get path of folder where the results are saved
path_folder = os.path.dirname(path)
# create new folder named "SNR results" in the folder where the results are saved
path_folder_results = os.path.join(path_folder, "SNR results")
# check if the folder already exists
if not os.path.exists(path_folder_results):
    # if not, create the folder
    os.makedirs(path_folder_results)

# read the data
data = pd.read_csv(path)
# convert the strings in the data to lists
data = convert_dataframe_strings_to_list_SNR(data)

# get the headers of the data
headers = data.columns.values
# unique experiments (folder)
unique_experiments = data['Folder'].unique()
# unique categories
unique_categories_all = headers[2:6]

# dictionary with the unique groups per experiment per category
groups_per_experiment_dict = {}
for experiment in unique_experiments:
    groups_per_experiment_dict[experiment] = {}
    for category in unique_categories_all:
        # extract unique metrics per experiment per category
        unique_groups = data[data['Folder'] == experiment][category].unique()
        # sort the unique groups
        unique_groups.sort()
        groups_per_experiment_dict[experiment][category] = unique_groups

# dictionary to store the results of the analysis
data_logger = {}

# loop over the experiments
for experiment in unique_experiments:
    # create a dictionary for the experiment
    data_logger[experiment] = {}

    # headers with average and std of the groups in the experiment
    headers = ['category', 'groups',  'p_value_bartlett', 'Variances equal?', 'ANOVA method', 'df within groups', 'df between groups',
               'p_value ANOVA method', 'Mean different?', 'Type of post hoc test', 'p_value_matrix', 'Significance_matrix']
    # create an empty dataframe to store the results of each step of the analysis
    results_logger = pd.DataFrame(columns=headers)

    # snr_metrics dataframe of the experiment and groups from A to B
    headers_snr = ['experiment', 'category', 'groups', 'SNR group A average', 'SNR group A std', 'group A number of samples', 'SNR group B average', 'SNR group B std',
                   'group B number of samples', 'SNR group C average', 'SNR group C std', 'group C number of samples', 'SNR group D average', 'SNR group D std', 'group D number of samples', ]
    # create an empty dataframe to store the results of each step of the analysis
    snr_metrics = pd.DataFrame(columns=headers_snr)

    # snr_metrics dataframe of the experiment and groups from A to B
    if 'Experiment_2' == experiment:
        # remove 'pixel_surface' from the unique categories
        unique_categories = [x for x in unique_categories_all if x != 'shape']
    else:
        unique_categories = unique_categories_all

    # create a dictionary for the categories
    data_logger[experiment]['category'] = {}
    headers_snr_extensive = ['experiment', 'category', 'groups', 'SNR group A average', 'SNR group A std', 'group A number of samples', 'Group A data', 'SNR group B average', 'SNR group B std', 'group B number of samples', 'Group B data',
                             'SNR group C average', 'SNR group C std', 'group C number of samples', 'Group C data', 'SNR group D average', 'SNR group D std', 'group D number of samples', 'Group D data']
    # create an empty dataframe to store the results of each step of the analysis
    snr_metrics_extensive = pd.DataFrame(columns=headers_snr_extensive)
    # loop over the categories
    for category in unique_categories:
        # create a dictionary for the category
        data_logger[experiment]['category'][category] = {}
        # create a row for the results_logger dataframe
        row = []

        # get the groups in the category
        groups_in_category_array = groups_per_experiment_dict[experiment][category]
        # create a dictionary for the groups
        data_analysis = {}
        # store the experiment, category and groups in the row
        row_snr = [experiment, category, groups_in_category_array]
        row_snr_extensive = [experiment, category, groups_in_category_array]
        # create a dictionary for the data of the participants
        data_per_participant_dict = {}
        # loop over the groups
        for group in groups_in_category_array:
            # create an empty dictionary for the group in the data_per_participant_dict
            data_per_participant_dict[group] = {}
            # create a dictionary for the group
            data_logger[experiment]['category'][category][group] = {}
            # extract snr data of participants in the group
            data_experiment = data[(data['Folder'] == experiment)]
            # extract all the snr data of the group from the participants
            data_group = data_experiment[data_experiment[category]
                                         == group].iloc[:, 6:]

            # extract keys of the data_group dataframe
            participant_keys = data_group.keys()
            # loop over the keys
            for key in participant_keys:
                # extract the data of the participant
                data_participant = data_group[key]
                # put the data in a dictionary
                data_per_participant_dict[group][key] = data_participant

            # convert the dataframe to a list
            data_group_list = data_group.values.tolist()
            # convert the list to a 1d array
            data_group_1d_array = np.array(data_group_list).flatten()

            # log the data
            data_logger[experiment]['category'][category][group]['data'] = data_group_1d_array
            # store the data in a dictionary
            data_analysis[group] = data_group_1d_array

            # calculate the average and std of the group
            average = np.mean(data_group_1d_array)
            std = np.std(data_group_1d_array)

            # log the average and std
            data_logger[experiment]['category'][category][group]['average'] = average
            data_logger[experiment]['category'][category][group]['std'] = std
            data_logger[experiment]['category'][category][group]['number of samples'] = len(
                data_group_1d_array)

            # append the average and std to the row
            row_snr.append(average)
            row_snr.append(std)
            row_snr.append(len(data_group_1d_array))
            row_snr_extensive.append(average)
            row_snr_extensive.append(std)
            row_snr_extensive.append(len(data_group_1d_array))
            row_snr_extensive.append(data_group_1d_array)

        if len(groups_in_category_array) == 3:
            # add the nan for the group D which is not present in the experiment
            row_snr.append(np.nan)
            row_snr.append(np.nan)
            row_snr.append(np.nan)
            row_snr_extensive.append(np.nan)
            row_snr_extensive.append(np.nan)
            row_snr_extensive.append(np.nan)
            row_snr_extensive.append(np.nan)

        try:
            # append the row to the snr_metrics dataframe
            print(
                f"Appending row to snr_metrics dataframe for experiment: {experiment} and category: {category}")
            snr_metrics.loc[len(snr_metrics)] = row_snr
            snr_metrics_extensive.loc[len(
                snr_metrics_extensive)] = row_snr_extensive
            # create the figures
            print("Creating figures for experiment: ", experiment)
            create_figure(row_snr, path_folder_results)
            create_boxplot(row_snr_extensive, path_folder_results,
                           data_per_participant_dict)

        except:
            print(
                'Error in the creation of the figure or the append of the row to the snr_metrics dataframe')
            #print('row_snr: ', row_snr)
            print('path_folder_results: ', path_folder_results)
            sys.exit()

        # Start the statistical analysis
        # perform bartlett's test
        # check number of groups
        if len(groups_in_category_array) == 3:
            # perform Bartlett's test
            print(stats.bartlett(data_analysis[groups_in_category_array[0]],
                  data_analysis[groups_in_category_array[1]], data_analysis[groups_in_category_array[2]]))
            # save the p-value from the test
            p_value_bartlett = stats.bartlett(
                data_analysis[groups_in_category_array[0]], data_analysis[groups_in_category_array[1]], data_analysis[groups_in_category_array[2]])[1]

        elif len(groups_in_category_array) == 4:
            # perform Bartlett's test
            print(stats.bartlett(data_analysis[groups_in_category_array[0]], data_analysis[groups_in_category_array[1]],
                  data_analysis[groups_in_category_array[2]], data_analysis[groups_in_category_array[3]]))
            # save the p-value from the test
            p_value_bartlett = stats.bartlett(data_analysis[groups_in_category_array[0]], data_analysis[groups_in_category_array[1]],
                                              data_analysis[groups_in_category_array[2]], data_analysis[groups_in_category_array[3]])[1]
        # perform bartlett's test
        alpha_bartlett = 0.05  # 95% confidence
        # compare p value with alpha (0.05). If the p value is lower than alpha, the variances are not equal
        if p_value_bartlett > alpha_bartlett:
            print('The variances are equal')
            variances_equal = 'Yes'
            # if the variances are equal, perform one-way ANOVA
            print("Performing one-way ANOVA")
            anova_method = 'One-way ANOVA'

            # perform one-way ANOVA
            alpha_one_way_anova = 0.05  # 95% confidence

            if len(groups_in_category_array) == 3:
                # perform one-way ANOVA
                print(stats.f_oneway(data_analysis[groups_in_category_array[0]],
                      data_analysis[groups_in_category_array[1]], data_analysis[groups_in_category_array[2]]))
                # save the p-value from the test
                pvalue_f_oneway = stats.f_oneway(
                    data_analysis[groups_in_category_array[0]], data_analysis[groups_in_category_array[1]], data_analysis[groups_in_category_array[2]])[1]

            elif len(groups_in_category_array) == 4:
                # perform one-way ANOVA
                print(stats.f_oneway(data_analysis[groups_in_category_array[0]], data_analysis[groups_in_category_array[1]],
                      data_analysis[groups_in_category_array[2]], data_analysis[groups_in_category_array[3]]))
                # save the p-value from the test
                pvalue_f_oneway = stats.f_oneway(data_analysis[groups_in_category_array[0]], data_analysis[groups_in_category_array[1]],
                                                 data_analysis[groups_in_category_array[2]], data_analysis[groups_in_category_array[3]])

            # # calculate the total degrees of freedom of the f one-way anova
            # # calculate the total number of observations over all groups
            total_number_of_observations = 0
            for group_i in groups_in_category_array:
                total_number_of_observations += len(data_analysis[group_i])
            # calculate all the degrees of freedom of the f one-way anova
            df_between_groups = len(groups_in_category_array) - 1
            df_within_groups = total_number_of_observations - \
                len(groups_in_category_array)
            df_total = total_number_of_observations - 1

            # save the p-value from the test
            p_value_anova_method = pvalue_f_oneway

            # compare p value with alpha (0.05). If the p value is lower than alpha, the means are not equal
            if pvalue_f_oneway > alpha_one_way_anova:
                print('The means are equal')
                mean_different = 'No'
                type_of_post_hoc_test = 'None'
                p_value_matrix = 'None'
                significance_matrix = 'None'

            elif pvalue_f_oneway <= alpha_one_way_anova:
                print('The means are not equal')
                mean_different = 'Yes'
                print("Performing Tukey's test")
                type_of_post_hoc_test = "Tukey's test"

                # perform Tukey's test
                # check number of groups
                if len(groups_in_category_array) == 3:
                    # perform Tukey's post-hoc test
                    print(stats.tukey_hsd(data_analysis[groups_in_category_array[0]],
                          data_analysis[groups_in_category_array[1]], data_analysis[groups_in_category_array[2]]))
                    # save the p-value from the test
                    results = stats.tukey_hsd(
                        data_analysis[groups_in_category_array[0]], data_analysis[groups_in_category_array[1]], data_analysis[groups_in_category_array[2]])

                elif len(groups_in_category_array) == 4:
                    # perform Tukey's post-hoc test
                    print(stats.tukey_hsd(data_analysis[groups_in_category_array[0]], data_analysis[groups_in_category_array[1]],
                          data_analysis[groups_in_category_array[2]], data_analysis[groups_in_category_array[3]]))
                    # save the p-value from the test
                    results = stats.tukey_hsd(data_analysis[groups_in_category_array[0]], data_analysis[groups_in_category_array[1]],
                                              data_analysis[groups_in_category_array[2]], data_analysis[groups_in_category_array[3]])

                # save the p-value from the test
                pvalue_matrix = results.pvalue

                alpha_tukey = 0.05  # 95% confidence
                tukey_table_significance = create_table_with_significance_using_pvalue(
                    pvalue_matrix, groups_in_category_array, alpha_tukey)
                # save the p-value matrix and significance matrix
                p_value_matrix = pvalue_matrix
                significance_matrix = tukey_table_significance

                # convert the p-value matrix to a dataframe with the same index and columns as the significance matrix
                p_value_matrix_df = pd.DataFrame(
                    p_value_matrix, index=significance_matrix.index, columns=significance_matrix.columns)

                # save the p-value matrix and significance matrix to a csv file
                p_value_matrix_df.to_csv(
                    f'{path_folder_results}/Tukey_p_value_matrix_{experiment}_{category}.csv', index=True)
                significance_matrix.to_csv(
                    f'{path_folder_results}/Tukey_significance_matrix_{experiment}_{category}.csv', index=True)

        elif p_value_bartlett <= alpha_bartlett:
            variances_equal = 'No'
            print('The variances are not equal')
            # if the variances are not equal, perform Welch's ANOVA
            print("Performing Welch's ANOVA")
            anova_method = "Welch's ANOVA"

            # check number of groups
            if len(groups_in_category_array) == 3:
                # create a dataframe with the values of the groups
                df = pd.DataFrame({'value': np.concatenate((data_analysis[groups_in_category_array[0]], data_analysis[groups_in_category_array[1]], data_analysis[groups_in_category_array[2]])), 'group': np.concatenate((np.repeat(groups_in_category_array[0], len(
                    data_analysis[groups_in_category_array[0]])), np.repeat(groups_in_category_array[1], len(data_analysis[groups_in_category_array[1]])), np.repeat(groups_in_category_array[2], len(data_analysis[groups_in_category_array[2]]))))})
            elif len(groups_in_category_array) == 4:
                # create a dataframe with the values of the groups
                df = pd.DataFrame({'value': np.concatenate((data_analysis[groups_in_category_array[0]], data_analysis[groups_in_category_array[1]], data_analysis[groups_in_category_array[2]], data_analysis[groups_in_category_array[3]])), 'group': np.concatenate((np.repeat(groups_in_category_array[0], len(
                    data_analysis[groups_in_category_array[0]])), np.repeat(groups_in_category_array[1], len(data_analysis[groups_in_category_array[1]])), np.repeat(groups_in_category_array[2], len(data_analysis[groups_in_category_array[2]])), np.repeat(groups_in_category_array[3], len(data_analysis[groups_in_category_array[3]]))))})

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
            # check if the p-value is smaller than the alpha. If it is, the mean values are different
            if p_value_welch_anova > alpha_welch_anova:
                mean_different = 'No'
                print('The mean values are equal')
                type_of_post_hoc_test = 'None'
                p_value_matrix = 'None'
                Significance_matrix = 'None'

            # if the mean values are different, perform pairwise comparisons
            elif p_value_welch_anova <= alpha_welch_anova:
                mean_different = 'Yes'
                print('The mean values are not equal')
                type_of_post_hoc_test = 'Games-Howell'
                # perform THE pairwise games-howell post hoc test
                print(pg.pairwise_gameshowell(
                    data=df, dv='value', between='group'))

                # create a table with the pairwise comparisons
                pairwise_comparisons = pg.pairwise_gameshowell(
                    data=df, dv='value', between='group')
                alpha_pairwise_comparisons = 0.05  # 95% confidence
                table_with_p_values, p_values_with_significance = create_table_with_pairwise_comparisons(
                    pairwise_comparisons, alpha_pairwise_comparisons)

                # save the table with the p values in a csv file data frame
                table_with_p_values.to_csv(
                    f'{path_folder_results}/GamesHowell_table_with_p_values_{experiment}_{category}.csv', index=True)
                # save the table with the p values with significance in a csv file data frame
                p_values_with_significance.to_csv(
                    f'{path_folder_results}/GamesHowell_p_values_with_significance_{experiment}_{category}.csv', index=True)

                # save the data in the logger
                p_value_matrix = table_with_p_values
                significance_matrix = p_values_with_significance

            p_value_anova_method = p_value_welch_anova
        try:
            # round the total degrees of freedom to 2 decimal places
            df_between_groups = round(df_between_groups, 2)
            df_within_groups = round(df_within_groups, 2)
            df_total = round(df_total, 2)

            # create a row with the results
            row = [category, np.array(groups_in_category_array), p_value_bartlett, variances_equal, anova_method, df_between_groups, df_within_groups,
                   p_value_anova_method, mean_different, type_of_post_hoc_test, p_value_matrix, significance_matrix]

            # add the row to  results_logger dataframe
            results_logger.loc[len(results_logger)] = row
        except:
            print(
                f"Error with appending the row to the results_logger dataframe for the category {category}")
            # stop the execution of the code
            sys.exit()

    # save the results in a csv file
    results_logger.to_csv(
        f'{path_folder_results}/results_logger_{experiment}.csv', index=False)
    # save the snr metrics in a csv file
    snr_metrics.to_csv(
        f'{path_folder_results}/snr_metrics_{experiment}.csv', index=False)


def process_data_logger(data_logger: dict):
    """ Processes the data_logger and performs the statistical analysis.
    Does the pairwise comparisons between the groups of the overlapping categories of the two experiments.

    Args:
        data_logger (dict): dictionary with the data of the two experiments
    """
    headers = ['category', 'groups',  'p_value_bartlett', 'Variances equal?', 'ANOVA method', 'df within groups', 'df between groups',
               'p_value ANOVA method', 'Mean different?', 'Type of post hoc test',  'p_value_matrix', 'Significance_matrix']
    # create an empty dataframe to store the results of each step of the analysis
    results_logger = pd.DataFrame(columns=headers)

    # extract the unique categories from the data_logger
    unique_categories_experiment_1 = list(
        data_logger['Experiment_1']['category'].keys())
    unique_categories_experiment_2 = list(
        data_logger['Experiment_2']['category'].keys())
    # find the overlapping categories between the two experiments
    unique_categories_all = list(set(unique_categories_experiment_1).intersection(
        unique_categories_experiment_2))
    n_number_of_experiments = 2
    for category in unique_categories_all:
        # extract the groups from the data_logger
        groups_experiment_1 = list(
            data_logger['Experiment_1']['category'][category].keys())
        groups_experiment_2 = list(
            data_logger['Experiment_2']['category'][category].keys())
        # find the overlapping groups between the two experiments
        groups_all = list(
            set(groups_experiment_1).intersection(groups_experiment_2))

        # create a dictionary containing the data of the groups from A to B
        data_experiment_1 = {}
        data_experiment_2 = {}
        for group in groups_all:
            data_experiment_1[group] = data_logger['Experiment_1']['category'][category][group]
            data_experiment_2[group] = data_logger['Experiment_2']['category'][category][group]
            print('----------------------------------------------------------------------------------------------------------------')
            print(f'Category: {category}')
            print('----------------------------------------------------------------------------------------------------------------')
            print(f"groups all: {groups_all}")

            # perform bartlett's test
            alpha_bartlett = 0.05  # 95% confidence
            # check number of groups

            # perform Bartlett's test for the group between the two experiments
            print(stats.bartlett(
                data_experiment_1[group]['data'], data_experiment_2[group]['data']))

            p_value_bartlett = stats.bartlett(
                data_experiment_1[group]['data'], data_experiment_2[group]['data'])[1]

            # if the p value is less than the alpha, the variances are not equal
            if p_value_bartlett > alpha_bartlett:
                print('The variances are equal')
                variances_equal = 'Yes'
                # if the variances are equal, perform one-way ANOVA
                print("Performing one-way ANOVA")
                anova_method = 'One-way ANOVA'

                # perform one-way ANOVA
                print(stats.f_oneway(
                    data_experiment_1[group]['data'], data_experiment_2[group]['data']))
                pvalue_f_oneway = stats.f_oneway(
                    data_experiment_1[group]['data'], data_experiment_2[group]['data'])[1]

                alpha_one_way_anova = 0.05  # 95% confidence

                # calculate the total number of observations over all groups
                total_number_of_observations = 0
                total_number_of_observations += len(
                    data_experiment_1[group]['data'])
                total_number_of_observations += len(
                    data_experiment_2[group]['data'])

                # calculate all the degrees of freedom of the f one-way anova
                df_between_groups = n_number_of_experiments - 1
                df_within_groups = total_number_of_observations - n_number_of_experiments
                df_total = total_number_of_observations - 1

                # save the p value of the one-way ANOVA
                p_value_anova_method = pvalue_f_oneway
                # if the p value is less than the alpha, the means are not equal
                if pvalue_f_oneway > alpha_one_way_anova:
                    print('The means are equal')
                    mean_different = 'No'
                    type_of_post_hoc_test = 'None'
                    p_value_matrix = 'None'
                    significance_matrix = 'None'

                elif pvalue_f_oneway <= alpha_one_way_anova:
                    print('The means are not equal')
                    mean_different = 'Yes'
                    print("Performing Tukey's test")
                    anova_method = 'Tukey\'s test'

                    # perform Tukey's post-hoc test
                    print(stats.tukey_hsd(
                        data_experiment_1[group]['data'], data_experiment_2[group]['data']))
                    # save the p-value from the test
                    results = stats.tukey_hsd(
                        data_experiment_1[group]['data'], data_experiment_2[group]['data'])

                    pvalue_matrix = results.pvalue
                    # Alpha is the significance level at which we reject the null hypothesis
                    alpha_tukey = 0.05  # 95% confidence
                    label_groups = [
                        f"Experiment 1: {group}", f"Experiment 2: {group}"]
                    # create a dataframe determining the significance of the pairwise comparisons
                    tukey_table_significance = create_table_with_significance_using_pvalue(
                        pvalue_matrix, label_groups, alpha_tukey)

                    # log the results of the analysis
                    p_value_matrix = pvalue_matrix
                    significance_matrix = tukey_table_significance
                    # save the matrices of the analysis
                    p_value_matrix_df.to_csv(
                        f'{path_folder_results}/Tukey_p_value_matrix_{category}_{group}.csv', index=True)
                    significance_matrix.to_csv(
                        f'{path_folder_results}/Tukey_significance_matrix_{category}_{group}.csv', index=True)

            # if the variances are not equal, perform Welch's ANOVA
            elif p_value_bartlett <= alpha_bartlett:
                variances_equal = 'No'
                print('The variances are not equal')
                # if the variances are not equal, perform Welch's ANOVA
                print("Performing Welch's ANOVA")
                anova_method = "Welch's ANOVA"
                # create a dataframe containing the data from the two experiments and the group name for welch's ANOVA
                df = pd.DataFrame({'value': np.concatenate((data_experiment_1[group]['data'], data_experiment_2[group]['data'])),
                                   'group': np.concatenate((np.repeat(f"Experiment 1: {group}", len(data_experiment_1[group]['data'])),
                                                            np.repeat(f"Experiment 2: {group}", len(data_experiment_2[group]['data']))))})

                # perform Welch's ANOVA
                print(pg.welch_anova(data=df, dv='value', between='group'))
                # PERFORM WELCH'S ANOVA USING PINGOUIN  and save the results in a variable
                results = pg.welch_anova(data=df, dv='value', between='group')
                # extract df between groups from the results
                df_between_groups = results['ddof1'][0]
                # extract the degrees of freedom within groups from the results
                df_within_groups = results['ddof2'][0]
                df_total = df_between_groups + df_within_groups

                # extract the degrees of freedom from the results which
                p_value_welch_anova = results['p-unc'][0]
                alpha_welch_anova = 0.05  # 95% confidence

                if p_value_welch_anova > alpha_welch_anova:
                    mean_different = 'No'
                    print('The mean values are equal')
                    type_of_post_hoc_test = 'None'
                    p_value_matrix = 'None'
                    significance_matrix = 'None'

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

                    # save the table with the p values in a csv file data frame
                    table_with_p_values.to_csv(
                        f'{path_folder_results}/GamesHowell_table_with_p_values_{category}_{group}.csv', index=True)
                    # save the table with the p values with significance in a csv file data frame
                    p_values_with_significance.to_csv(
                        f'{path_folder_results}/GamesHowell_p_values_with_significance_{category}_{group}.csv', index=True)
                    p_value_matrix = table_with_p_values
                    significance_matrix = p_values_with_significance

                # save the p-value from the test
                p_value_anova_method = p_value_welch_anova

            try:
                # round the total degrees of freedom to 2 decimal places
                df_between_groups = round(df_between_groups, 2)
                df_within_groups = round(df_within_groups, 2)
                df_total = round(df_total, 2)

                # create a row with the results of the analysis
                row = [category, group, p_value_bartlett, variances_equal, anova_method, df_between_groups, df_within_groups,  p_value_anova_method,
                       mean_different, type_of_post_hoc_test, p_value_matrix, significance_matrix]

                # add the row to  results_logger dataframe
                results_logger.loc[len(results_logger)] = row
            except:
                print(
                    f" Error in {category} {group} with {anova_method} and trying to save the results in the results_logger dataframe")
                sys.exit()

        # save the results in a csv file
        results_logger.to_csv(
            f'{path_folder_results}/results_logger_exp1_vs_exp2.csv', index=False)


process_data_logger(data_logger)
