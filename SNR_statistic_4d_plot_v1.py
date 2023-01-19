import numpy as np
import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from collections import OrderedDict


def create_combinations_of_categories(categories: np.ndarray) -> list:
    """ Create all possible combinations of the categories

    Args:
        categories (np.ndarray): The categories to be combined

    Returns:
        list: All possible combinations of the categories
    """
    # create all possible combinations of the categories with the main category being frequency and two subcategories
    main_category = 'frequency'
    # filter out the categories that or not the main category
    subcategories = [
        category for category in categories if category != main_category]
    # find all possbile combinations between the categories
    combinations_of_categories = []
    for i in range(len(subcategories)):
        for j in range(i+1, len(subcategories)):
            combinations_of_categories.append(
                [main_category, subcategories[i], subcategories[j]])

    return combinations_of_categories


def create_combinations_of_groups(dictionary_of_categories_with_groups: dict) -> list:
    """ Create all possible combinations of the groups

    Args:
        dictionary_of_categories_with_groups (dict): The dictionary of categories with groups

    Returns:
        list: All possible combinations of the groups
    """
    # unpack the categories keys from the dictionary
    categories = list(
        dictionary_of_categories_with_groups['categories'].keys())
    # create all possible combinations of the groups between the categories
    group_settings_of_categories = dictionary_of_categories_with_groups['categories']
    # category 1 is the main category and category 2 and 3 are the subcategories
    # find all possbile combinations between the groups of the categories
    combinations_of_groups = []
    for group_1 in group_settings_of_categories[categories[0]]:
        for group_2 in group_settings_of_categories[categories[1]]:
            for group_3 in group_settings_of_categories[categories[2]]:
                combinations_of_groups.append([group_1, group_2, group_3])

    return combinations_of_groups


def create_4d_plot(dict_of_combinations_with_data: dict,  groups_per_experiment_dict: dict, unique_experiments: np.ndarray, path_folder_results: str):
    """ Create 3d plots for the SNR statistic

    Args:
        dict_of_combinations_with_data (dict): Dictionary of combinations with data
        groups_per_experiment_dict (dict): Dictionary of groups per experiment
        unique_experiments (np.ndarray): Unique experiments
        path_folder_results (str):  Path to the folder where the results are saved
    """
    # get the name of the experiments
    experiments = list(dict_of_combinations_with_data.keys())
    for experiment in experiments:
        # unpack the dict of combinations with data for the experiment
        experiment_data = dict_of_combinations_with_data[experiment]
        # find the keys in the dict of combinations with data
        combinations_categories = list(experiment_data.keys())
        for combination_of_categories in combinations_categories:
            # split the combination of categiries by '-'
            combination_of_categories_seperated = combination_of_categories.split(
                '-')
            # unpack the dict of combinations with data for the combination of categories except the key categories
            combination_of_categories_data = experiment_data[combination_of_categories]
            # unpacking values for x axis from the categories dict
            three_categories = combination_of_categories_data['categories']
            three_categories_keys = list(three_categories.keys())
            # list all the keys in the dict of combinations with data for the combination of categories
            combinations_groups = list(combination_of_categories_data.keys())
            # filter out the key categories
            combinations_groups = [
                combination for combination in combinations_groups if combination != 'categories']
            # types of frequencies
            frequencies = three_categories[three_categories_keys[0]]
            # create a figure
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            # label the x axis SNR
            ax.set_zlabel('SNR [dB]')

            # set the x axis ticks
            if three_categories_keys[1] == 'pixel_surface':
                ax.set_xlabel('pixel surface [pixels]')
                x_values_axis = [10000, 20000, 30000]
                # set the x axis ticks
                ax.set_xticks(x_values_axis)
                str_add = 'pixels'

            elif three_categories_keys[1] == 'color':
                ax.set_xlabel('color')
                # count the number of colors
                number_of_colors = len(
                    three_categories[three_categories_keys[1]])
                xticks = np.arange(number_of_colors)
                # set the x axis ticks
                x_values_axis = three_categories[three_categories_keys[1]]
                ax.set_xticks(xticks)
                ax.set_xticklabels(x_values_axis)
                str_add = ''

            elif three_categories_keys[1] == 'shape':
                ax.set_xlabel('shape')
                # count the number of shapes
                number_of_shapes = len(
                    three_categories[three_categories_keys[1]])
                xticks = np.arange(number_of_shapes)
                # set the x axis ticks
                x_values_axis = three_categories[three_categories_keys[1]]
                ax.set_xticks(xticks)
                ax.set_xticklabels(x_values_axis)
                str_add = ''

            # label the y axis
            if three_categories_keys[2] == 'pixel_surface':
                # label the y axis
                ax.set_ylabel('pixel surface [pixels]')
                y_values_axis = [10000, 20000, 30000]
                # set the y axis ticks
                ax.set_yticks(y_values_axis)
                str_add = 'pixels'

            elif three_categories_keys[2] == 'color':
                # label the y axis
                ax.set_ylabel('color')
                # count the number of colors
                number_of_colors = len(
                    three_categories[three_categories_keys[2]])
                yticks = np.arange(number_of_colors)
                y_values_axis = three_categories[three_categories_keys[2]]
                # set the y axis ticks
                ax.set_yticks(yticks)
                ax.set_yticklabels(y_values_axis)
                str_add = ''

            elif three_categories_keys[2] == 'shape':
                # label the y axis
                ax.set_ylabel('shape')
                # count the number of shapes
                number_of_shapes = len(
                    three_categories[three_categories_keys[2]])
                yticks = np.arange(number_of_shapes)
                y_values_axis = three_categories[three_categories_keys[2]]
                # set the y axis ticks
                ax.set_yticks(yticks)
                ax.set_yticklabels(y_values_axis)
                str_add = ''

            # create a look up dict for the x axis
            if three_categories_keys[2] != 'pixel_surface':
                look_up_dict_y_axis = {}
                for i in range(len(y_values_axis)):
                    look_up_dict_y_axis[y_values_axis[i]] = i
            # create a look up dict for the y axis
            if three_categories_keys[1] != 'pixel_surface':
                look_up_dict_x_axis = {}
                for i in range(len(x_values_axis)):
                    look_up_dict_x_axis[x_values_axis[i]] = i

            # for each combination of settings in the combination of categories
            for combination_of_settings in combinations_groups:
                # split the combination of settings by '-'
                combination_of_settings_seperated = combination_of_settings.split(
                    '-')
                # unpack the data for the combination of settings
                x_value = combination_of_settings_seperated[1]
                y_value = combination_of_settings_seperated[2]
                # try to convert the x value to an int
                try:
                    x_value = int(x_value)
                except ValueError:
                    # look up the x value in the look up dict
                    x_value = look_up_dict_x_axis[x_value]
                # try to convert the y value to an int
                try:
                    y_value = int(y_value)
                except ValueError:
                    # look up the y value in the look up dict
                    y_value = look_up_dict_y_axis[y_value]

                # for each frequency use a different color and label
                if int(combination_of_settings_seperated[0]) == frequencies[0]:
                    color = 'red'
                    label = 'frequency = ' + str(frequencies[0]) + ' Hz'
                elif int(combination_of_settings_seperated[0]) == frequencies[1]:
                    color = 'blue'
                    label = 'frequency = ' + str(frequencies[1]) + ' Hz'
                    y_value = y_value + 0.15
                elif int(combination_of_settings_seperated[0]) == frequencies[2]:
                    color = 'green'
                    label = 'frequency = ' + str(frequencies[2]) + ' Hz'
                    y_value = y_value + 0.30
                elif int(combination_of_settings_seperated[0]) == frequencies[3]:
                    color = 'black'
                    label = 'frequency = ' + str(frequencies[3]) + ' Hz'
                    y_value = y_value + 0.45

                # unpack the dict of combinations with data for the combination of settings
                combination_of_settings_data = combination_of_categories_data[
                    combination_of_settings]
                # unpack the mean snr and std from the dict of combinations with data for the combination of settings
                snr = combination_of_settings_data['mean']
                std = combination_of_settings_data['std']
                z_value = snr
                # show mean snr as bar
                ax.bar3d(x_value, y_value, 0, 0.01, 0.1, snr,
                         color=color, alpha=0.35, label=label)
                # show std  error bars
                ax.plot([x_value, x_value], [y_value, y_value], [
                        z_value-std, z_value+std], color=color, alpha=1.0, marker='_', label=label)

            # set zlim between -3 and 15
            ax.set_zlim(-3, 15)
            ax.view_init(35, 136)
            # get the legend handles and labels
            handles, labels = ax.get_legend_handles_labels()
            # create a dict of the legend handles and labels
            by_label = OrderedDict(zip(labels, handles))
            # set the legend
            plt.legend(by_label.values(), by_label.keys(
            ), loc='upper left', bbox_to_anchor=(0.7, 1.0), borderaxespad=0.)
            experiment = experiment.replace('_', ' ')
            category_1 = three_categories_keys[0].replace('_', ' ')
            category_2 = three_categories_keys[1].replace('_', ' ')
            category_3 = three_categories_keys[2].replace('_', ' ')
            # set the title of the plot
            plt.title(experiment + ': ' + category_1 +
                      ' vs ' + category_2 + ' vs ' + category_3)
            # make sure legend fits on tight layout
            plt.tight_layout()
            # save the plot
            plt.savefig(path_folder_results + '/' + experiment + '_' +
                        three_categories_keys[0] + '_' + three_categories_keys[1] + '_' + three_categories_keys[2] + str_add + '.png')


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)
    # check if in kwargs is a max value for the colorbar
    if 'vmax' in kwargs:
        # if so, set the colorbar to that value
        im.set_clim(0, kwargs['vmax'])
    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def create_boxplot(dict_of_combinations_with_data: dict,  groups_per_experiment_dict: dict, unique_experiments: np.ndarray, path_folder_results: str):
    """ Create 3d plots for the SNR statistic

    Args:
        dict_of_combinations_with_data (dict): Dictionary of combinations with data
        groups_per_experiment_dict (dict): Dictionary of groups per experiment
        unique_experiments (np.ndarray): Unique experiments
        path_folder_results (str):  Path to the folder where the results are saved
    """
    # get the name of the experiments
    experiments = list(dict_of_combinations_with_data.keys())

    for experiment in experiments:
        # unpack the dict of combinations with data for the experiment
        experiment_data = dict_of_combinations_with_data[experiment]
        # find the keys in the dict of combinations with data
        combinations_categories = list(experiment_data.keys())

        for combination_of_categories in combinations_categories:
            # split the combination of categiries by '-'
            combination_of_categories_seperated = combination_of_categories.split(
                '-')
            # unpack the dict of combinations with data for the combination of categories except the key categories
            combination_of_categories_data = experiment_data[combination_of_categories]

            # unpacking values for x axis from the categories dict
            three_categories = combination_of_categories_data['categories']
            three_categories_keys = list(three_categories.keys())
            # list all the keys in the dict of combinations with data for the combination of categories
            combinations_groups = list(combination_of_categories_data.keys())
            # filter out the key categories
            combinations_groups = [
                combination for combination in combinations_groups if combination != 'categories']

            # types of frequencies
            frequencies = three_categories[three_categories_keys[0]]
            for frequency in frequencies:
                # create a figure with 3 subplots
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

                # extract the groups of the last category
                groups = groups_per_experiment_dict[experiment][three_categories_keys[2]]

                # for each subplot set the groups variables as title
                ax1.set_title(str(groups[0]))
                ax2.set_title(str(groups[1]))
                ax3.set_title(str(groups[2]))

                # set the title of the plot
                # remove the "_" from experiment name
                experiment_title = experiment.replace('_', ' ')
                combination_of_categories_title = combination_of_categories.replace(
                    '_', ' ')

                title = 'Boxplot: ' + experiment_title + ': ' + three_categories_keys[0] + ' = ' + str(
                    frequency) + ' Hz, ' + combination_of_categories_title + ' without outliers'
                fig.suptitle(title)
                print("Creating boxplot for: ", title)

                # set the x axis ticks
                if three_categories_keys[1] == 'pixel_surface':
                    ax1.set_xlabel('pixel surface [pixels]')
                    ax2.set_xlabel("pixel surface [pixels]")
                    ax3.set_xlabel("pixel surface [pixels]")
                    # count the number of pixel surfaces
                    number_of_sizes = len(
                        three_categories[three_categories_keys[1]])
                    xticks = np.arange(number_of_sizes)
                    # set the x axis ticks labels
                    x_values_axis = three_categories[three_categories_keys[1]]
                    x_values_axis = [str(x) for x in x_values_axis]
                    str_add = 'pixels'

                elif three_categories_keys[1] == 'color':
                    ax1.set_xlabel('color')
                    ax2.set_xlabel('color')
                    ax3.set_xlabel('color')
                    # count the number of colors
                    number_of_colors = len(
                        three_categories[three_categories_keys[1]])
                    xticks = np.arange(number_of_colors)
                    # set the x axis ticks labels
                    x_values_axis = three_categories[three_categories_keys[1]]
                    x_values_axis = [str(x) for x in x_values_axis]
                    str_add = ''

                elif three_categories_keys[1] == 'shape':
                    ax1.set_xlabel('shape')
                    ax2.set_xlabel('shape')
                    ax3.set_xlabel('shape')
                    # count the number of shapes
                    number_of_shapes = len(
                        three_categories[three_categories_keys[1]])
                    xticks = np.arange(number_of_shapes)
                    # set the x axis ticks labels
                    x_values_axis = three_categories[three_categories_keys[1]]
                    x_values_axis = [str(x) for x in x_values_axis]
                    str_add = ''

                # filter the combinations groups for the target frequency
                filtered_combinations_groups_target_frequency = [
                    combination for combination in combinations_groups if combination.split('-')[0] == str(frequency)]
                for combination_of_settings in filtered_combinations_groups_target_frequency:
                    # split the combination of settings by '-'
                    combination_of_settings_seperated = combination_of_settings.split(
                        '-')
                    x_value_lookup = combination_of_settings_seperated[1]
                    # find the index of the x value in the x values axis
                    x_values_axis = np.array(x_values_axis)
                    x_value_index = np.where(
                        x_value_lookup == x_values_axis)[0][0]
                    x_values_axis = list(x_values_axis)
                    # get the look up key for the current combination of settings to determine the correct subplot
                    look_up_key = combination_of_settings_seperated[2]
                    plot_number = np.where(look_up_key == groups)[0][0]

                    # set the correct subplot
                    if plot_number == 0:
                        ax = ax1
                    elif plot_number == 1:
                        ax = ax2
                    elif plot_number == 2:
                        ax = ax3

                    # get the data for the current combination of settings
                    dict_of_combination = dict_of_combinations_with_data[experiment][
                        combination_of_categories][combination_of_settings]

                    # get the data
                    data = dict_of_combination['data']
                    mean = dict_of_combination['mean']
                    std = dict_of_combination['std']

                    # create the boxplot
                    bplot = ax.boxplot(data, positions=[x_value_index], widths=0.6, showmeans=True, meanline=True,
                                       showfliers=False, patch_artist=True, medianprops=dict(color='blue'), meanprops=dict(color='red'))
                    # set the color of the boxplot to be filled the color light blue
                    for patch in bplot['boxes']:
                        patch.set_facecolor('lightblue')
                    # extract the participant specific data
                    participant_specific_data = dict_of_combination['participant_specific_data']
                    # extract the participant specific keys
                    participant_specific_keys = list(
                        participant_specific_data.keys())

                    # create a list of colors for the legend
                    colors = ['green', 'purple',
                              'brown', 'grey', 'olive', 'cyan']
                    # create a list of labels for the legend
                    labels = []
                    # create a list of handles for the legend
                    handles = []

                    # measure the number of participants and divide 1 by the number of participants to get the stepsize in the x axis
                    stepsize = 1 / len(participant_specific_keys)

                    # sort the participant specific keys in ascending order
                    participant_specific_keys = sorted(
                        participant_specific_keys, key=lambda x: int(x.split('pp')[1]))
                    for i, participant in enumerate(participant_specific_keys):
                        # remove the pp from the participant key and correct it. pp5 is now pp1
                        participant_number = int(participant[2:])
                        # get the participant specific data
                        participant_data = participant_specific_data[participant]
                        # get the mean and std of the participant
                        participant_mean = np.mean(participant_data)
                        participant_std = np.std(participant_data)

                        # create a label for the legend
                        label = f'pp {participant_number} mean'
                        # add the label to the list of labels
                        labels.append(label)
                        # create a handle for the legend
                        handles.append(plt.Line2D(
                            [0], [0], color=colors[i], lw=4))

                        # show the lines of the mean of each participant in the boxplot within each box with an alpha of 0.5 at the correct position with respect to the x axis
                        ax.plot([x_value_index-0.5 + i*stepsize, x_value_index - 0.4 + i*stepsize], [
                                participant_mean, participant_mean], color=colors[i], lw=3)

                    # set the x axis ticks
                    ax.set_ylabel('SNR [dB]')
                    ax.yaxis.grid(True)

                    # create vertical lines at the x axis ticks positions of the x axis
                    for x in xticks:
                        ax.axvline(x-0.5, color='black',
                                   linestyle='-', linewidth=0.5)

                    # create a legend for the mean which is color red and the median which is color blue
                    labels.append(f'mean of all data')
                    labels.append(f'median of all data')

                    # create handles for the legend
                    # create a red line for the mean
                    handles.append(plt.Line2D([0], [0], color='red', lw=4))
                    # create a blue line for the median
                    handles.append(plt.Line2D([0], [0], color='blue', lw=4))

                    # use subplots adjust to set the space between the subplots
                    plt.subplots_adjust(wspace=1.2)
                    # create a custom legend with the labels and the colors for each subplot located exactly in the upper right corner
                    ax.legend(handles, labels, loc='upper right',
                              bbox_to_anchor=(1.9, 1.0))

                # set the x axis ticks and labels
                plt.setp([ax1, ax2, ax3], xticks=xticks,
                         xticklabels=x_values_axis)

                # save the plt image
                plt.savefig(path_folder_results + '/boxplot_' + experiment + '_' +
                            combination_of_categories + '_' + str(frequency) + 'Hz' + '.png', dpi=1000, bbox_inches='tight')


def create_heatmap_plot(dict_of_combinations_with_data: dict,  groups_per_experiment_dict: dict, unique_experiments: np.ndarray, path_folder_results: str):
    """ Create 3d plots for the SNR statistic

    Args:
        dict_of_combinations_with_data (dict): Dictionary of combinations with data
        groups_per_experiment_dict (dict): Dictionary of groups per experiment
        unique_experiments (np.ndarray): Unique experiments
        path_folder_results (str):  Path to the folder where the results are saved
    """
    # get the name of the experiments
    experiments = list(dict_of_combinations_with_data.keys())

    max_value_mean_snr = 0
    max_value_std_snr = 0

    for experiment in experiments:
        # unpack the dict of combinations with data for the experiment
        experiment_data = dict_of_combinations_with_data[experiment]
        # find the keys in the dict of combinations with data
        combinations_categories = list(experiment_data.keys())

        for combination_of_categories in combinations_categories:
            # split the combination of categiries by '-'
            combination_of_categories_seperated = combination_of_categories.split(
                '-')
            # unpack the dict of combinations with data for the combination of categories except the key categories
            combination_of_categories_data = experiment_data[combination_of_categories]

            # unpacking values for x axis from the categories dict
            three_categories = combination_of_categories_data['categories']
            three_categories_keys = list(three_categories.keys())
            # list all the keys in the dict of combinations with data for the combination of categories
            combinations_groups = list(combination_of_categories_data.keys())
            # filter out the key categories
            combinations_groups = [
                combination for combination in combinations_groups if combination != 'categories']
            # types of frequencies

            frequencies = three_categories[three_categories_keys[0]]
            for frequency in frequencies:
                # create two heatmaps for each frequency in the experiment. One for the mean and one for the std

                # create a figures
                fig = plt.figure()
                # create a subplot for the mean
                ax1 = fig.add_subplot(121)
                # create a subplot for the std
                ax2 = fig.add_subplot(122)
                # set the title of the plot
                # remove the "_" from experiment name
                experiment_title = experiment.replace('_', ' ')
                combination_of_categories_title = combination_of_categories.replace(
                    '_', ' ')
                title = experiment_title + ': ' + three_categories_keys[0] + ' = ' + str(
                    frequency) + ' Hz, ' + combination_of_categories_title
                plt.suptitle(title)
                # set the title of the mean plot
                ax1.set_title('Mean SNR [dB]')
                # set the title of the std plot
                ax2.set_title('Std SNR [dB]')
                # set the x axis ticks
                if three_categories_keys[1] == 'pixel_surface':
                    ax1.set_xlabel('pixel surface [pixels]')
                    ax2.set_xlabel("pixel surface [pixels]")
                    x_values_axis = [10000, 20000, 30000]
                    # set the x axis ticks
                    ax1.set_xticks(x_values_axis)
                    ax2.set_xticks(x_values_axis)
                    str_add = 'pixels'

                elif three_categories_keys[1] == 'color':
                    ax1.set_xlabel('color')
                    ax2.set_xlabel('color')
                    # count the number of colors
                    number_of_colors = len(
                        three_categories[three_categories_keys[1]])
                    xticks = np.arange(number_of_colors)
                    # set the x axis ticks
                    x_values_axis = three_categories[three_categories_keys[1]]
                    ax1.set_xticks(xticks)
                    ax2.set_xticks(xticks)
                    ax1.set_xticklabels(x_values_axis)
                    ax2.set_xticklabels(x_values_axis)
                    str_add = ''

                elif three_categories_keys[1] == 'shape':
                    ax1.set_xlabel('shape')
                    ax2.set_xlabel('shape')
                    # count the number of shapes
                    number_of_shapes = len(
                        three_categories[three_categories_keys[1]])
                    xticks = np.arange(number_of_shapes)
                    # set the x axis ticks
                    x_values_axis = three_categories[three_categories_keys[1]]
                    ax1.set_xticks(xticks)
                    ax2.set_xticks(xticks)
                    ax1.set_xticklabels(x_values_axis)
                    ax2.set_xticklabels(x_values_axis)
                    str_add = ''

                # label the y axis
                if three_categories_keys[2] == 'pixel_surface':
                    # label the y axis
                    ax1.set_ylabel('pixel surface [pixels]')
                    ax2.set_ylabel('pixel surface [pixels]')
                    y_values_axis = [10000, 20000, 30000]
                    # set the y axis ticks
                    ax1.set_yticks(y_values_axis)
                    ax2.set_yticks(y_values_axis)
                    str_add = 'pixels'

                elif three_categories_keys[2] == 'color':
                    # label the y axis
                    ax1.set_ylabel('color')
                    ax2.set_ylabel('color')
                    # count the number of colors
                    number_of_colors = len(
                        three_categories[three_categories_keys[2]])
                    yticks = np.arange(number_of_colors)
                    y_values_axis = three_categories[three_categories_keys[2]]
                    # set the y axis ticks
                    ax1.set_yticks(yticks)
                    ax2.set_yticks(yticks)
                    ax1.set_yticklabels(y_values_axis)
                    ax2.set_yticklabels(y_values_axis)
                    str_add = ''

                elif three_categories_keys[2] == 'shape':
                    # label the y axis
                    ax1.set_ylabel('shape')
                    ax2.set_ylabel('shape')
                    # count the number of shapes
                    number_of_shapes = len(
                        three_categories[three_categories_keys[2]])
                    yticks = np.arange(number_of_shapes)
                    y_values_axis = three_categories[three_categories_keys[2]]
                    # set the y axis ticks
                    ax1.set_yticks(yticks)
                    ax2.set_yticks(yticks)
                    ax1.set_yticklabels(y_values_axis)
                    ax2.set_yticklabels(y_values_axis)
                    str_add = ''

                # create a look up dict for the x axis
                look_up_dict_x_axis = {}
                for i in range(len(x_values_axis)):
                    look_up_dict_x_axis[str(x_values_axis[i])] = i
                # create a look up dict for the y axis
                look_up_dict_y_axis = {}
                for i in range(len(y_values_axis)):
                    look_up_dict_y_axis[str(y_values_axis[i])] = i

                # create an empty array for the mean values
                mean_values = np.zeros(
                    (len(x_values_axis), len(y_values_axis)))
                # create an empty array for the std values
                std_values = np.zeros((len(x_values_axis), len(y_values_axis)))

                # filter the combinations groups for the target frequency
                filtered_combinations_groups_target_frequency = [
                    combination for combination in combinations_groups if combination.split('-')[0] == str(frequency)]
                for combination_of_settings in filtered_combinations_groups_target_frequency:
                    # split the combination of settings by '-'
                    combination_of_settings_seperated = combination_of_settings.split(
                        '-')
                    # unpack the data for the combination of settings
                    x_value_index = combination_of_settings_seperated[1]
                    y_value_index = combination_of_settings_seperated[2]

                    # look up the x value in the look up dict
                    x_value = look_up_dict_x_axis[x_value_index]
                    # look up the y value in the look up dict
                    y_value = look_up_dict_y_axis[y_value_index]

                    # unpack the dict of combinations with data for the combination of settings
                    combination_of_settings_data = combination_of_categories_data[
                        combination_of_settings]
                    # unpack the mean snr and std from the dict of combinations with data for the combination of settings
                    snr_mean = combination_of_settings_data['mean']
                    snr_std = combination_of_settings_data['std']
                    # add the mean and std to the arrays
                    mean_values[x_value, y_value] = snr_mean
                    std_values[x_value, y_value] = snr_std
                    if snr_mean > max_value_mean_snr:
                        max_value_mean_snr = snr_mean
                    if snr_std > max_value_std_snr:
                        max_value_std_snr = snr_std

                # plot the mean values using the heatmap function
                image_mean, cbar_mean = heatmap(mean_values, x_values_axis, y_values_axis, ax=ax1,
                                                cmap="YlGn", cbarlabel="SNR [dB]", vmax=7)
                # annotate the heatmap with the mean values
                annotate_heatmap(image_mean, valfmt="{x:.2f}")

                # plot the std values using the heatmap function
                image_std, cbar_mean = heatmap(std_values, x_values_axis, y_values_axis, ax=ax2,
                                               cmap="YlGn", cbarlabel="SNR [dB]", vmax=9)
                # annotate the heatmap with the std values
                annotate_heatmap(image_std, valfmt="{x:.2f}")

                plt.tight_layout()
                # save the plt image
                plt.savefig(path_folder_results + '/' + 'heatmap_{experiment}_{frequency}_{category1}_{category2}.png'.format(
                    experiment=experiment, frequency=frequency, category1=three_categories_keys[1], category2=three_categories_keys[2]))
    print(f'finished plotting the heatmaps for the experiment {experiment}')
    print(f'max_value_mean_snr: {max_value_mean_snr}')
    print(f'max_value_std_snr: {max_value_std_snr}')


def convert_dataframe_strings_to_list_SNR(df: pd.DataFrame) -> pd.DataFrame:
    """ Convert the strings in the dataframe to list of floats

    Args:
        df (pd.DataFrame): dataframe with the strings to convert

    returns:
        df_new (pd.DataFrame): dataframe with the converted strings
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


def preprocess_analysis(data: pd.DataFrame, combinations_categories: list, groups_per_experiment_dict: dict, unique_experiments: np.ndarray, path_folder_results: str) -> dict:
    """ Preprocess the data for the analysis by creating a dict with the combinations of categories and the groups of each experiment. Data is also saved, the mean snr and std snr of each group is calculated and saved in a table.

    Args:
        data (pd.DataFrame): The data to be preprocessed of the experiments.
        combinations_categories (list): The combinations of categories to be analyzed.
        groups_per_experiment_dict (dict): Dictionary with the groups of each experiment with respect to each category.
        unique_experiments (np.ndarray): The unique experiments.
        path_folder_results (str): The path to the folder where the results will be saved.

    Returns:
        dict: A dict with the combinations of categories and the groups of each experiment according with the data extracted from the data.
    """
    # create a dict to store the results
    combination_dict = {}
    # create a list to store the headers of the table
    headers = ['experiment', 'combination', 'group',
               'SNR mean', 'SNR std', 'number of samples']
    row = []
    # iterate over the unique experiments (folder)
    for experiment in unique_experiments:
        variables_dict = {}

        # remove the combinations with 'shape' if the experiment is 'Experiment_2'
        if experiment == 'Experiment_2':
            combinations_categories_filtered = [
                combination for combination in combinations_categories if 'shape' not in combination]
        else:
            combinations_categories_filtered = combinations_categories

        for combination in combinations_categories_filtered:
            # create a unique string for the combination of categories
            combination_string = '-'.join(combination)
            # create an empty dict for the combination of categories
            variables_dict[combination_string] = {}
            # iterate over the groups in each category
            variables_dict[combination_string]['categories'] = {}
            for category in combination:
                variables_dict[combination_string]['categories'][category] = groups_per_experiment_dict[experiment][category]
            # create all combinations of groups in the combination of categories
            combinations_groups = create_combinations_of_groups(
                variables_dict[combination_string])

            # iterate over the combinations of groups
            for combination_group_setting_loaded in combinations_groups:
                # create a unique string for the combination of groups
                combination_group_setting_loaded_string = str(combination_group_setting_loaded[0]) + '-' + str(
                    combination_group_setting_loaded[1]) + '-' + str(combination_group_setting_loaded[2])
                # create an empty dict for the combination of groups
                variables_dict[combination_string][combination_group_setting_loaded_string] = {
                }

                # extract the data of the combination of groups
                data_combination_group_setting_loaded = data[(data['Folder'] == experiment) & (data[combination[0]] == combination_group_setting_loaded[0]) & (
                    data[combination[1]] == combination_group_setting_loaded[1]) & (data[combination[2]] == combination_group_setting_loaded[2])]
                # extract only the SNR metrics
                data_combination_group_setting_loaded_SNR_df = data_combination_group_setting_loaded.iloc[
                    :, 6:]

                # convert the dataframe to a 1D array
                data_combination_group_setting_loaded_SNR = np.array(
                    data_combination_group_setting_loaded_SNR_df.values.tolist()).flatten()

                # save the data of the combination of groups in the dict under the scope data
                variables_dict[combination_string][combination_group_setting_loaded_string]['data'] = data_combination_group_setting_loaded_SNR
                # save the number of measurements in the dict under the scope data_length
                variables_dict[combination_string][combination_group_setting_loaded_string]['data_length'] = len(
                    data_combination_group_setting_loaded_SNR)

                # measure the mean and standard deviation of the data
                mean = np.mean(data_combination_group_setting_loaded_SNR)
                std = np.std(data_combination_group_setting_loaded_SNR)

                # save the mean and standard deviation in the dict under the scope mean and std
                variables_dict[combination_string][combination_group_setting_loaded_string]['mean'] = mean
                variables_dict[combination_string][combination_group_setting_loaded_string]['std'] = std

                # create a dict for participant specific data
                variables_dict[combination_string][combination_group_setting_loaded_string]['participant_specific_data'] = {
                }
                # get all headers of data_combination_group_setting_loaded_SNR_df (participant specific data)
                headers_participants = data_combination_group_setting_loaded_SNR_df.columns
                for participant in headers_participants:
                    # extract the data of teh specific participant from data_combination_group_setting_loaded_SNR_df
                    data_combination_group_setting_loaded_SNR_df_participant = data_combination_group_setting_loaded_SNR_df[
                        participant]
                    # convert the dataframe to a 1D array
                    data_combination_group_setting_loaded_SNR_participant = np.array(
                        data_combination_group_setting_loaded_SNR_df_participant.values.tolist()).flatten()
                    # save the data of the specific participant in the dict under the scope participant_specific
                    variables_dict[combination_string][combination_group_setting_loaded_string][
                        'participant_specific_data'][participant] = data_combination_group_setting_loaded_SNR_participant

                # create a row for the table with the results
                row.append([experiment, combination_string, combination_group_setting_loaded_string, mean, std, len(
                    data_combination_group_setting_loaded_SNR)])
                # add the row to the table with the results
        # save the dict of the combination of categories in the dict of the experiment
        combination_dict[experiment] = variables_dict
    # create a dataframe with the results
    results_df = pd.DataFrame(row, columns=headers)
    # save the dataframe with the results
    results_df.to_csv(os.path.join(path_folder_results,
                      'Combination_analysis_results.csv'), index=False)

    return combination_dict


def main():
    path = r"/media/sjoerd/BackUp Drive/Thesis_project/Data_SNR/SNR_sorted_by_participant.csv"
    # get path of folder where the results are saved
    path_folder = os.path.dirname(path)
    # create new folder named "SNR results" in the folder where the results are saved
    path_folder_results = os.path.join(path_folder, "SNR results")
    # check if the folder already exists
    if not os.path.exists(path_folder_results):
        # if not, create the folder
        os.makedirs(path_folder_results)

    # read the csv file
    data = pd.read_csv(path)

    # convert the strings in the dataframe to a list
    data = convert_dataframe_strings_to_list_SNR(data)

    # get the headers of the dataframe
    headers = data.columns.values

    # unique experiments (folder)
    unique_experiments = data['Folder'].unique()
    # unique categories
    unique_categories_all = headers[2:6]

    # create a dict with the unique groups per experiment per category
    groups_per_experiment_dict = {}
    for experiment in unique_experiments:
        groups_per_experiment_dict[experiment] = {}
        for category in unique_categories_all:
            # extract unique metrics per experiment per category
            unique_groups = data[data['Folder']
                                 == experiment][category].unique()
            # sort the unique groups
            unique_groups.sort()
            groups_per_experiment_dict[experiment][category] = unique_groups

    # create combinations of categories
    combinations_categories = create_combinations_of_categories(
        unique_categories_all)
    # create a dict with the combinations of categories and the data of the combinations of groups
    dict_of_combinations_with_data = preprocess_analysis(
        data, combinations_categories, groups_per_experiment_dict, unique_experiments, path_folder_results)
    # create a 3D and 4D plot for each combination of categories
    create_boxplot(dict_of_combinations_with_data,
                   groups_per_experiment_dict, unique_experiments, path_folder_results)
    create_heatmap_plot(dict_of_combinations_with_data,
                        groups_per_experiment_dict, unique_experiments, path_folder_results)
    create_4d_plot(dict_of_combinations_with_data,
                   groups_per_experiment_dict, unique_experiments, path_folder_results)


main()
