import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import os
from math import pi
import mne
from proces_eyes import process_eyes
import copy
from typing import Union


def weighted_average_interpolation_snr(desired_freq: float, freqs: np.ndarray, snrs: np.ndarray) -> np.ndarray:
    """ Calculate the weighted average of the SNR values of the desired frequency because it is between two frequencies in the spectrum.

    Args:
        desired_freq (float): the desired frequency
        freqs (np.ndarray): arrayt with the frequencies
        snrs (np.ndarray): array of the same dimensions as freqs with the SNR values across all the channels

    Returns:
        np.ndarray: the weighted average of the SNR values of the desired frequency across all the channels
    """
    # find bin with closest frequency to desired frequency
    i_bin = np.argmin(np.abs(freqs - desired_freq))
    # calculate the error between the desired frequency and the frequency in the bin
    error = desired_freq - freqs[i_bin]
    # if the desired frequency is in the bin, return the SNR value and perform no interpolation
    if error == 0:
        # if the desired frequency is in the bin, return the SNR value
        print(
            f" No interpolation needed, desired frequency {desired_freq} is in bin {i_bin} with frequency {freqs[i_bin]}")
        return snrs[0, :, i_bin]
    else:
        # if the desired frequency is not in the bin, calculate the weighted average of the SNR values of the two frequencies in the bin
        print(
            f" Interpolation needed, desired frequency {desired_freq} is not in bin {i_bin} with frequency {freqs[i_bin]}")

        # calculate the weight of the two frequencies in the bin
        if error > 0:
            # eror is positive, desired frequency is higher than the frequency in the bin
            print(
                f" Error is positive, desired frequency {desired_freq} is higher than frequency {freqs[i_bin]}")
            # upper_bin
            upper_bin = i_bin + 1
            # lower_bin
            lower_bin = i_bin
            # measure the difference between the desired frequency and the two frequencies in the bin
            difference_with_lower_bin = freqs[lower_bin] - desired_freq
            difference_with_upper_bin = desired_freq - freqs[upper_bin]
            # calculate the weight of the two frequencies in the bin
            weight_upper_bin = abs(difference_with_lower_bin) / (
                abs(difference_with_lower_bin) + abs(difference_with_upper_bin))
            weight_lower_bin = abs(difference_with_upper_bin) / (
                abs(difference_with_lower_bin) + abs(difference_with_upper_bin))
            print(
                f" Weight of frequency {freqs[upper_bin]} is {weight_upper_bin} and weight of frequency {freqs[lower_bin]} is {weight_lower_bin}")
            # check if the sum of the weights is 1
            assert (weight_lower_bin+weight_upper_bin == 1)

        else:
            print(
                f"Error is negative, desired frequency {desired_freq} is lower than frequency {freqs[i_bin]}")

            # upper_bin
            upper_bin = i_bin
            # lower_bin
            lower_bin = i_bin - 1
            # measure the difference between the desired frequency and the two frequencies in the bin
            difference_with_lower_bin = freqs[lower_bin] - desired_freq
            difference_with_upper_bin = desired_freq - freqs[upper_bin]
            # calculate the weight of the two frequencies in the bin
            weight_upper_bin = abs(difference_with_lower_bin) / (
                abs(difference_with_lower_bin) + abs(difference_with_upper_bin))
            weight_lower_bin = abs(difference_with_upper_bin) / (
                abs(difference_with_lower_bin) + abs(difference_with_upper_bin))
            print(
                f" Weight of frequency {freqs[upper_bin]} is {weight_upper_bin} and weight of frequency {freqs[lower_bin]} is {weight_lower_bin}")
            # check if the sum of the weights is 1
            assert (weight_lower_bin+weight_upper_bin == 1)

        # calculate new SNR value
        new_snr = (snrs[0, :, upper_bin] * weight_upper_bin) + \
            (snrs[0, :, lower_bin] * weight_lower_bin)
        print(f" New SNR value is {new_snr}")

        return new_snr


def calculate_statistics(data: np.ndarray) -> Union[float, float, float, float, float, float]:
    """Calculate the mean, average, standard deviation, variance, minimum and maximum of the data.

    Args:
        data (np.ndarray): The data to calculate the statistics for.

    Returns:
        tuple: The mean, average, standard deviation, variance, minimum and maximum of the data.
    """
    # calculate the mean, average, standard deviation, variance, minimum and maximum of the data
    data_mean = np.mean(data)
    data_average = np.average(data)
    data_std = np.std(data)
    data_variance = np.var(data)
    data_min = np.min(data)
    data_max = np.max(data)

    return data_mean, data_average, data_std, data_variance, data_min, data_max

# -------------------------- EEG data -------------------------- #


def generate_paths(path: str) -> Union[str, str, str, str]:
    """Generate the paths to the .eeg, .vhdr, .vmrk and .txt files.

    Args:
        path (str): The path to the directory containing the files.

    Returns:
        tuple: The paths to the .eeg, .vhdr, .vmrk and .txt files.
    """
    eeg_path, vhdr_path, vmrk_path, txt_path = None, None, None, None
    # list all files in the directory
    files = os.listdir(path)
    # find paths to the .eeg, .vhdr, .vmrk and .txt files
    for file in files:
        if file.endswith('.eeg'):
            eeg_path = os.path.join(path, file)
        elif file.endswith('.vhdr'):
            vhdr_path = os.path.join(path, file)
        elif file.endswith('.vmrk'):
            vmrk_path = os.path.join(path, file)
        elif file.endswith('.txt'):
            txt_path = os.path.join(path, file)

    return eeg_path, vhdr_path, vmrk_path, txt_path


def snr_spectrum(psd: np.ndarray, noise_n_neighbor_freqs=1, noise_skip_neighbor_freqs=1) -> np.ndarray:
    """Compute SNR spectrum from PSD spectrum using convolution.

    Parameters
    ----------
    psd : ndarray, shape ([n_trials, n_channels,] n_frequency_bins)
        Data object containing PSD values. Works with arrays as produced by
        MNE's PSD functions or channel/trial subsets.
    noise_n_neighbor_freqs : int
        Number of neighboring frequencies used to compute noise level.
        increment by one to add one frequency bin ON BOTH SIDES
    noise_skip_neighbor_freqs : int
        set this >=1 if you want to exclude the immediately neighboring
        frequency bins in noise level calculation

    Returns
    -------
    snr : ndarray, shape ([n_trials, n_channels,] n_frequency_bins)
        Array containing SNR for all epochs, channels, frequency bins.
        NaN for frequencies on the edges, that do not have enough neighbors on
        one side to calculate SNR.
    """
    # Construct a kernel that calculates the mean of the neighboring
    # frequencies
    averaging_kernel = np.concatenate((
        np.ones(noise_n_neighbor_freqs),
        np.zeros(2 * noise_skip_neighbor_freqs + 1),
        np.ones(noise_n_neighbor_freqs)))
    averaging_kernel /= averaging_kernel.sum()

    # Calculate the mean of the neighboring frequencies by convolving with the
    # averaging kernel.
    mean_noise = np.apply_along_axis(
        lambda psd_: np.convolve(psd_, averaging_kernel, mode='valid'),
        axis=-1, arr=psd
    )

    # The mean is not defined on the edges so we will pad it with nas. The
    # padding needs to be done for the last dimension only so we set it to
    # (0, 0) for the other ones.
    edge_width = noise_n_neighbor_freqs + noise_skip_neighbor_freqs
    pad_width = [(0, 0)] * (mean_noise.ndim - 1) + [(edge_width, edge_width)]
    mean_noise = np.pad(
        mean_noise, pad_width=pad_width, constant_values=np.nan
    )

    return psd / mean_noise


def load_and_setup_eeg_data(eeg_path: str, vhdr_path: str, vmrk_path: str, txt_path: str) -> Union[pd.DataFrame,  np.ndarray, np.ndarray, int, str,
                                                                                                   mne.io.brainvision.brainvision.RawBrainVision, np.ndarray,
                                                                                                   float, list, float, float]:
    """Load and setup the EEG data.

    eeg_path (str): The path to the .eeg file.
    vhdr_path (str): The path to the .vhdr file.
    vmrk_path (str): The path to the .vmrk file.
    txt_path (str): The path to the .txt file.

    Returns:
        tuple: The data, sampling frequency, channel names, raw data, data mean, data average, data standard deviation, data variance, data minimum and data maximum.

    """
    # generate paths to the eeg, vhdr, vmrk and txt file
    df_txt = pd.read_csv(txt_path, sep='\t')

    # read the data using mne
    raw = mne.io.read_raw_brainvision(vhdr_path, preload=True)
    # get the data
    data = raw.get_data()
    # get the sampling frequency
    sfreq = raw.info['sfreq']
    # get the channel names
    ch_names = raw.info['ch_names']
    # save only the O1 and O2 channels except O1 only for participant 5
    channels_to_save = ['O1', 'O2']
    # drop all channels except O1, 02
    channels_to_drop = [ch for ch in ch_names if ch not in channels_to_save]

    # get the index of the channels to drop
    channels_to_drop_idx = [ch_names.index(ch) for ch in channels_to_drop]

    # drop the channels
    raw.drop_channels(channels_to_drop)
    # drop the channels from the channel names
    ch_names = [ch for ch in ch_names if ch not in channels_to_drop]

    # convert the list of trials to a mne epochs object
    all_events, _ = mne.events_from_annotations(raw, verbose=False)

    # create a mne epochs object from the data and the event
    tmin = 0.5  # start of each epoch (500ms after the trigger)
    tmax = 4    # end of each epoch (4000ms after the trigger)

    # number of events
    if '1X1' in txt_path:
        print(f"experiment 1X1")
        trials_events = np.where(all_events[:, 2] == 2)[0]
    else:
        print(f"experiment 2X2")
        trials_events = np.where(all_events[:, 2] == 4)[0]
    # number of trials
    number_of_trials = len(trials_events)

    return df_txt, all_events, trials_events, number_of_trials, txt_path, raw, data, sfreq, ch_names,  tmin, tmax


def process_eeg_trial_data(trial_index: int, df_txt: pd.DataFrame, all_events: np.ndarray, trials_events: np.ndarray, txt_path: str,
                           raw: mne.io.brainvision.brainvision.RawBrainVision, sfre: float, ch_names: list, tmin: float, tmax: float) -> \
        Union[np.ndarray, float]:
    """Process the EEG trial data.

    trial_index (int): The index of the trial.
    df_txt (pd.DataFrame): The dataframe containing the txt file.
    all_events (np.ndarray): The array containing all events.
    trials_events (np.ndarray): The array containing the trial events.
    txt_path (str): The path to the txt file.
    raw (mne.io.brainvision.brainvision.RawBrainVision): The raw data.
    sfre (float): The sampling frequency.
    ch_names (list): The list of channel names.
    tmin (float): The start of each epoch (500ms after the trigger).
    tmax (float): The end of each epoch (4000ms after the trigger).

    Returns:
        tuple: 
        max_snr_interval (np.ndarray): The max snr interval.
        max_snr_interval_frequeny (float): The max snr interval frequency.
    """

    # extract the trial events
    trial_events = [all_events[trials_events[trial_index]]]
    # extract the ith row of the txt_df
    txt_row = df_txt.iloc[trial_index]

    # check if the experiment is 1X1 or 2X2
    if '1X1' in txt_path:
        # get the stimulus frequency
        stimulus_freq = txt_row['frequency']
        # pixel surface
        pixel_surface = txt_row['pixel_surface']
        # shape of the stimulus
        shape = txt_row['shape']
        # color of the stimulus
        color = txt_row['color_mode']
        # name  video clip
        video_clip = txt_row['video_clip']
        # block number
        block_number = txt_row['Block_variable']
        # trial number
        trial_number = trial_index
    else:
        # get the stimulus frequency
        stimulus_freq = txt_row['frequency_2']
        # pixel surface
        pixel_surface = txt_row['pixel_surface_2']
        # shape of the stimulus
        shape = txt_row['shape_2']
        # color of the stimulus
        color = txt_row['color_mode_2']
        # name  video clip
        video_clip = txt_row['video_clip_2']
        # block number
        block_number = txt_row['Block_variable']
        # trial number
        trial_number = trial_index

    # check if the experiment is 1X1 or 2X2
    if '1X1' in txt_path:
        print(f" experiment 1x1")
        # Construct epochs
        event_id = {
            'Stimulus/S  2': 2
        }
        baseline = None
        epochs = mne.Epochs(
            raw, events=trial_events,
            event_id=[event_id['Stimulus/S  2']], tmin=tmin,
            tmax=tmax, baseline=baseline, verbose=False)
    else:
        print(f" experiment 2x2")
        # Construct epochs
        event_id = {
            'Stimulus/S  4': 4
        }
        baseline = None

        epochs = mne.Epochs(
            raw, events=trial_events,
            event_id=[event_id['Stimulus/S  4']], tmin=tmin,
            tmax=tmax, baseline=baseline, verbose=False)

    # get the data from the epochs from time tmin to tmax and frequency fmin to fmax
    tmin = 0.5
    tmax = 4.
    fmin = 1.
    fmax = 90.

    # get the sampling frequency
    sfreq = epochs.info['sfreq']
    # calculate teh EpochsSpectrum output using the welch method to calculate the psd across the entire epochs. Zero overlap between segments.
    spectrum = epochs.compute_psd('welch', n_fft=int(sfreq * (tmax - tmin)), n_overlap=0,
                                  n_per_seg=None, tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax, window='boxcar', verbose=False)

    # calculate teh power spectrum density (psd) spectrum
    spectrum = epochs.compute_psd(
        'welch',
        n_fft=int(sfreq * (tmax - tmin)),
        n_overlap=0, n_per_seg=None,
        tmin=tmin, tmax=tmax,
        fmin=fmin, fmax=fmax,
        window='boxcar',
        verbose=False)

    # extract the psd and the frequencies
    psds, freqs = spectrum.get_data(return_freqs=True)

    # calculate the snr spectrum
    snrs = snr_spectrum(psds, noise_n_neighbor_freqs=3,
                        noise_skip_neighbor_freqs=1)

    # Plot the SNR spectrum
    fig, axes = plt.subplots(2, 1, sharex='all', sharey='none', figsize=(8, 5))
    freq_range = range(np.where(np.floor(freqs) == 1.)[0][0],
                       np.where(np.ceil(freqs) == fmax - 1)[0][0])

    psds_plot = 10 * np.log10(psds)
    psds_mean = psds_plot.mean(axis=(0, 1))[freq_range]
    psds_std = psds_plot.std(axis=(0, 1))[freq_range]

    # set the main tile of the figure
    fig.suptitle(f'PSD and SNR spectrum for {video_clip}', fontsize=16)
    # draw vertical line at stimulus frequency
    axes[0].axvline(stimulus_freq, color='k', linestyle='--',
                    label='stimulus frequency')
    axes[1].axvline(stimulus_freq, color='k', linestyle='--',
                    label='stimulus frequency')

    axes[0].plot(freqs[freq_range], psds_mean, color='b')
    axes[0].fill_between(
        freqs[freq_range], psds_mean - psds_std, psds_mean + psds_std,
        color='b', alpha=.2)
    axes[0].set(title="PSD spectrum", ylabel='Power Spectral Density [dB]')

    # SNR spectrum
    snr_mean = snrs.mean(axis=(0, 1))[freq_range]
    snr_std = snrs.std(axis=(0, 1))[freq_range]

    axes[1].plot(freqs[freq_range], snr_mean, color='r')
    axes[1].fill_between(
        freqs[freq_range], snr_mean - snr_std, snr_mean + snr_std,
        color='r', alpha=.2)
    axes[1].set(
        title="SNR spectrum", xlabel='Frequency [Hz]',
        ylabel='SNR', ylim=[-2, 30], xlim=[fmin, fmax])

    # save fig
    video_clip = video_clip.replace('.mp4', '')
    # create a directory to save the figures in the same directory as the txt file
    directory = os.path.dirname(txt_path)
    # create a directory to save the figures
    if not os.path.exists(os.path.join(directory, 'figures')):
        os.makedirs(os.path.join(directory, 'figures'), exist_ok=True)
    # save the figure
    fig.savefig(os.path.join(directory, 'figures',
                f'{video_clip}_{block_number}_{trial_number}_dual_channel.png'))
    print(f'Figure saved for {video_clip}')

    # calculate the snr at the stimulus frequency
    snr_at_stimulus_freq = weighted_average_interpolation_snr(
        stimulus_freq, freqs, snrs)
    print(f'SNR at stimulus frequency: {snr_at_stimulus_freq}')

    # set up the frequency interval to search for the snr value at the stimulus frequency
    upper_limit_freq = stimulus_freq + 0.15
    lower_limit_freq = stimulus_freq - 0.15
    # calculate the snr at the upper and lower limit frequencies of the interval with interpolation
    snr_at_upper_limit_freq = weighted_average_interpolation_snr(
        upper_limit_freq, freqs, snrs)
    snr_at_lower_limit_freq = weighted_average_interpolation_snr(
        lower_limit_freq, freqs, snrs)
    # combine the snr values at the upper and lower limit frequencies with the snr value at the stimulus frequency
    snr_values_interval = np.array(
        [snr_at_lower_limit_freq, snr_at_stimulus_freq, snr_at_upper_limit_freq])
    frequency_interval = np.array(
        [lower_limit_freq, stimulus_freq, upper_limit_freq])

    # calculate the mean snr value for each channel
    mean_snr_values_channels = snr_values_interval.mean(axis=1)
    # get the maximal snr value
    max_snr_interval = mean_snr_values_channels.max()
    # get index of the channel with the maximal snr
    index_maximal_snr = np.argmax(mean_snr_values_channels)
    # get the frequency with the maximal snr
    max_snr_interval_frequency = frequency_interval[index_maximal_snr]

    print(
        f'Maximal SNR: {max_snr_interval} at frequency: {max_snr_interval_frequency}')

    return max_snr_interval, max_snr_interval_frequency


# -------------------------- Eye tracking data -------------------------- #
def process_eye_tracking_data(eye_tracking_path: str):
    """ Process the eye tracking data from the eye tracking software

    Args:
        eye_tracking_path (str): path to the eye tracking data
    """
    # load the data with utf-8 encoding
    data = pd.read_csv(eye_tracking_path, encoding='utf-16', delimiter='\t')

    # rename the columns
    # TIMESTAMP = time
    # LEFT_GAZE_X = left_gaze_x
    # LEFT_GAZE_Y = left_gaze_y
    # RIGHT_GAZE_X = right_gaze_x
    # RIGHT_GAZE_Y = right_gaze_y
    # LEFT_PUPIL_SIZE = left_p
    # RIGHT_PUPIL_SIZE = right_p
    # renamce the columns
    data.rename(columns={'TIMESTAMP': 'time', 'LEFT_GAZE_X': 'left_x', 'LEFT_GAZE_Y': 'left_y', 'RIGHT_GAZE_X': 'right_x',
                'RIGHT_GAZE_Y': 'right_y', 'LEFT_PUPIL_SIZE': 'left_p', 'RIGHT_PUPIL_SIZE': 'right_p'}, inplace=True)
    # seperaate the data into trials by the trial number so that the trial_index can be used to access the data under the trial
    # get the trial numbers
    trial_numbers = data['TRIAL_INDEX'].unique()
    # get the data for each trial
    trial_data = []
    for trial in trial_numbers:
        data_per_trial = data[data['TRIAL_INDEX'] == trial]
        # remove data where the VIDEO_NAME is '.'
        data_per_trial = data_per_trial[data_per_trial['VIDEO_NAME'] != '.']
        # add empty columns for the average gaze and pupil size
        data_per_trial['average_p'] = np.nan
        # make a deep copy of the data
        data_per_trial = copy.deepcopy(data_per_trial)
        trial_data.append(data_per_trial)
    try:
        t, xf, yf, pf, fixations, saccades, fd, sl, ff, sf, sa, fa = process_eyes(
            trial_data)
    except:
        print('Error processing eye tracking data')
        t, xf, yf, pf, fixations, saccades, fd, sl, ff, sf, sa, fa = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    # create a new dataframe using the same columns as the variables t, xf, yf, pf, fixations, saccades, fd, sl, ff, sf, sa, fa
    new_data = pd.DataFrame(columns=[
                            't', 'xf', 'yf', 'pf', 'fixations', 'saccades', 'fd', 'sl', 'ff', 'sf', 'sa', 'fa'])
    # add the data to the new dataframe
    new_data['t'] = t
    new_data['xf'] = xf
    new_data['yf'] = yf
    new_data['pf'] = pf
    new_data['fixations'] = fixations
    new_data['saccades'] = saccades
    new_data['fd'] = fd
    new_data['sl'] = sl
    new_data['ff'] = ff
    new_data['sf'] = sf
    new_data['sa'] = sa
    new_data['fa'] = fa

    # save the data at the same path folder as the csv file
    new_data.to_csv(eye_tracking_path[:-4] +
                    '_processed_fixations.csv', index=False)
    print(
        f"Saved the processed data to {eye_tracking_path[:-4]}_processed_fixations.csv")


def _draw_circles(image: np.ndarray, pixel_surface: int, center_coordinates: np.ndarray, color_tuples: list) -> np.ndarray:
    """ Draw circles on the image

    Args:
        image (np.ndarray): the image on which the circles are drawn
        pixel_surface (int): the surface of the circles in pixels
        center_coordinates (np.ndarray): the center coordinates of the circles to be drawn in pixels
        color_tuples (list): the color of the circles to be drawn

    Returns:
        np.ndarray: the image with the circles drawn on it
    """
    # calculate radius of circle
    radius = int(np.sqrt(pixel_surface/pi))
    #  draw circles
    for i, center_coordinate in enumerate(center_coordinates):
        # draw all circles.
        color_tuple = color_tuples[i]
        # draw the circle
        image = cv2.circle(image, center_coordinate, radius, color_tuple, -1)

    return image


def _draw_squares(image: np.ndarray, pixel_surface: int, center_coordinates: np.ndarray, color_tuples: list) -> np.ndarray:
    """ Draw squares on the image

    Args:
        image (np.ndarray): the image on which the squares are drawn
        pixel_surface (int): the surface of the squares in pixels
        center_coordinates (np.ndarray): the center coordinates of the squares to be drawn in pixels
        color_tuples (list): the color of the squares to be drawn

    Returns:
        np.ndarray: the image with the squares drawn on it
    """

    # calculate side length of square
    side_length = int(np.sqrt(pixel_surface))
    # create squares
    for i, center_coordinate in enumerate(center_coordinates):
        # draw all squares.
        color_tuple = color_tuples[i]
        # calculate the coordinates of the top left and bottom right corners of the square
        coordinate_1 = (int(
            center_coordinate[0]-side_length//2), int(center_coordinate[1]-side_length//2))
        coordinate_2 = (int(
            center_coordinate[0]+side_length//2), int(center_coordinate[1]+side_length//2))
        # draw the square
        image = cv2.rectangle(image, coordinate_1,
                              coordinate_2, color_tuple, -1)

    return image


def _draw_triangles(image: np.ndarray, pixel_surface: int, center_coordinates: list, color_tuples: list) -> np.ndarray:
    """ Draw triangles on the image

    Args:
        image (np.ndarray): the image on which the triangles are drawn
        pixel_surface (int): the surface of the triangles in pixels
        center_coordinates (np.ndarray): the center coordinates of the triangles to be drawn in pixels
        color_tuples (list): the color of the triangles to be drawn

    Returns:
        np.ndarray: the image with the triangles drawn on it
    """
    # us the pixel surface to derive corner coordinates of the iscoceles triangle. All sides of the triangle are equal
    # calculate side length of triangle
    diagonal = np.sqrt(pixel_surface*4/np.sqrt(3))
    # half base length
    half_base_length = diagonal/2
    height = diagonal/2*np.sqrt(3)

    # create triangles
    for i, center_coordinate in enumerate(center_coordinates):
        # draw all triangles. The coordinates are measured in integer values, so the triangles are not perfectly centered
        color_tuple = color_tuples[i]
        coordinate_1 = (
            int(center_coordinate[0]-half_base_length), int(center_coordinate[1]+height//2))
        coordinate_2 = (
            int(center_coordinate[0]+half_base_length), int(center_coordinate[1]+height//2))
        coordinate_3 = (int(center_coordinate[0]), int(
            center_coordinate[1]-height//2))
        triangle_center_coordinate = (int(center_coordinate[0]), int(
            (coordinate_1[1]+coordinate_2[1]+coordinate_3[1])/3))
        # correct the 3 coordinates down to make center_coordinate  equal to the center coordinate of the triangle
        if triangle_center_coordinate[1] > center_coordinate[1]:
            correction = center_coordinate[1] - triangle_center_coordinate[1]
            coordinate_1 = (coordinate_1[0], coordinate_1[1]+correction)
            coordinate_2 = (coordinate_2[0], coordinate_2[1]+correction)
            coordinate_3 = (coordinate_3[0], coordinate_3[1]+correction)

        elif triangle_center_coordinate[1] < center_coordinate[1]:
            correction = triangle_center_coordinate[1] - center_coordinate[1]
            coordinate_1 = (coordinate_1[0], coordinate_1[1]-correction)
            coordinate_2 = (coordinate_2[0], coordinate_2[1]-correction)
            coordinate_3 = (coordinate_3[0], coordinate_3[1]-correction)

        # draw the triangle
        triangle_cnt = np.array([coordinate_1, coordinate_2, coordinate_3])
        image = cv2.drawContours(image, [triangle_cnt], 0, color_tuple, -1)
    return image


def draw_shape(shape: str, pixel_surface: int, center_coordinates: np.ndarray, color_tuples: list) -> np.ndarray:
    """ Draw shapes on the image

    Args:
        shape (str): the shape to be drawn. Must be 'circle', 'square' or 'triangle'
        pixel_surface (int): the surface of the shapes in pixels
        center_coordinates (np.ndarray): the center coordinates of the shapes to be drawn in pixels
        color_tuples (list): the color of the shapes to be drawn

    Returns:
        np.ndarray: the image with the shapes drawn on it
    """
    # create empty image
    image = np.zeros((1080, 1920, 3), np.uint8)
    # draw the shapes
    if '.mp4' in shape:
        shape = shape[:-4]
    if shape == "circles":
        image = _draw_circles(image, pixel_surface,
                              center_coordinates, color_tuples)
    elif shape == "squares":
        image = _draw_squares(image, pixel_surface,
                              center_coordinates, color_tuples)
    elif shape == "triangles":
        image = _draw_triangles(image, pixel_surface,
                                center_coordinates, color_tuples)
    else:
        raise ValueError("shape must be 'circle', 'square' or 'triangle'")

    return image


def draw_points(img: np.ndarray, x: int, y: int, gaze_x: np.ndarray, gaze_y: np.ndarray) -> Union[np.ndarray, list]:
    """ Draw the gaze points on the image

    Args:
        img (np.ndarray): the image on which the gaze points are drawn
        x (int): the x coordinate of the center of the shape
        y (int): the y coordinate of the center of the shape
        gaze_x (np.ndarray): the x coordinates of the gaze points
        gaze_y (np.ndarray): the y coordinates of the gaze points

    Returns:
        Union[np.ndarray, list]: the image with the gaze points drawn on it and the points on the shape
    """
    # create a mask with the shape
    mask = cv2.inRange(img, (0, 0, 0), (0, 0, 0))
    # flip the mask to find the shape
    mask = cv2.bitwise_not(mask)
    # find the shape
    points_shape_mask = np.where(mask == 255)

    # draw a circle at x, y with radius 5 and color red
    cv2.circle(img, (x, y), 5, (0, 0, 0), -1)
    # draw all the gaze points

    points_on_target = []
    for index in range(len(gaze_x)):

        int_x = int(round(gaze_x[index]))
        int_y = int(round(gaze_y[index]))
        # if mask is 255, the point is on the shape
        try:
            if mask[int_y, int_x] == 255:
                points_on_target.append((int_x, int_y))
        except IndexError:
            print(f"Point {int_x, int_y} is not in the image")
        cv2.circle(img, (int_x, int_y), 1, (255, 0, 0), -1)

    return img, points_on_target


def create_heatmap(x: int, y: int, gaze_x: np.ndarray, gaze_y: np.ndarray) -> np.ndarray:
    """ Create a heatmap of the gaze points

    Args:
        x (int): the x coordinate of the center of the shape
        y (int): the y coordinate of the center of the shape
        gaze_x (np.ndarray): the x coordinates of the gaze points
        gaze_y (np.ndarray): the y coordinates of the gaze points

    Returns:
        np.ndarray: the heatmap of the gaze points
    """
    # create empty image of size 1920 x 1080
    img = np.zeros((1080, 1920, 3), np.uint8)
    # draw a circle at x, y with radius 5 and color red
    cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
    # generate set of points convering the drawn circle at coordinate x, y
    # create a mask of the circle
    mask = np.zeros((1080, 1920), np.uint8)
    cv2.circle(mask, (x, y), 5, (255, 255, 255), -1)
    # find the coordinates of the points in the mask
    points_circle = np.where(mask == 255)
    # convert the points to a list of tuples
    points_circle = list(zip(points_circle[0], points_circle[1]))

    # convert gaze_x and gaze_y to one dimensional arrays instead of a list of arrays
    gaze_x = np.concatenate(gaze_x)
    gaze_y = np.concatenate(gaze_y)

    print(f" Creating int points")
    # draw all the gaze points
    points = []
    for index in range(len(gaze_x)):
        x_coordinate = float(gaze_x[index])
        y_coordinate = float(gaze_y[index])
        int_x = int(round(x_coordinate))
        int_y = int(round(y_coordinate))
        points.append((int_y, int_x))

    print(f" Done creating int points")
    point = np.array(points)
    # find number of occurences of each point
    unique, counts = np.unique(point, axis=0, return_counts=True)

    maximal_value = np.max(counts)
    # normalize the counts to be between 0 and 255
    counts = (counts/maximal_value)*255

    # draw the points with the number of occurences as the color
    for index in range(len(unique)):
        y_coordinate, x_coordinate = unique[index]

        if tuple(unique[index]) in points_circle:
            color = (0, int(counts[index]), 0)
        else:
            color = (int(counts[index]), int(
                counts[index]), int(counts[index]))
        cv2.circle(img, (x_coordinate, y_coordinate), 1, color, -1)

    return img


def process_2x2_data(data_eye_tracking: pd.DataFrame, path: str, path_image_dir: str, data_trial_sequence: pd.DataFrame, folder_path: str):
    """ Process the data from the 2x2 experiment

    Args:
        data_eye_tracking (pd.DataFrame): data from the eye tracker
        path (str): path to the eye tracking data
        path_image_dir (str): path to the directory with the images
        data_trial_sequence (pd.DataFrame):  data from the trial sequence (Expereriment X.txt/csv)
        folder_path (str): path to the main folder of the experiment
    """
    # generate paths to the files
    eeg_path, vhdr_path, vmrk_path, txt_path = generate_paths(folder_path)
    # load the eeg data
    df_txt, all_events, trials_events, num_of_trials, txt_path, raw, data, sfreq, ch_names,  tmin, tmax = load_and_setup_eeg_data(
        txt_path, vhdr_path, vmrk_path, txt_path)
    # get and process the data from the eye tracker
    process_eye_tracking_data(path)
    # extract the headers of the data_eye_tracking
    headers = data_eye_tracking.columns.values

    # correct the trial index
    if data_eye_tracking["TRIAL_INDEX"].iloc[0] > 1:
        difference_trial_index = data_eye_tracking["TRIAL_INDEX"].iloc[0] - 1
        # correct the trial index in the data_eye_tracking
        data_eye_tracking["TRIAL_INDEX"] = data_eye_tracking["TRIAL_INDEX"] - \
            difference_trial_index
    # get the unique trial indices
    trial_indices = data_eye_tracking['TRIAL_INDEX']
    # get the unique trial indices
    trial_indices_unique = np.unique(trial_indices)
    # add trial_index column to data_trial_sequence
    data_trial_sequence['TRIAL_INDEX'] = data_trial_sequence.index+1
    # extract the headers of the data_trial_sequence
    headers_trial_sequence = data_trial_sequence.columns.values

    # get name of header with video_clip in it
    video_clip_header = [
        header for header in headers_trial_sequence if 'video_clip' in header][0]

    # create empty data frame to store the processed data
    headers_processed_data = ["TRIAL_INDEX", 'VIDEO_NAME', "BLOCK", 'PIXEL_SURFACE', "COLOR", "SHAPE", "FREQUENCY", "DISPLAYED_FRAMES", "DROPPED_FRAMES", "MAX_SNR", 'FREQUENCY_SAMPLED_AT', 'MEAN_GAZE_DISTANCE', 'AVERAGE_GAZE_DISTANCE',
                              'VARIANCE_GAZE_DISTANCE', 'MEAN_GAZE_ANGLE', 'AVERAGE_GAZE_ANGLE', 'VARIANCE_GAZE_ANGLE', "MEAN_X", "X_AVERAGE", "X", "X_VARIANCE",  "Y_MEAN", "Y_AVERAGE", "Y", "Y_VARIANCE", "NUMBER_OF_GAZE_POINTS", "POINTS_ON_TARGET"]
    processed_data = pd.DataFrame(columns=headers_processed_data)
    # create empty lists to store the gaze data
    all_gaze_x = []
    all_gaze_y = []

    # get rid of the data_eye_tracking['VIDEO_NAME'] is '.'
    data_eye_tracking = data_eye_tracking[data_eye_tracking['VIDEO_NAME'] != '.']

    # loop through all the videos
    experimental_counter = 0
    counter = 0
    for trial_index in trial_indices_unique:
        # Extract max SNR
        trial_index_for_list = trial_index - 1
        # block variable
        block = data_trial_sequence.iloc[trial_index_for_list]['Block_variable']
        # get the number of displayed frames
        displayed_frames = data_trial_sequence.iloc[trial_index_for_list]['displayed_frame_count']
        # get the number of dropped frames
        dropped_frames = data_trial_sequence.iloc[trial_index_for_list]['dropped_frame_count']
        max_snr, frequency_sampled_at = process_eeg_trial_data(
            trial_index_for_list, df_txt, all_events, trials_events, txt_path, raw, sfreq, ch_names, tmin, tmax)

        # extract the video name from the data_trial_sequence
        print(f"extracting video name for trial index {trial_index}")
        # check if name available in data_eye_tracking
        if '2x2' in data_eye_tracking['VIDEO_NAME'].iloc[trial_index-1]:
            # extract the video name of the trial index
            first_index_video_name = np.where(
                data_eye_tracking['TRIAL_INDEX'] == trial_index)[0][0]
            # extract the video name using the first index
            video_name_1 = data_eye_tracking['VIDEO_NAME'].iloc[first_index_video_name]
            # extract the video name from the data_trial_sequence
            video_name_2 = data_trial_sequence.loc[data_trial_sequence['TRIAL_INDEX']
                                                   == trial_index][video_clip_header].values[0]
            # check if the video names are the same
            assert (video_name_1 == video_name_2)
            # set the video name
            video_name = video_name_1

        # extract the # extract the coordinates of the video from the video name
        parsed_video_name = video_name.split('_')
        # extract the shape, color, and frequency of the shape
        pixel_surface = parsed_video_name[3]
        color = parsed_video_name[6]
        frequency = parsed_video_name[8]
        shape = parsed_video_name[10]
        x = int(parsed_video_name[12][1:])
        y = int(parsed_video_name[13].split(')')[0])

        # check if the video has gaze data
        gaze_data = True

        print(f"extracting data for video {video_name}")

        # extract all the data for the current video
        video_data = data_eye_tracking.loc[data_eye_tracking['TRIAL_INDEX'] == trial_index]
        # extract the gaze data for the left and right eye

        gaze_x, gaze_y = video_data['AVERAGE_GAZE_X'], video_data['AVERAGE_GAZE_Y']
        try:
            gaze_x = np.array([float(x) for x in gaze_x])
            gaze_y = np.array([float(y) for y in gaze_y])
        except ValueError:
            # if the video has no gaze data
            print(f"video {video_name} has no gaze data")
            gaze_data = False

        print(f"calculating the metrics for video {video_name}")
        if gaze_data:
            # convert the data to a numpy array
            gaze_x = np.array(gaze_x)
            gaze_y = np.array(gaze_y)

            # calcualte the between the gaze and the center of the shape
            gaze_x_diff = gaze_x - x
            gaze_y_diff = gaze_y - y

            # calculate the distance between the gaze and the center of the shape
            gaze_distance = np.zeros(len(gaze_x_diff))
            gaze_angle = np.zeros(len(gaze_x_diff))
            for index in range(len(gaze_x_diff)):
                gaze_distance[index] = np.sqrt(
                    gaze_x_diff[index] ** 2 + gaze_y_diff[index] ** 2)
                gaze_angle[index] = np.arctan2(
                    gaze_y_diff[index], gaze_x_diff[index])
            # convert to numpy array
            gaze_distance = np.array(gaze_distance)
            gaze_angle = np.array(gaze_angle)
            # calculate the number of gaze measurements
            number_of_gaze_measurements_trial = len(gaze_x)

            # calculate the STATISTICS
            #data_mean, data_average, data_std, data_variance, data_min, data_max
            gaze_x_mean, gaze_x_average, gaze_x_std, gaze_x_variance, gaze_x_min, gaze_x_max = calculate_statistics(
                gaze_x)
            gaze_y_mean, gaze_y_average, gaze_y_std, gaze_y_variance, gaze_y_min, gaze_y_max = calculate_statistics(
                gaze_y)
            gaze_distance_mean, gaze_distance_average, gaze_distance_std, gaze_distance_variance, gaze_distance_min, gaze_distance_max = calculate_statistics(
                gaze_distance)
            gaze_angle_mean, gaze_angle_average, gaze_angle_std, gaze_angle_variance, gaze_angle_min, gaze_angle_max = calculate_statistics(
                gaze_angle)

            print(f"video name: {video_name}| mean gaze distance: {gaze_distance_mean} | average gaze distance: {gaze_distance_average} | variance gaze distance: {gaze_distance_variance} | mean gaze angle: {gaze_angle_mean} | average gaze angle: {gaze_angle_average} | variance gaze angle: {gaze_angle_variance}, max snr: {max_snr}")
        else:
            if trial_index > 4:
                counter += 1
            else:
                experimental_counter += 1

        # create a figure
        # select the color of the shape
        if color == 'red':
            color_tuple = (0, 0, 255)
        elif color == "green":
            color_tuple = (0, 255, 0)
        elif color == "blue":
            color_tuple = (255, 0, 0)
        elif color == "white":
            color_tuple = (255, 255, 255)
        # draw the shape
        img = draw_shape(shape, int(pixel_surface), [(x, y)], [color_tuple])

        if gaze_data:
            # draw the points on the target
            img, points_on_target = draw_points(img, x, y, gaze_x, gaze_y)
            # calculate the number of points on the target
            number_of_points_on_target = len(points_on_target)
            # calculate the number of points on the target
            row = [trial_index, video_name, block, pixel_surface, color, shape, frequency, displayed_frames, dropped_frames, max_snr, frequency_sampled_at, gaze_distance_mean, gaze_distance_average, gaze_distance_variance,
                   gaze_angle_mean, gaze_angle_average, gaze_angle_variance, gaze_x_mean, gaze_x_average, x, gaze_x_variance, gaze_y_mean, gaze_y_average,  y, gaze_y_variance, number_of_gaze_measurements_trial, number_of_points_on_target]
        else:
            # when there is no gaze data
            row = [trial_index, video_name, block, pixel_surface, color, shape, frequency, displayed_frames, dropped_frames, max_snr, frequency_sampled_at,
                   np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]

        # append the row to the processed data
        processed_data.loc[len(processed_data)] = row
        # save the figure
        video_name_to_jpg = str(trial_index)+"_" + \
            video_name.split('.')[0] + '.jpg'

        # save the image
        cv2.imwrite(os.path.join(path_image_dir, video_name_to_jpg), img)
        # save the processed data
        processed_data.to_csv(os.path.join(
            path_image_dir, 'processed_data.csv'), index=False)
        if gaze_data:
            all_gaze_x.append(gaze_x)
            all_gaze_y.append(gaze_y)

    # get the dropped frames across all the trials
    dropped_frames = processed_data['DROPPED_FRAMES']
    # plot the dropped frames across all the trials with matplotlib
    # create new figure
    plt.figure(figsize=(20, 10))
    # plot the dropped frames
    plt.plot(dropped_frames)
    # set the x and y labels
    plt.ylabel('dropped frames')
    plt.xlabel('trial index')
    # set x limits to the number of trials
    plt.xlim(0, len(dropped_frames))
    # save the figure
    plt.savefig(os.path.join(path_image_dir, 'dropped_frames.png'))
    plt.close()
    # create a heatmap of the gaze data
    heatmap = create_heatmap(x, y, all_gaze_x, all_gaze_y)
    # save the heatmap
    cv2.imwrite(os.path.join(path_image_dir, 'heatmap.jpg'), heatmap)
    print(
        f" Of all the videos {counter} videos had no gaze data, and {len(trial_indices_unique)-4-counter} videos had gaze data")
    print(
        f" Of all the Experimental videos {experimental_counter} videos had no gaze data, and {4-experimental_counter} videos had gaze data")
    # skip the first 4 trials because they are not relevant
    experimental_processed_data = processed_data.iloc[4:]
    # extract the video names
    video_names = experimental_processed_data['VIDEO_NAME']
    # count the unique video names and number of occurences
    unique_video_names, counts = np.unique(video_names, return_counts=True)
    # check if the number of occurences is the same for all the video names is 3
    if np.all(counts == 3):
        print("All the video names have 3 occurences")


def process_1x1_data(data_eye_tracking: pd.DataFrame, path: str, path_image_dir: str, data_trial_sequence: pd.DataFrame, folder_path: str):
    """ Process the data from the 1x1 experiment

    Args:
        data_eye_tracking (pd.DataFrame): data from the eye tracker
        path (str): path to the eye tracking data
        path_image_dir (str): path to the directory with the images
        data_trial_sequence (pd.DataFrame):  data from the trial sequence (Expereriment X.txt/csv)
        folder_path (str): path to the main folder of the experiment
    """
    # generate paths to the files
    eeg_path, vhdr_path, vmrk_path, txt_path = generate_paths(folder_path)
    # load the eeg data
    df_txt, all_events, trials_events, num_of_trials, txt_path, raw, data, sfreq, ch_names,  tmin, tmax = load_and_setup_eeg_data(
        txt_path, vhdr_path, vmrk_path, txt_path)
    # get and process the data from the eye tracker
    process_eye_tracking_data(path)
    # extract the headers of the data_eye_tracking
    headers = data_eye_tracking.columns.values
    # correct the trial index
    if data_eye_tracking["TRIAL_INDEX"].iloc[0] > 1:
        difference_trial_index = data_eye_tracking["TRIAL_INDEX"].iloc[0] - 1
        # correct the trial index in the data_eye_tracking
        data_eye_tracking["TRIAL_INDEX"] = data_eye_tracking["TRIAL_INDEX"] - \
            difference_trial_index
    # get the trial indices
    trial_indices = data_eye_tracking['TRIAL_INDEX']
    # get the unique trial indices
    trial_indices_unique = np.unique(trial_indices)
    # add trial_index column to data_trial_sequence
    data_trial_sequence['TRIAL_INDEX'] = data_trial_sequence.index+1
    # extract the headers of the data_trial_sequence
    headers_trial_sequence = data_trial_sequence.columns.values

    # get name of header with video_clip in it
    video_clip_header = [
        header for header in headers_trial_sequence if 'video_clip' in header][0]

    # create empty data frame to store the processed data
    headers_processed_data = ["TRIAL_INDEX", 'VIDEO_NAME', "BLOCK", 'PIXEL_SURFACE', "COLOR", "SHAPE", "FREQUENCY", "DISPLAYED_FRAMES", "DROPPED_FRAMES", "MAX_SNR", 'FREQUENCY_SAMPLED_AT', 'MEAN_GAZE_DISTANCE', 'AVERAGE_GAZE_DISTANCE',
                              'VARIANCE_GAZE_DISTANCE', 'MEAN_GAZE_ANGLE', 'AVERAGE_GAZE_ANGLE', 'VARIANCE_GAZE_ANGLE', "MEAN_X", "X_AVERAGE", "X", "X_VARIANCE",  "Y_MEAN", "Y_AVERAGE", "Y", "Y_VARIANCE", "NUMBER_OF_GAZE_POINTS", "POINTS_ON_TARGET"]
    processed_data = pd.DataFrame(columns=headers_processed_data)

    # create empty lists to store the gaze data
    all_gaze_x = []
    all_gaze_y = []

    # get rid of the data_eye_tracking['VIDEO_NAME'] is '.'
    data_eye_tracking = data_eye_tracking[data_eye_tracking['VIDEO_NAME'] != '.']

    experimental_counter = 0
    counter = 0
    # loop through all the videos
    for trial_index in trial_indices_unique:
        # Extract max SNR
        trial_index_for_list = trial_index - 1
        # block variable
        block = data_trial_sequence.iloc[trial_index_for_list]['Block_variable']
        # get the number of displayed frames
        displayed_frames = data_trial_sequence.iloc[trial_index_for_list]['displayed_frame_count']
        # get the number of dropped frames
        dropped_frames = data_trial_sequence.iloc[trial_index_for_list]['dropped_frame_count']
        max_snr, frequency_sampled_at = process_eeg_trial_data(
            trial_index_for_list, df_txt, all_events, trials_events, txt_path, raw, sfreq, ch_names, tmin, tmax)

        # extract the video name from the data_trial_sequence
        print(f"extracting video name for trial index {trial_index}")
        if '1x1' in data_eye_tracking['VIDEO_NAME'].iloc[trial_index-1]:
            # extract the video name of the trial index by finding the first index of the trial index in the data_eye_tracking
            first_index_video_name = np.where(
                data_eye_tracking['TRIAL_INDEX'] == trial_index)[0][40]
            # extract the video name using the first index
            video_name_1 = data_eye_tracking['VIDEO_NAME'].iloc[first_index_video_name]
            # extract the video name from the data_trial_sequence
            video_name_2 = data_trial_sequence.loc[data_trial_sequence['TRIAL_INDEX']
                                                   == trial_index][video_clip_header].values[0]
            # check if the video names are the same
            assert (video_name_1 == video_name_2)
            # set the video name
            video_name = video_name_1

        # extract the # extract the coordinates of the video from the video name
        parsed_video_name = video_name.split('_')
        # extract the shape, color, and frequency of the shape
        pixel_surface = parsed_video_name[3]
        color = parsed_video_name[6]
        frequency = parsed_video_name[8]
        shape = parsed_video_name[10][:-4]

        # check if the video has gaze data
        gaze_data = True

        print(f"extracting data for video {video_name}")
        # set the x and y coordinates of the center of the shape
        x = 960
        y = 540
        # extract all the data for the current video
        video_data = data_eye_tracking.loc[data_eye_tracking['TRIAL_INDEX'] == trial_index]
        # extract the average gaze data for the left and right eye
        gaze_x, gaze_y = video_data['AVERAGE_GAZE_X'], video_data['AVERAGE_GAZE_Y']
        try:
            gaze_x = np.array([float(x) for x in gaze_x])
            gaze_y = np.array([float(y) for y in gaze_y])
        except ValueError:
            # if the video has no gaze data
            print(f"video {video_name} has no gaze data")
            gaze_data = False

        print(f"calculating the metrics for video {video_name}")
        if gaze_data:
            # convert the data to a numpy array
            gaze_x = np.array(gaze_x)
            gaze_y = np.array(gaze_y)

            # calcualte the between the gaze and the center of the shape
            gaze_x_diff = gaze_x - x
            gaze_y_diff = gaze_y - y

            # calculate the distance between the gaze and the center of the shape
            gaze_distance = np.zeros(len(gaze_x_diff))
            gaze_angle = np.zeros(len(gaze_x_diff))
            for index in range(len(gaze_x_diff)):
                gaze_distance[index] = np.sqrt(
                    gaze_x_diff[index] ** 2 + gaze_y_diff[index] ** 2)
                gaze_angle[index] = np.arctan2(
                    gaze_y_diff[index], gaze_x_diff[index])
            # convert to numpy array
            gaze_distance = np.array(gaze_distance)
            gaze_angle = np.array(gaze_angle)
            # calculate the number of gaze measurements
            number_of_gaze_measurements_trial = len(gaze_x)

            #data_mean, data_average, data_std, data_variance, data_min, data_max
            gaze_x_mean, gaze_x_average, gaze_x_std, gaze_x_variance, gaze_x_min, gaze_x_max = calculate_statistics(
                gaze_x)
            gaze_y_mean, gaze_y_average, gaze_y_std, gaze_y_variance, gaze_y_min, gaze_y_max = calculate_statistics(
                gaze_y)
            gaze_distance_mean, gaze_distance_average, gaze_distance_std, gaze_distance_variance, gaze_distance_min, gaze_distance_max = calculate_statistics(
                gaze_distance)
            gaze_angle_mean, gaze_angle_average, gaze_angle_std, gaze_angle_variance, gaze_angle_min, gaze_angle_max = calculate_statistics(
                gaze_angle)
            print(f"video name: {video_name}| mean gaze distance: {gaze_distance_mean} | average gaze distance: {gaze_distance_average} | variance gaze distance: {gaze_distance_variance} | mean gaze angle: {gaze_angle_mean} | average gaze angle: {gaze_angle_average} | variance gaze angle: {gaze_angle_variance}")

        else:
            # if the no practice trials anymore
            if trial_index > 4:
                counter += 1
            else:
                experimental_counter += 1

        # create a figure and start with choosing the color of the shape
        if color == 'red':
            color_tuple = (0, 0, 255)
        elif color == "green":
            color_tuple = (0, 255, 0)
        elif color == "blue":
            color_tuple = (255, 0, 0)
        elif color == "white":
            color_tuple = (255, 255, 255)

        # draw the shape
        img = draw_shape(shape, int(pixel_surface), [(x, y)], [color_tuple])

        # draw the gaze points
        if gaze_data:
            # draw the gaze points and extract the number of points on the target
            img, points_on_target = draw_points(img, x, y, gaze_x, gaze_y)
            number_of_points_on_target = len(points_on_target)
            # calculate the number of points on the target
            row = [trial_index, video_name, block, pixel_surface, color, shape, frequency, displayed_frames, dropped_frames, max_snr, frequency_sampled_at, gaze_distance_mean, gaze_distance_average, gaze_distance_variance,
                   gaze_angle_mean, gaze_angle_average, gaze_angle_variance, gaze_x_mean, gaze_x_average, x, gaze_x_variance, gaze_y_mean, gaze_y_average,  y, gaze_y_variance, number_of_gaze_measurements_trial, number_of_points_on_target]
        else:
            # when there is no gaze data
            row = [trial_index, video_name, block, pixel_surface, color, shape, frequency, displayed_frames, dropped_frames, max_snr, frequency_sampled_at,
                   np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
        # append the row to the processed data
        processed_data.loc[len(processed_data)] = row
        # save the figure
        video_name_to_jpg = str(trial_index)+"_" + \
            video_name.split('.')[0] + '.jpg'

        # save the image
        cv2.imwrite(os.path.join(path_image_dir, video_name_to_jpg), img)
        # save the processed data
        processed_data.to_csv(os.path.join(
            path_image_dir, 'processed_data.csv'), index=False)
        # save the gaze data
        if gaze_data:
            all_gaze_x.append(gaze_x)
            all_gaze_y.append(gaze_y)

    # get the dropped frames across all the trials
    dropped_frames = processed_data['DROPPED_FRAMES']
    # plot the dropped frames across all the trials with matplotlib
    # create new figure
    plt.figure(figsize=(20, 10))
    # plot the dropped frames
    plt.plot(dropped_frames)
    # set the x and y labels
    plt.xlabel('trial index')
    plt.ylabel('dropped frames')
    # set x limits to the number of trials
    plt.xlim(0, len(dropped_frames))
    # save the figure
    plt.savefig(os.path.join(path_image_dir, 'dropped_frames.png'))
    plt.close()

    # create a heatmap of the gaze data
    heatmap = create_heatmap(x, y, all_gaze_x, all_gaze_y)
    # save the heatmap
    cv2.imwrite(os.path.join(path_image_dir, 'heatmap.jpg'), heatmap)
    print(
        f" Of all the videos {counter} videos had no gaze data, and {len(trial_indices_unique)-4-counter} videos had gaze data")
    print(
        f" Of all the Experimental videos {experimental_counter} videos had no gaze data, and {4-experimental_counter} videos had gaze data")
    # skip the first 4 trials because they are not relevant
    experimental_processed_data = processed_data.iloc[4:]
    # extract the video names
    video_names = experimental_processed_data['VIDEO_NAME']
    # count the unique video names and number of occurences
    unique_video_names, counts = np.unique(video_names, return_counts=True)
    # check if the number of occurences is the same for all the video names is 3
    if np.all(counts == 3):
        print("All the video names have 3 occurences")


def load_csv_data_utf8(path: str) -> pd.DataFrame:
    """ Load the data from the csv file with UTF-8 encoding

    Args:
        path (str): Path to the csv file

    Returns:
        pd.DataFrame: Dataframe with the data from the csv file
    """
    # convert path to raw string
    new_path = r'{}'.format(path)
    with open(new_path, encoding='UTF-8') as f:
        # read csv file and store in dataframe with the correct headers
        data = pd.read_csv(f, sep='\t')
    return data


def main():
    # Load the eye tracking data
    folder_path = r'/media/sjoerd/BackUp Drive/Thesis_project/participant data/raw data/pp5/Experiment 1'

    if 'Experiment 2' in folder_path:
        path_eye_tracking = folder_path + '/Experiment_2x2_eye_tracking.csv'
        path_trial_sequence = folder_path + '/EXPERIMENT_2X2.csv'
    elif 'Experiment 1' in folder_path:
        path_eye_tracking = folder_path + '/Experiment_1x1_eye_tracking.csv'
        path_trial_sequence = folder_path + '/EXPERIMENT_1X1.csv'
    else:
        print("Please select the correct folder")
        return

    # create folder for images
    path_img_folder = path_eye_tracking.split('.csv')[0]+'/images'
    print(f" The folder to save the images to is {path_img_folder}")

    # create folder for images
    if not os.path.exists(path_img_folder):
        os.makedirs(path_img_folder)
    # convert path to raw string
    new_path = r'{}'.format(path_eye_tracking)
    with open(new_path, encoding='UTF-16') as f:
        data_eye_tracking = pd.read_csv(f, sep='\t')
    # extract the headers
    headers = data_eye_tracking.columns.values
    video_names = data_eye_tracking['VIDEO_NAME']
    # remove all the video names that are '.'
    video_names = np.array(video_names[video_names != '.'])
    # pdb.set_trace()

    data_trial_sequence = load_csv_data_utf8(path_trial_sequence)
    if '2x2' in video_names[20]:
        print('2x2 found')
        process_2x2_data(data_eye_tracking, path_eye_tracking,
                         path_img_folder, data_trial_sequence, folder_path)
    else:
        print('2x2 not found')
        if '1x1' in video_names[20]:
            print('1x1 found')
            process_1x1_data(data_eye_tracking, path_eye_tracking,
                             path_img_folder, data_trial_sequence, folder_path)
        else:
            print("no valid video name found")
            # stop the program
            return
    print(f"finished processing data for {path_eye_tracking}")


main()
