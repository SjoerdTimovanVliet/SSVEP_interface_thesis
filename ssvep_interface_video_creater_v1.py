import cv2
import numpy as np
from math import sin, pi
from datetime import datetime
import os


class SSVEP_Interface():

    def __init__(self):
        # set the frame size
        self.frame_size = (1080, 1920)
        # set the frame rate
        self.frame_rate = 60

        self.setup_settings()
        self.create_stimuli()
        # set the extension of the video
        self.extension = '.mp4'

        if self.extension == '.mp4':
            # define the codec of the video as H264
            fourcc_type = 'avc1'
            self.fourcc = cv2.VideoWriter_fourcc(*fourcc_type)

        elif self.extension == '.avi':
            # define the codec of the video as MPEG
            self.fourcc = cv2.VideoWriter_fourcc(*'MPEG')
        # save the current date and time for folder name
        self.date_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    def calculate_possible_frequencies(self) -> list:
        """ Calculates the number of possible frequencies for the stimuli that are not multiples of each other.
        The frequencies are alos whole number and not decimals as this would allow different methods to evoket the wave (square or sine)

        Returns:
            list: list of possible frequencies
        """
        # calculate possible frequencies
        possible_frequencies = []
        for i in range(1, int(self.frame_rate/2)):
            # check if the number is a whole number
            if self.frame_rate % i == 0:
                possible_frequencies.append(i)
        print(f"Possible frequencies: {possible_frequencies}")
        # find 4 frequencies in the list of possible_frequencies that are unique and not multiples of each other
        frequencies = []
        # loop through the list of possible frequencies and check if the number is a multiple of the previous number
        for i in range(len(possible_frequencies)):

            if len(frequencies) == 4:
                break
            # loop through the list of frequencies and check if the number is already in the list
            if possible_frequencies[i] not in frequencies:
                frequencies.append(possible_frequencies[i])
                for j in range(i+1, len(possible_frequencies)):
                    # check if the number is a multiple of the previous number
                    if possible_frequencies[j] % possible_frequencies[i] == 0:
                        break
                    # check if the number is the last number in the list
                    if j == len(possible_frequencies)-1:
                        frequencies.append(possible_frequencies[i])
        print("Possible frequencies after removing multiples: ", frequencies)
        return possible_frequencies

    def setup_settings(self):
        """ Sets up the settings for the stimuli
        """
        # set frequencies for the stimuli
        self.frequencies = [8, 13, 22, 29]
        #  red, green, white
        self.colors = ['red', 'green', 'white']
        # pixel size of the stimuli
        self.pixel_surface = [10000, 20000, 30000]
        # types of shapes
        self.shapes = ['circles', 'squares', 'triangles']

        # draw thickness of the shapes
        self.thickness = -1  # -1 to fill circle

    def create_random_order_of_stimuli_settings_1x1(self) -> list:
        """ Creates a random order of the stimuli settings

        Returns:
            list: list of tuples with the indices of the settings
        """
        # create a list of indices for all settings
        frequency_indices = list(range(len(self.frequencies)))
        color_indices = list(range(len(self.colors)))
        pixel_surface_indices = list(range(len(self.pixel_surface)))
        shape_indices = list(range(len(self.shapes)))
        # calculate the number of stimuli
        calculate_number_of_stimuli = len(
            frequency_indices)*len(color_indices)*len(pixel_surface_indices)*len(shape_indices)
        print(f"Number of stimuli in one block: {calculate_number_of_stimuli}")
        # shuffle the indices of all settings
        np.random.shuffle(frequency_indices)
        np.random.shuffle(color_indices)
        np.random.shuffle(pixel_surface_indices)
        np.random.shuffle(shape_indices)
        # create a list of tuples with the indices of all the different combinations of settings
        indices = []
        for k in pixel_surface_indices:
            for j in color_indices:
                for i in frequency_indices:
                    for l in shape_indices:
                        indices.append((k, j, i, l))
        # shuffle the list of combinations
        np.random.shuffle(indices)
        return indices

    def create_random_order_of_stimuli_settings_2x2(self) -> list:
        """ Creates a random order of the stimuli settings

        Returns:
            list: list of tuples with the indices of the settings
        """
        # create a list of indices for all settings
        frequency_indices = list(range(len(self.frequencies)))
        color_indices = list(range(len(self.colors)))
        pixel_surface_indices = list(range(len(self.pixel_surface)))
        shape_indices = [1]  # only squares
        center_coordinates_indices = list(range(len(self.center_coordinates)))
        # calculate the number of stimuli
        calculate_number_of_stimuli = len(frequency_indices)*len(color_indices)*len(
            pixel_surface_indices)*len(shape_indices)*len(center_coordinates_indices)
        print(f"Number of stimuli in one block: {calculate_number_of_stimuli}")
        # shuffle the indices of all settings
        np.random.shuffle(frequency_indices)
        np.random.shuffle(color_indices)
        np.random.shuffle(pixel_surface_indices)
        np.random.shuffle(shape_indices)
        np.random.shuffle(center_coordinates_indices)
        # create a list of tuples with the indices of the settings
        indices = []
        for k in pixel_surface_indices:
            for j in color_indices:
                for i in frequency_indices:
                    for l in shape_indices:
                        # 1 means it is always squares
                        indices.append((k, j, i, 1))
        # shuffle the list of combinations
        np.random.shuffle(indices)
        return indices

    def create_stimuli(self):
        """Creates the stimuli by calculating the coordinates for the stimuli
        """
        # Get center coordinates of the 4 quadrants of the screen
        width_1 = int(self.frame_size[0]/4)
        height_1 = int(self.frame_size[1]/4)
        width_2 = int(self.frame_size[0]/4*3)
        height_2 = int(self.frame_size[1]/4*3)
        # create a list of tuples with the center coordinates of the 4 quadrants
        self.center_coordinates = [
            (height_1, width_1), (height_1, width_2), (height_2, width_1), (height_2, width_2)]

    def _draw_circles(self, image: np.ndarray, pixel_surface: int, center_coordinates: list, color_tuples: list) -> np.ndarray:
        """ Draws circles on the image
        Args:
            image (np.ndarray): The image on which the circles are drawn
            pixel_surface (int): The size of the circles in pixels
            center_coordinates (list): the coordinates of the center of the circles
            color_tuples (list): the colors of the circles

        Returns:
            np.ndarray: The image with the circles drawn on it
        """
        # calculate radius of circle
        radius = int(np.sqrt(pixel_surface/pi))
        #  draw circles
        for i, center_coordinate in enumerate(center_coordinates):
            # unpack color tuple
            color_tuple = color_tuples[i]
            image = cv2.circle(image, center_coordinate,
                               radius, color_tuple, self.thickness)
        return image

    def _draw_squares(self, image: np.ndarray, pixel_surface: int, center_coordinates: list, color_tuples: list) -> np.ndarray:
        """ Draws squares  on the image
        Args:
            image (np.ndarray): The image on which the squares are drawn
            pixel_surface (int): The size of the squares in pixels
            center_coordinates (list): the coordinates of the center of the squares
            color_tuples (list): the colors of the squares

        Returns:
            np.ndarray: The image with the squares drawn on it
        """
        # calculate side length of square
        side_length = int(np.sqrt(pixel_surface))
        # create squares
        for i, center_coordinate in enumerate(center_coordinates):
            # draw all squares. The coordinates are measured in integer values.
            # unpack color tuple
            color_tuple = color_tuples[i]
            coordinate_1 = (int(
                center_coordinate[0]-side_length//2), int(center_coordinate[1]-side_length//2))
            coordinate_2 = (int(
                center_coordinate[0]+side_length//2), int(center_coordinate[1]+side_length//2))
            image = cv2.rectangle(image, coordinate_1,
                                  coordinate_2, color_tuple, self.thickness)

        return image

    def _draw_triangles(self, image: np.ndarray, pixel_surface: int, center_coordinates: list, color_tuples: list) -> np.ndarray:
        """ Draws triangles on the image
        Args:
            image (np.ndarray): The image on which the triangles are drawn
            pixel_surface (int): The size of the triangles in pixels
            center_coordinates (list): the coordinates of the center of the triangles
            color_tuples (list): the colors of the triangles

        Returns:
            np.ndarray: The image with the triangles drawn on it
        """
        # us the pixel surface to derive corner coordinates of the iscoceles triangle. All sides of the triangle are equal
        # calculate side length of triangle
        diagonal = np.sqrt(pixel_surface*8/np.sqrt(3))
        # half base length
        half_base_length = diagonal/2
        height = diagonal/2*np.sqrt(3)

        # create triangles
        self.triangle_center_coordinates = []
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
                correction = center_coordinate[1] - \
                    triangle_center_coordinate[1]
                coordinate_1 = (coordinate_1[0], coordinate_1[1]+correction)
                coordinate_2 = (coordinate_2[0], coordinate_2[1]+correction)
                coordinate_3 = (coordinate_3[0], coordinate_3[1]+correction)

            elif triangle_center_coordinate[1] < center_coordinate[1]:
                correction = triangle_center_coordinate[1] - \
                    center_coordinate[1]
                coordinate_1 = (coordinate_1[0], coordinate_1[1]-correction)
                coordinate_2 = (coordinate_2[0], coordinate_2[1]-correction)
                coordinate_3 = (coordinate_3[0], coordinate_3[1]-correction)

            # save the center coordinates of the triangles
            self.triangle_center_coordinates.append(
                (int((coordinate_1[0]+coordinate_2[0]+coordinate_3[0])/3), int((coordinate_1[1]+coordinate_2[1]+coordinate_3[1])/3)))
            # draw triangle
            triangle_cnt = np.array([coordinate_1, coordinate_2, coordinate_3])
            image = cv2.drawContours(
                image, [triangle_cnt], 0, color_tuple, self.thickness)

        return image

    def _calculate_screen_color_sinusoidal(self, frequency: float, frame_number: int) -> int:
        """ Calculates the color of the screen for a given frequency and frame number
        Args:
            frequency (float): The frequency of the sinusoidal color change
            frame_number (int): The frame number

        Returns:
            int: The color of the screen
        """
        tmp = 1/2*(1+sin(2*pi*frequency*(frame_number/self.frame_rate)))
        color = int(round(255*tmp))
        return color

    def create_video_1x1(self):
        """ Creates a video with 1 stimulus on the screen
        """
        # create a video with 1 stimulus on the screen
        length_of_video = 4  # seconds
        inter_trial_time = 1  # seconds
        # calculate number of frames for the video
        number_of_frames = length_of_video * self.frame_rate
        # The inter trial interval which is replace for a png but can be replaced by a video (OPTIONAL)
        number_of_frames_inter_trial = inter_trial_time * self.frame_rate
        # calculate center coordinate of the screen
        center_coordinate = (
            int(self.frame_size[1]/2), int(self.frame_size[0]/2))
        # create folder name and location
        folder = f"stimuli_videos/1x1_stimuli_{self.frequencies[0]}_{self.frequencies[1]}_{self.frequencies[2]}_{self.frequencies[3]}_{self.extension[1:]}_{self.date_time}"
        # create folder if it does not exist
        if not os.path.exists(folder):
            os.makedirs(folder)

        # create the order of trials
        order_settings = self.create_random_order_of_stimuli_settings_1x1()
        # for each combination of settings
        for combination in order_settings:
            # get the indices of the settings
            pixel_surface_index, color_mode_index, frequency_index, shape_index = combination
            # get the values of the settings
            pixel_surface = self.pixel_surface[pixel_surface_index]
            color_mode = self.colors[color_mode_index]
            frequency = self.frequencies[frequency_index]
            shape = self.shapes[shape_index]
            # create the video name for the trial and photo name for inter trial
            video_name_trial = f'{folder}/1x1_pixel_surface_' + str(pixel_surface) + "_color_mode_" + str(
                color_mode) + "_frequency_" + str(frequency) + "_shape_" + str(shape) + self.extension
            video_name_inter_trial = f"{folder}/1x1_pixel_surface_" + str(pixel_surface) + "_color_mode_" + str(
                color_mode) + "_frequency_" + str(frequency) + "_shape_" + str(shape) + "_inter_trial" + self.extension
            # get the values of the settings
            pixel_surface = self.pixel_surface[pixel_surface_index]
            color_mode = self.colors[color_mode_index]
            frequency = self.frequencies[frequency_index]
            shape = self.shapes[shape_index]

            # create a list of frames for the trial and inter trial to save the frames
            frame_list_trial = []
            frame_list_inter_trial = []
            # create the trial video
            for frame_number in range(number_of_frames):
                # create a black image
                image = np.zeros(self.frame_size, np.uint8)
                # convert color to from BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # calculate the color of the screen
                color = self._calculate_screen_color_sinusoidal(
                    frequency, frame_number)
                # create a color tuple
                if color_mode == 'red':
                    # in BGR space
                    color_tuple = (0, 0, color)
                elif color_mode == 'green':
                    color_tuple = (0, color, 0)
                elif color_mode == 'white':
                    color_tuple = (color, color, color)

                # draw the stimulus
                if shape == "circles":
                    image = self._draw_circles(image, pixel_surface, [
                                               center_coordinate], [color_tuple])
                elif shape == "squares":
                    image = self._draw_squares(image, pixel_surface, [
                                               center_coordinate], [color_tuple])
                elif shape == "triangles":
                    image = self._draw_triangles(image, pixel_surface, [
                                                 center_coordinate], [color_tuple])
                # append the frame to the list
                frame_list_trial.append(image)

            # create the inter trial video (OPTIONAL). currently it is a png
            for frame_number in range(1):
                # create a black image
                image = np.zeros(self.frame_size, np.uint8)
                # convert color to from BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # calculate the color of the screen
                color = self._calculate_screen_color_sinusoidal(frequency, 0)
                # create a color tuple
                if color_mode == 'red':
                    # in BGR space
                    color_tuple = (0, 0, color)
                elif color_mode == 'green':
                    color_tuple = (0, color, 0)
                elif color_mode == 'white':
                    color_tuple = (color, color, color)

                # draw the stimulus
                if shape == "circles":
                    image = self._draw_circles(image, pixel_surface, [
                                               center_coordinate], [color_tuple])
                elif shape == "squares":
                    image = self._draw_squares(image, pixel_surface, [
                                               center_coordinate], [color_tuple])
                elif shape == "triangles":
                    image = self._draw_triangles(image, pixel_surface, [
                                                 center_coordinate], [color_tuple])
                # append the frame to the list of inter trial
                frame_list_inter_trial.append(image)

            # create video writer for the trial
            self.video_writer = cv2.VideoWriter(
                video_name_trial, self.fourcc, self.frame_rate, (self.frame_size[1], self.frame_size[0]))
            # write the frames to the video
            for frame in frame_list_trial:
                self.video_writer.write(frame)
            # close video writer
            self.video_writer.release()

            # replace video_name_inter_trial by .png to save the images
            photo_name_inter_trial = video_name_inter_trial.replace(
                self.extension, ".png")
            for frame in frame_list_inter_trial:
                # save frame as png
                cv2.imwrite(photo_name_inter_trial, frame)

    def create_random_order_3_neighbouring_frequencies(self, frequency: float, delta_frequency: float) -> list:
        """ create a list of 3 frequencies with 1 frequency being the same as the frequency argument and the other 2 frequencies being the frequency argument 
        plus or minus delta_frequency. The list is shuffled.
        Args:
            frequency (float): frequency of the stimulus
            delta_frequency (float):  difference between the frequencies

        Returns:
            list: list of 3 frequencies
        """
        # create a list of 3 frequencies
        frequency_list = [frequency-delta_frequency,
                          frequency+delta_frequency, frequency+delta_frequency*2]
        # shuffle the list
        np.random.shuffle(frequency_list)
        return frequency_list

    def update_screen_by_frequency(self, image: np.ndarray, frequency: float, center_coordinate: tuple, color_mode: str, shape: str, pixel_surface: int, frame_number=0) -> np.ndarray:
        """ update the screen by the frequency. The frequency is used to calculate the color of the screen. The color of the screen is used to draw the stimulus on the screen.

        Args:
            image (np.ndarray): image to draw on
            frequency (float): frequency of the stimulus
            center_coordinate (tuple): center coordinate of the stimulus
            color_mode (str): color mode of the stimulus
            shape (str): shape of the stimulus
            pixel_surface (int): pixel surface of the stimulus
            frame_number (int, optional): frame number. Defaults to 0.

        Returns:
            np.ndarray: updated image
        """
        # calculate the color of the screen
        color = self._calculate_screen_color_sinusoidal(
            frequency, frame_number)

        # create a color tuple
        if color_mode == 'red':
            # in BGR space
            color_tuple = (0, 0, color)
        elif color_mode == 'green':
            color_tuple = (0, color, 0)
        elif color_mode == 'white':
            color_tuple = (color, color, color)

        # draw the stimulus
        if shape == 'circles':
            image = self._draw_circles(image, pixel_surface, [
                                       center_coordinate], [color_tuple])
        elif shape == 'squares':
            image = self._draw_squares(image, pixel_surface, [
                                       center_coordinate], [color_tuple])
        elif shape == 'triangles':
            image = self._draw_triangles(image, pixel_surface, [
                                         center_coordinate], [color_tuple])

        return image

    def create_video_2x2(self):
        """ create a video with 4 stimuli on the screen
        """
       # create a video with 4 stimuli on the screen
        length_of_video = 4  # seconds
        inter_trial_time = 2  # seconds
        # calculate the number of frames for the trial video and the inter trial video
        number_of_frames = length_of_video * self.frame_rate
        number_of_frames_inter_trial = inter_trial_time * self.frame_rate  # OPTIONAL
        # define the radius of the fixation circle
        self.fixation_circle_radius = 10
        # define the delta frequency with which the frequencies are changed
        delta_frequency = 0.3  # Hz
        # create a  older for the experiment videos
        folder = f"stimuli_videos/2x2_stimuli_{self.frequencies[0]}_{self.frequencies[1]}_{self.frequencies[2]}_{self.frequencies[3]}_delta_{delta_frequency}_{self.extension[1:]}_{self.date_time}"
        # create the folder if it does not exist
        if not os.path.exists(folder):
            os.makedirs(folder)

        # create a list of all possible combinations of the stimuli in a random order
        order_settings = self.create_random_order_of_stimuli_settings_2x2()

        # for each stimuli location
        for stimuli_center_coordinate in self.center_coordinates:
            # for each combination of the stimuli settings
            for combination in order_settings:
                # extract the stimuli settings
                pixel_surface_index = combination[0]
                color_mode_index = combination[1]
                frequency_index = combination[2]
                shape_index = combination[3]

                # get the stimuli settings
                pixel_surface = self.pixel_surface[pixel_surface_index]
                color_mode = self.colors[color_mode_index]
                frequency = self.frequencies[frequency_index]
                shape = self.shapes[shape_index]

                # create a list of 3 frequencies
                frequency_list_neigbours = self.create_random_order_3_neighbouring_frequencies(
                    frequency, delta_frequency)
                # create a list of all frequencies
                all_frequency_list = [frequency, frequency_list_neigbours[0],
                                      frequency_list_neigbours[1], frequency_list_neigbours[2]]

                # get the other 3 stimuli center coordinates that are not the target stimulus center coordinate
                other_stimuli_center_coordinates = self.center_coordinates.copy()
                other_stimuli_center_coordinates.remove(
                    stimuli_center_coordinate)

                # create video name for the trial video and the inter trial video
                video_name_trial = f"{folder}/2x2_pixel_surface_" + str(pixel_surface) + "_color_mode_" + str(color_mode) + "_frequency_" + str(
                    frequency) + "_shape_" + str(shape) + "_coordinate_" + str(stimuli_center_coordinate) + self.extension
                video_name_inter_trial = f"{folder}/2x2_pixel_surface_" + str(pixel_surface) + "_color_mode_" + str(color_mode) + "_frequency_" + str(
                    frequency) + "_shape_" + str(shape) + "_coordinate_" + str(stimuli_center_coordinate) + "_inter_trial" + self.extension

                # create a list of frames for the trial video and the inter trial video
                list_of_frames_inter_trial = []
                list_of_frames = []

                # for the number of frames in the inter trial video. OPTIONAL. Currently it is a png image
                for frame_i in range(1):
                    # create a black rgb image
                    image = np.zeros(
                        (self.frame_size[0], self.frame_size[1], 3), np.uint8)
                    # convert image from BGR to RGB
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    # add stimuli to image
                    for i, frequency_i in enumerate(frequency_list_neigbours):
                        # get the center coordinate of the stimulus
                        center_coordinate = other_stimuli_center_coordinates[i]
                        # add the stimulus to the image
                        image = self.update_screen_by_frequency(
                            image, frequency_i, center_coordinate, color_mode, shape, pixel_surface, 0)

                    # add the target stimulus to the image
                    image = self.update_screen_by_frequency(
                        image, frequency, stimuli_center_coordinate, color_mode, shape, pixel_surface, 0)
                    # add fixation circle
                    image = cv2.circle(image, stimuli_center_coordinate,
                                       self.fixation_circle_radius, (255, 0, 0), self.thickness)
                    # add the image to the list of frames
                    list_of_frames_inter_trial.append(image)

                # for the number of frames in the trial video
                for frame_number in range(number_of_frames):
                    # create a black rgb image
                    image = np.zeros(
                        (self.frame_size[0], self.frame_size[1], 3), np.uint8)
                    # convert image from BGR to RGB
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    # add neigbouring stimuli to image
                    for i, frequency_i in enumerate(frequency_list_neigbours):
                        # get the center coordinate of the stimulus
                        center_coordinate = other_stimuli_center_coordinates[i]
                        # add the stimulus to the image
                        image = self.update_screen_by_frequency(
                            image, frequency_i, center_coordinate, color_mode, shape, pixel_surface, frame_number)

                    # add the target stimulus to the image
                    image = self.update_screen_by_frequency(
                        image, frequency, stimuli_center_coordinate, color_mode, shape, pixel_surface, frame_number)

                    # save frames for video writer
                    list_of_frames.append(image)

                print(
                    f"length of list_of_frames: {len(list_of_frames)} and length of list_of_frames_inter_trial: {len(list_of_frames_inter_trial)}")

                # create inter trial video. OPTIONAL. Currently it is a png image
                photo_name_inter_trial = video_name_inter_trial.replace(
                    self.extension, ".png")
                for frame in list_of_frames_inter_trial:
                    # save frame as png
                    cv2.imwrite(photo_name_inter_trial, frame)

                # create video fir trial
                self.video_writer = cv2.VideoWriter(
                    video_name_trial, self.fourcc, self.frame_rate, (self.frame_size[1], self.frame_size[0]))
                for frame in list_of_frames:
                    self.video_writer.write(frame)
                # close video writer
                self.video_writer.release()


if __name__ == '__main__':
    # create interface
    interface = SSVEP_Interface()
    # create videos
    interface.create_video_1x1()
    interface.create_video_2x2()
