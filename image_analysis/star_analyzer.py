import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS
from datetime import datetime
import matplotlib
matplotlib.use('tkagg')
from matplotlib import pyplot as plt
import os


class StarImage:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image_dir = os.path.join(*(image_path.split('/')[0:-1]))
        self.image_name = image_path.split('/')[-1].split('.')[0]
        self.image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        self.metadata = self._get_image_metadata()  
        self.stars = None
        self.star_stats = None
        self.star_stats_full = None
        
    def get_date_time(self):
        """Extracts date time metadata

        Returns:
            datetime object: date and time`
        """        
        datetime_object = datetime.strptime(self.metadata['DateTime'], "%Y:%m:%d %H:%M:%S")
        return datetime_object


    import numpy as np

    def plot_star_stats(self, SAVE_STAR_CONTOURS: bool = False, SHOW: bool = True, SAVE_STATS: bool = False):
        """Plots the stats
        """        
        
        if self.star_stats_full is None:
            self.compute_star_stats(SAVE_STAR_CONTOURS=SAVE_STAR_CONTOURS, full_data=True)

        fig, axs = plt.subplots(2, 2)
        
        plt.suptitle('Number of stars: ' + str(self.star_stats['star_count']))
        
        no_of_bins = int(np.sqrt(len(self.stars)))
        
        # ELONGATIONS
        axs[0, 0].hist(self.star_stats_full['elongations'], bins=no_of_bins)
        axs[0, 0].set_xlabel('Star elongation')
        axs[0, 0].set_ylabel('No of Stars')
        axs[0, 0].set_title('Distribution of star elongation')
        # add summary statistics
        avg_elongation = self.star_stats['avg_elongation']
        median_elongation = self.star_stats['median_elongation']
        mode_elongation = self.star_stats['mode_elongation']
        axs[0, 0].axvline(avg_elongation, 
                        color='r', linestyle='--', label='Mean: ' + str(round(avg_elongation, 1)))
        axs[0, 0].axvline(median_elongation, 
                        color='g', linestyle='--', label='Median: ' + str(round(median_elongation, 1)))
        axs[0, 0].axvline(mode_elongation, 
                        color='b', linestyle='--', label='Mode: ' + str(round(mode_elongation, 1)))
        axs[0, 0].legend()
        
        # SIZES
        data = self.star_stats_full['sizes']
        hist, bin_edges = np.histogram(data, bins=np.arange(int(round(min(data))), int(round(max(data))) + 2, 1))
        axs[0, 1].bar(bin_edges[:-1], hist, width=1)
        axs[0, 1].set_xticks(np.arange(min(data), max(data) + 1, 1))
        axs[0, 1].set_xlabel('Star sizes [pixels]')
        axs[0, 1].set_ylabel('No of Stars')
        axs[0, 1].set_title('Distribution of star sizes')
        # Limit the number of tick labels to a maximum of 10
        n_ticks = min(len(bin_edges), 10)
        tick_indices = np.linspace(0, len(bin_edges) - 1, n_ticks, dtype=int)
        tick_labels = bin_edges[tick_indices]
        axs[0, 1].set_xticks(tick_labels)
        # add summary statistics
        avg_size = self.star_stats['avg_size']
        median_size = self.star_stats['median_size']
        mode_size = self.star_stats['mode_size']
        axs[0, 1].axvline(avg_size, 
                        color='r', linestyle='--', label='Mean: ' + str(round(avg_size, 1)))
        axs[0, 1].axvline(median_size, 
                        color='g', linestyle='--', label='Median: ' + str(round(median_size, 1)))
        axs[0, 1].axvline(mode_size, 
                        color='b', linestyle='--', label='Mode: ' + str(round(mode_size, 1)))
        axs[0, 1].legend()
        
        # ANGLES
        axs[1, 0].hist(self.star_stats_full['angles'], bins=no_of_bins)
        axs[1, 0].set_xlabel('Star angles')
        axs[1, 0].set_ylabel('No of Stars')
        axs[1, 0].set_title('Distribution of star angles')
        # add summary statistics
        avg_angle = self.star_stats['avg_angle']
        median_angle = self.star_stats['median_angle']
        mode_angle = self.star_stats['mode_angle']
        axs[1, 0].axvline(avg_angle, 
                        color='r', linestyle='--', label='Mean: ' + str(round(avg_angle, 1)))
        axs[1, 0].axvline(median_angle, 
                        color='g', linestyle='--', label='Median: ' + str(round(median_angle, 1)))
        axs[1, 0].axvline(mode_angle, 
                        color='b', linestyle='--', label='Mode: ' + str(round(mode_angle, 1)))
        axs[1, 0].legend()
        
        # DEVIATIONS
        data = self.star_stats_full['deviations']
        hist, bin_edges = np.histogram(data, bins=np.arange(int(round(min(data))), int(round(max(data))) + 2, 1))
        axs[1, 1].bar(bin_edges[:-1], hist, width=1)
        axs[1, 1].set_xticks(np.arange(min(data), max(data) + 1, 1))
        axs[1, 1].set_xlabel('Star deviations [pixels]')
        axs[1, 1].set_ylabel('No of Stars')
        axs[1, 1].set_title('Distribution of star deviations')
        # Limit the number of tick labels to a maximum of 10
        n_ticks = min(len(bin_edges), 10)
        tick_indices = np.linspace(0, len(bin_edges) - 1, n_ticks, dtype=int)
        tick_labels = bin_edges[tick_indices]
        axs[1, 1].set_xticks(tick_labels)
        # add summary statistics
        avg_deviation = self.star_stats['avg_deviation']
        median_deviation = self.star_stats['median_deviation']
        mode_deviation = self.star_stats['mode_deviation']
        axs[1, 1].axvline(avg_deviation, 
                        color='r', linestyle='--', label='Mean: ' + str(round(avg_deviation, 1)))
        axs[1, 1].axvline(median_deviation, 
                        color='g', linestyle='--', label='Median: ' + str(round(median_deviation, 1)))
        axs[1, 1].axvline(mode_deviation, 
                        color='b', linestyle='--', label='Mode: ' + str(round(mode_deviation, 1)))
        axs[1, 1].legend()
        
        plt.subplots_adjust(hspace=0.5, wspace=0.3)  # Adjust the spacing between subplots
        
        if SAVE_STATS is True:
            plt.savefig('/' + str(self.image_dir) + '/' + str(self.image_name) + '_STATS.png')
        
        if SHOW is True:
            plt.show()
        else: 
            plt.close()


    def _get_image_metadata(self):
        """Gets image metadata with PIL. The tags
        are translated using the ExifTags 

        Returns:
            dictionary: dictionary of metadata
        """
        
        pil_image = Image.open(self.image_path)
        exifdata = pil_image.getexif()
        
        metadata = {}
        
        for tag_id in exifdata:
            # get the tag name, instead of human unreadable tag id
            tag = TAGS.get(tag_id, tag_id)
            data = exifdata.get(tag_id)
            # decode bytes 
            if isinstance(data, bytes):
                data = data.decode()
            metadata[tag] = data
            
        return metadata 


    def compute_star_stats(self, full_data: bool = False, SAVE_STAR_CONTOURS: bool = False):
        """Computes the stats of the stars such as elongations, 
        average elongation, std in elongation, number of stars.

        Args:
            full_data (bool, optional): whether we save the full stats
            or just the summary stats. Defaults to False.
        """        
        # detect the stars
        self._detect_stars(SAVE_STAR_CONTOURS)
        
        # Compute the total elongation and count of stars
        star_count = 0
        total_elongation = 0
        total_size = 0
        total_angle = 0
        total_deviation = 0
        
        # Compute the elongation/size/angle of each detected star
        elongations = []
        sizes = []
        angles = []
        deviations = []
        
        for star in self.stars:
            _star_stats = self._compute_star_shape(star)
            elongation = _star_stats[2]
            size = _star_stats[1]

            length = _star_stats[0]
            angle = _star_stats[3]
            if elongation is not None:
                deviation = length - size
                
                star_count += 1
                total_elongation += elongation
                total_size += size
                total_angle += angle
                total_deviation += deviation
                
                elongations.append(elongation)
                sizes.append(size)
                angles.append(angle)
                deviations.append(deviation)

        # Calculate the average stats
        average_elongation = total_elongation / star_count
        average_size = total_size / star_count
        average_angle = total_angle / star_count
        average_deviation = total_deviation / star_count
        
        #calculate medians and modes
        median_elongation = np.median(elongations)
        mode_elongation = np.argmax(np.bincount(np.asarray(elongations).astype(int)))
        median_size = np.median(sizes)
        mode_size = np.argmax(np.bincount(np.asarray(sizes).astype(int)))
        median_angle = np.median(angles)
        mode_angle = np.argmax(np.bincount(np.asarray(angles).astype(int)))
        median_deviation = np.median(deviations)
        mode_deviation = np.argmax(np.bincount(np.asarray(deviations).astype(int)))
        
        if full_data is False:
            self.star_stats =  {'star_count': star_count, 
                                'avg_elongation': average_elongation,
                                'median_elongation': median_elongation,
                                'mode_elongation': mode_elongation,
                                'avg_size': average_size,
                                'median_size': median_size,
                                'mode_size': mode_size,
                                'avg_angle': average_angle,
                                'median_angle': median_angle,
                                'mode_angle': mode_angle,
                                'avg_deviation': average_deviation,
                                'mode_deviation': mode_deviation,
                                'median_deviation': median_deviation}
        else:
            self.star_stats =  {'star_count': star_count, 
                                'avg_elongation': average_elongation,
                                'median_elongation': median_elongation,
                                'mode_elongation': mode_elongation,
                                'avg_size': average_size,
                                'median_size': median_size,
                                'mode_size': mode_size,
                                'avg_angle': average_angle,
                                'median_angle': median_angle,
                                'mode_angle': mode_angle,
                                'avg_deviation': average_deviation,
                                'mode_deviation': mode_deviation,
                                'median_deviation': median_deviation}
            self.star_stats_full =  {'star_count': star_count, 
                                    'elongations': elongations,
                                    'avg_elongation': average_elongation,
                                    'median_elongation': median_elongation,
                                    'mode_elongation': mode_elongation,
                                    'sizes': sizes,
                                    'avg_size': average_size,
                                    'median_size': median_size,
                                    'mode_size': mode_size,
                                    'angles': angles,
                                    'avg_angle': average_angle,
                                    'median_angle': median_angle,
                                    'mode_angle': mode_angle,
                                    'deviations': deviations,
                                    'avg_deviation': average_deviation,
                                    'mode_deviation': mode_deviation,
                                    'median_deviation': median_deviation}


    def _detect_stars(self, SAVE_STAR_CONTOURS: bool = False):
        """Takes in the image data and uses canny edge detection
        to find the stars. The stars are returned as a list of
        contours.
        """        

        if self.stars is not None:
            # If the stars have been detected already, then nothing should 
            # be done anymore
            return 

        # PARAMETERS
        x_center = len(self.image[0]) // 2
        y_center = len(self.image[:,0]) // 2
        crop_x = 3000
        crop_y = 3000
        
        _gaussian_blur_size = 3
        _minimum_gradient = 10
        _maximum_gradient = 200
        _min_area = 10  # Minimum contour area to consider as a star (adjust as needed)
        _max_area = 1000  # Maximum contour area to consider as a star (adjust as needed)
        _contour_thickness = 1
        
        # Crop the image based on the specified parameters
        cropped_image = self.image[y_center - crop_y // 2:y_center + crop_y // 2, x_center - crop_x // 2:x_center + crop_x // 2]

        if cropped_image.size == 0:
            # If the cropped image is empty, return without further processing
            return

        # Convert the cropped image to grayscale
        gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (_gaussian_blur_size, _gaussian_blur_size), 0)

        # Increase contrast using contrast stretching
        stretched = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)   

        # Apply Canny edge detection 
        edges = cv2.Canny(stretched, _minimum_gradient, _maximum_gradient)  # Adjust the threshold values as needed

        # Find contours of the edges
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours based on area (size) to exclude noise
        stars = [cnt for cnt in contours if cv2.contourArea(cnt) > _min_area and cv2.contourArea(cnt) < _max_area and len(cnt) > 4]

        if SAVE_STAR_CONTOURS is True:
            # Create a copy of the cropped image
            image_with_contours = np.copy(cropped_image)

            # Draw the contours on the image
            cv2.drawContours(image_with_contours, stars, -1, (0, 255, 0), _contour_thickness)  # Green color, thickness = 1

            # Save the image with contours
            cv2.imwrite('/' + self.image_dir + '/' + self.image_name + '_contours.jpg', image_with_contours)

        self.stars = stars



    def _compute_star_shape(self, star):
        """Takes in a contour (a star) and computes
        the elongation of the star by fitting an ellipse

        Args:
            star (contour): star contour computed with _detect_stars

        Returns:
            float: elongation of star
        """        

        # Fit an ellipse to the star contour
        ellipse = cv2.fitEllipse(star)
        (center, axes, angle) = ellipse

        # Calculate the major and minor axis lengths
        major_axis = max(axes)
        minor_axis = min(axes)

        # Calculate the elongation as the ratio of major axis length to minor axis length
        elongation = major_axis / minor_axis

        return (major_axis, minor_axis, elongation, angle)
