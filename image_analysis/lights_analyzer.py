import star_analyzer
import os
import glob
import matplotlib
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np

############################
# SELECT DATA FOR PROCESSING
############################

# Specify the directory path
directory = '/mnt/c/Users/adria/Desktop/astronomy/Cepheus/elephant_trunk'

# Specify the file pattern
file_pattern = 'IMG*.CR2'

# Create the file path pattern by joining the directory and file pattern
file_path_pattern = os.path.join(directory, file_pattern)


#########
# OPTIONS
#########

# select number of processors to use
num_processors = 8

# save images with the detected stars
SAVE_STAR_CONTOURS = True

# save individual stats
SAVE_INDIVIDUAL_STATS = True

# Save summary of stats
SAVE_SUMMARY = True

#####################
# DO STATS PROCESSING
#####################

# Create summary data/stats containers
image_times = []
star_counts = []
average_star_sizes = []
median_star_sizes = []
mode_star_sizes = []
average_star_elongations = []
median_star_elongations = []
mode_star_elongations = []
average_star_angles = []
median_star_angles = []
mode_star_angles = []
average_star_deviations = []
median_star_deviations = []
mode_star_deviations = []
exposure_durations = []
# Function to process a file
def process_file(file_path):
    if os.path.isfile(file_path):
        # Convert relative path to absolute path
        absolute_file_path = os.path.abspath(file_path)

        image = star_analyzer.StarImage(absolute_file_path)
        image.compute_star_stats(full_data=True, SAVE_STAR_CONTOURS=SAVE_STAR_CONTOURS)

        if SAVE_INDIVIDUAL_STATS:
            image.plot_star_stats(SAVE_STAR_CONTOURS=SAVE_STAR_CONTOURS, SHOW=False, SAVE_STATS=True)
        
        return (image.get_date_time(),
                image.star_stats['star_count'],
                image.star_stats['avg_size'],
                image.star_stats['median_size'],
                image.star_stats['mode_size'],
                image.star_stats['avg_elongation'],
                image.star_stats['median_elongation'],
                image.star_stats['mode_elongation'],
                image.star_stats['avg_angle'],
                image.star_stats['median_angle'],
                image.star_stats['mode_angle'],
                image.star_stats['avg_deviation'],
                image.star_stats['median_deviation'],
                image.star_stats['mode_deviation'])

# Get a list of file paths
file_paths = glob.glob(file_path_pattern)

# Process files in parallel using multiprocessing.Pool with tqdm progress bar
with Pool(processes=num_processors) as pool, tqdm(total=len(file_paths)) as pbar:
    results = []
    for res in pool.imap_unordered(process_file, file_paths):
        results.append(res)
        pbar.update()

# Unzip the results
image_times, star_counts, \
average_star_sizes, median_star_sizes, mode_star_sizes, \
average_star_elongations, median_star_elongations, mode_star_elongations, \
average_star_angles, median_star_angles, mode_star_angles, \
average_star_deviations, median_star_deviations, mode_star_deviations = zip(*results)

# Sort in ascending time order
combined = zip(image_times, star_counts,
               average_star_sizes, average_star_elongations, average_star_angles, average_star_deviations,
               median_star_sizes, median_star_elongations, median_star_angles, median_star_deviations,
               mode_star_sizes, mode_star_elongations, mode_star_angles, mode_star_deviations)

sorted_combined = sorted(combined, key=lambda x: x[0])

image_times, star_counts, \
average_star_sizes, average_star_elongations, average_star_angles, average_star_deviations, \
median_star_sizes, median_star_elongations, median_star_angles, median_star_deviations, \
mode_star_sizes, mode_star_elongations, mode_star_angles, mode_star_deviations = zip(*sorted_combined)

# PLOT
fig, axs = plt.subplots(2, 2)        
plt.suptitle('Statistics')

exposure_durations = np.asarray([timedelta(seconds=60) for _ in image_times])
blank_spaces = np.asarray([timedelta(0) for _ in image_times])

# NO OF STARS
axs[0, 0].errorbar(image_times, star_counts, xerr=[exposure_durations, blank_spaces], lw=1, linestyle="--", marker='o')
axs[0, 0].set_xlabel('Time')
axs[0, 0].set_ylabel('No of Stars')
axs[0, 0].set_title('Number of stars')

# STAR SIZES
axs[0, 1].errorbar(image_times, average_star_sizes, xerr=[exposure_durations, blank_spaces], lw=1, linestyle="--", marker='o')
axs[0, 1].plot(image_times, mode_star_sizes, lw=0.5)
axs[0, 1].plot(image_times, median_star_sizes, lw=0.5)
axs[0, 1].set_xlabel('Time')
axs[0, 1].set_ylabel('Star size')
axs[0, 1].set_title('Star size')

# ELONGATIONS
axs[1, 0].errorbar(image_times, average_star_elongations, xerr=[exposure_durations, blank_spaces], lw=1, linestyle="--", marker='o')
axs[1, 0].plot(image_times, mode_star_elongations, lw=0.5)
axs[1, 0].plot(image_times, median_star_elongations, lw=0.5)
axs[1, 0].set_xlabel('Time')
axs[1, 0].set_ylabel('Star elongations')
axs[1, 0].set_title('Star elongations')

# DEVIATIONS
axs[1, 1].errorbar(image_times, average_star_deviations, xerr=[exposure_durations, blank_spaces], lw=1, linestyle="--", marker='o')
axs[1, 1].plot(image_times, mode_star_deviations, lw=0.5)
axs[1, 1].plot(image_times, median_star_deviations, lw=0.5)
axs[1, 1].set_xlabel('Time')
axs[1, 1].set_ylabel('Star deviations')
axs[1, 1].set_title('Star deviations')

plt.subplots_adjust(hspace=0.5, wspace=0.3)  # Adjust the spacing between subplots

plt.show()