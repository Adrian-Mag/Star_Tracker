import star_analyzer

path_to_image = '/home/adrian/PhD/Testing_ground/tracking/image_analysis/images/IMG_5882.CR2'

image = star_analyzer.StarImage(path_to_image)
print(image.image_name)
