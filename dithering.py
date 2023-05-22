from tracking import *

# General parameters
omega = (2*360*60 / (24*3600))               # Earth's rotation  [rad/sec]
dithering_range = 1                        # arcminutes

pause = dithering_range / omega

print(pause)