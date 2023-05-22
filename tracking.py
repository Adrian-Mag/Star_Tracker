"""  """

from mpmath import mp
import numpy as np
import matplotlib
matplotlib.use('tkagg')
from matplotlib import pyplot as plt


def cross(a: mp.matrix, b: mp.matrix):
    """Cross product implementation with mp matrices

    Args:
        a (mp.matrix): first matrix
        b (mp.matrix): second matrix

    Raises:
        Exception: This function only works if the two matrices are
        3x1 matrices (vectors in R^3)

    Returns:
        mp.matrix: the cross product result
    """    
    if (a.rows == 3) and (a.cols == 1) and \
        (b.rows == 3) and (b.cols == 1): 
        return mp.matrix([[a[1]*b[2] - a[2]*b[1]],
                          [a[2]*b[0] - a[0]*b[2]],
                          [a[1]*b[2] - a[2]*b[1]]])
    else:
        raise Exception("The inputs must be 3D vectors!")


#################
# TRANSFORMATIONS
#################
def M(t: np.array, omega: float or np.array):
    """This is Earth's rotation matrix 

    Args:
        t (np.array): may be a scalar or an array of scalars
        of time points
        omega (float): Earth's rotation speed (angular)

    Returns:
        mp.matrix: the rotation matrix at a later time
    """    
    M = mp.matrix([[mp.cos(omega * t), mp.sin(omega * t), 0],
                   [-mp.sin(omega * t), mp.cos(omega * t), 0],
                   [0,                        0,           1]])
    return M


def T(Delta: float or np.array):
    """Transforms the matrix components from Earth's frame
    to the tracker's frame 

    Args:
        Delta (np.array): allginement angular error

    Returns:
        mp.matrix: transform matrix
    """    
    T = mp.matrix([[1,         0,           0     ],
                   [0, mp.cos(Delta), mp.sin(Delta)],
                   [0, -mp.sin(Delta), mp.cos(Delta)]])
    return T


def m(t: float, omega: float, delta_omega: float):
    """Rotation matrix of the tracker

    Args:
        t (np.array): time (may be scalar or array)
        omega (scalar): Earth's rotation speed
        delta_omega (scalar): error in tracker's rotation

    Returns:
        mp.matrix: tracker rotation
    """    
    # Tracker rotation matrix
    m = mp.matrix([[mp.cos(omega * t + delta_omega), mp.sin(omega * t + delta_omega), 0],
                   [-mp.sin(omega * t + delta_omega), mp.cos(omega * t + delta_omega), 0],
                   [                    0,                            0,               1]])
    return m


########
# POINTS
########
def p(t: float, omega: float, delta_omega: float, Delta: float, s0: mp.matrix):
    """Position vector of the tracked point in the tracker's frame

    Args:
        t (float): time
        omega (float): Earth's rotation
        delta_omega (float): Tracker rotation error
        Delta (float): alignement angular error
        s0 (mp.matrix): initial position

    Returns:
        mp.matrix: as mentioned
    """    
    # Tracked point 
    p = T(Delta).T * m(t, omega, delta_omega) * T(Delta) * s0

    return p


def s(t: float, omega: float, s0: mp.matrix):
    """Position vector of the target point in Earth's frame

    Args:
        t (float): time
        omega (float): Earth's rotation
        s0 (mp.matrix): initial position of target/tracked point

    Returns:
        mp.matrix: as said
    """    
    # Target
    s = M(t, omega) * s0

    return s


#########
# STREAKS
#########
def streak(angular_deviation: float, focal_length: float):
    """length of the streak in micro meters as appears on the sensor

    Args:
        angular_deviation (float): angular deviation between the target and the tracked point
        focal_length (float): focal length of the OTA 

    Returns:
        float: length of the streak in micro meters 
    """    
    # focal length must be in mm and angular deviation in rad
        
    return focal_length * 1000 * mp.tan(angular_deviation)


def compute_angular_deviation(s: mp.matrix,p: mp.matrix, unit: str):
    """Angular deviation between the target and the tracked point

    Args:
        s (mp.matrix): position of the target point in Earth's frame 
        p (mp.matrix): position of the tracked point in tracker's frame
        unit (string): deg/min/sec/rad

    Returns:
        float: as said
    """    
    
    
    Phi = mp.norm(s.T * p, 2)
    if unit == 'deg':
        return 180*(mp.acos(Phi))/mp.pi
    elif unit == 'min':
        60*180*(mp.acos(Phi))/mp.pi
    elif unit == 'sec':
        return 60*60*180*(mp.acos(Phi))/mp.pi
    elif unit == 'rad':
        return mp.acos(Phi)


############
# LOCAL BASE
############
def local_basis(s: mp.matrix):
    """Gives two perpendicular vectors that form a
    tangent basis to the celestial sphere. Projecting the 
    error vector on this subspace will yield an appproximation
    of the streak shape on the sensor

    Args:
        s (mp.matrix): tracked point position at current time

    Returns:
        list: basis
    """    
    z = mp.matrix([[0],
                   [0],
                   [1]])
    epsilon1 = z - (z.T * s)[0] * s
    epsilon2 = cross(epsilon1, s) 
    
    return [epsilon1, epsilon2]