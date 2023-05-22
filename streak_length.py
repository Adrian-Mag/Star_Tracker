from tracking import *

############
# PARAMETERS
############

omega = (2*mp.pi / (24*3600))               # Earth's rotation  [rad/sec]
time = np.linspace(0,60,100)           # time  [sec]
Deltas_deg = [0]               # array of angular missalignements [deg]
Deltas = (np.asarray(Deltas_deg)/180)*mp.pi # array of angular missalignements [rad]
delta_omega = -omega                       # tracking error [rad/s]
lambda0s_deg = np.asarray([45])             # latitudes of target point [deg]
lambda0s = (lambda0s_deg/180)*mp.pi         # latitudes of target point [rad]
phi0 = 0                                    # Longitude of target point [rad]
focal_length = 200                          # Focal length [mm]
pixel_size = 3.72                           # pixel size [um]


########################
# PLOT STREAK VS TIME
########################
plt.figure()
for Delta in Deltas:
    for lambda0 in lambda0s:
        # Initial position
        s0 = mp.matrix([[mp.cos(lambda0) * mp.cos(phi0)],
                        [mp.cos(lambda0) * mp.sin(phi0)],
                        [mp.sin(lambda0)]])
        
        streaks = []
        angular_deviations = []
        for t in time:
            angular_deviation = compute_angular_deviation(s(t, omega, s0), p(t, omega, delta_omega, Delta, s0), 'rad') 
            angular_deviations.append(180*60*60*angular_deviation/mp.pi)
            streaks.append(streak(angular_deviation, focal_length)) 

        plt.plot(time, np.asarray(streaks) / pixel_size)
plt.xlabel('Time [s]')
plt.ylabel('Streak [pixels]')
plt.legend([str(lambda0) for lambda0 in lambda0s_deg])
plt.show()

