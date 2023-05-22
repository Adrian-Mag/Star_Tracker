from tracking import *

############
# PARAMETERS
############
omega = (2*mp.pi / (24*3600))
time = np.linspace(0,120,100)
Deltas_deg = [0.5]
Deltas = (np.asarray(Deltas_deg)/180)*mp.pi
delta_omega = 0.00000
lambda0s_deg = np.asarray([0, 90])
lambda0s = (lambda0s_deg/180)*mp.pi
phi0 = 0
focal_length = 200
pixel_size = 3.72 #micro meters

#############
# PLOT STREAK
############
lambda0 = lambda0s[0]
Delta = Deltas[0]

s0 = mp.matrix([[mp.cos(lambda0) * mp.cos(phi0)],
                [mp.cos(lambda0) * mp.sin(phi0)],
                [mp.sin(lambda0)]])

# Get components on the local plane in the local basis
disp_x = []
disp_y = []

angular_deviations = []

for t in time:
    s_ = s(t, omega, s0)
    p_ = p(t, omega, delta_omega, Delta, s0)
    disp =  p_ - s_
    [epsilon1, epsilon2] = local_basis(s_)
    
    projected_streak = mp.matrix([[(disp.T * epsilon1)[0]],
                                  [(disp.T * epsilon2)[0]]])
    # renormalzie 
    angular_deviation = compute_angular_deviation(s_, p_, 'rad') 
    angular_deviations.append(angular_deviation)
    
    streak_length = focal_length * mp.tan(angular_deviation)
    if (projected_streak.T * projected_streak)[0] > 1e-10:
        renormalized_projected_streak = streak_length * projected_streak / (projected_streak.T * projected_streak)[0]
        disp_x.append(renormalized_projected_streak[0])
        disp_y.append(renormalized_projected_streak[1])
    else:
        disp_x.append(0)
        disp_y.append(0)
        
     
plt.figure()
plt.scatter(disp_x,disp_y)
plt.show()

plt.figure()
plt.scatter(time, angular_deviations)
plt.show()