import numpy as np
from numpy import sqrt
from scipy.stats import truncnorm
from copy import copy

def get_lead_traj(lead_init,n,a,b,T,v0,l,s0,s1,delt,traj_len, dt):
  lead_traj = np.zeros((traj_len, 2))
  lead_traj[0][:] = lead_init
  x = lead_init.copy()
  # x = np.zeros((2,1))
  # x[0] = lead_init[0]
  # x[1] = lead_init[1]

  for t in range(1, traj_len):
    f = np.zeros(n)
    s = np.zeros(n)

    alpha0 = 0.0
    alpha1 = 0.0
    alpha2 = 0.0
    alpha3 = 0.0
    alpha4 = 0.0
    alpha5 = 0.0
    alpha6 = 0.0
    alpha7 = 0.0
    alpha8 = 0.0

    va = v0/4
    xinf = 300
    s[0] = xinf - l - x[0]
    ss = s[0]**2
    alpha0 = -a/(v0**delt)
    alpha1 = -1/(4*b*(ss))
    alpha2 = (va/(2*b*ss)) - (sqrt(a)*T)/(sqrt(b)*ss)
    alpha3 = (sqrt(a)*s1)/(sqrt(b*v0)*ss)
    alpha4 = ((sqrt(a)*s0)/(sqrt(b)*ss)) - ((a*T**2)/ss)-((va*va)/(4*b*ss))-((sqrt(a)*va*T)/(sqrt(b)*ss))
    alpha5 = ((-2*sqrt(a)*s0*s1*va)/(sqrt(b*v0)*ss))-((2*a*s1*T)/(sqrt(v0)*ss))
    alpha6 = ((-2*a*s1**2)/(v0*ss)) - ((sqrt(a)*s0*va)/(sqrt(b)*ss)) - ((2*a*s0*T)/ss)
    alpha7 = ((-2*a*s0*s1)/(sqrt(v0)*ss))
    alpha8 = ((-a*s0**2)/ss)+a

    f[0] = x[1]   # set position of first/leader car = velocity of first/leader car for original ode
    f[1] = alpha0*x[1]**delt + alpha1*x[1]**4 + alpha2*x[1]**3 + alpha3*x[1]**(5/2)+alpha4*x[1]**2 + alpha5*x[1]**(3/2) + alpha6*x[1] + alpha7*x[1]**(1/2) + alpha8

    # print(x)
    # print(f*dt)
    lead_next = np.add(x, f*dt)
    lead_traj[t][:] = lead_next
    x = lead_next.copy()
  
  return lead_traj


# def generate_followers(nTraj, init_state, leader_traj, n,a,b,T,v0,l,s0,s1,delt, traj_len, dt):

#   # full trajectory of follow car pos, vel, and accel
  
#   full_traj_state = np.zeros((nTraj,traj_len,3))

#   for i in range(nTraj):
#     current_state = init_state[0:2]
#     full_traj_state[i,0,:] = init_state
    
#     for j in range(1, traj_len):
#       current_state, accel = get_next_state(current_state, leader_traj[j,:], n,a,b,T,v0,l,s0,s1,delt,dt)
#       full_traj_state[i,j,:] = [current_state[0], current_state[1], accel]

#   return full_traj_state
  

def get_follow_traj(follow_init, leader_traj, n,a,b,T,v0,l,s0,s1,delt,traj_len, dt):
  follow_traj = np.zeros((traj_len, 3))
  follow_traj[0][:] = follow_init
  follow_curr = follow_init.copy()

  for t in range(1, traj_len):
    follow_next = get_next_state(follow_curr, leader_traj[t,:], n,a,b,T,v0,l,s0,s1,delt, dt)
    follow_traj[t][:] = follow_next
    follow_curr = follow_next.copy()

  return follow_traj


def idm(lead_curr, x,n,a,b,T,v0,l,s0,s1,delt):

  f = np.zeros(n)
  s = np.zeros(n)

  # calculate behavior of car 2 to car n
  alpha0 = 0.0
  alpha1 = 0.0
  alpha2 = 0.0
  alpha3 = 0.0
  alpha4 = 0.0
  alpha5 = 0.0
  alpha6 = 0.0
  alpha7 = 0.0
  alpha8 = 0.0

  va = lead_curr[1]         # velocity of car 1
  s = lead_curr[0]-l-x[0]   # pos of car 1
  ss = s**2

  alpha0 = -a/(v0**delt)
  alpha1 = -1/(4*b*(ss))
  alpha2 = (va/(2*b*ss)) - (sqrt(a)*T)/(sqrt(b)*ss)
  alpha3 = (sqrt(a)*s1)/(sqrt(b*v0)*ss)
  alpha4 = ((sqrt(a)*s0)/(sqrt(b)*ss)) - ((a*T**2)/ss) - ((va*va)/(4*b*ss))-((sqrt(a)*va*T)/(sqrt(b)*ss))
  alpha5 = ((-2*sqrt(a)*s0*s1*va)/(sqrt(b*v0)*ss)) - ((2*a*s1*T)/(sqrt(v0)*ss))
  alpha6 = ((-2*a*s1**2)/(v0*ss)) - ((sqrt(a)*s0*va)/(sqrt(b)*ss)) - ((2*a*s0*T)/ss)
  alpha7 = ((-2*a*s0*s1)/(sqrt(v0)*ss))
  alpha8 = ((-a*s0**2)/ss) + a

  f[0] = x[1]   # set pos car 2 as velocity of car 2
  f[1] = alpha0*x[1]**delt + alpha1*x[1]**4 + alpha2*x[1]**3 + alpha3*x[1]**(5/2) + alpha4*x[1]**2 + alpha5*x[1]**(3/2) + alpha6*x[1] + alpha7*x[1]**(1/2) + alpha8

  return f

def get_next_state(current_state, leader_current, n,a,b,T,v0,l,s0,s1,delt,dt):
  next_state = np.zeros(3)  

  # and noisy pos and velocity of follow car and accel of follow car
  R = 0.1
  noise = np.random.multivariate_normal([0,0],R**2*dt*np.eye(2))

  yout = idm(leader_current, current_state,n,a,b,T,v0,l,s0,s1,delt) # follower [vel, acc]
  next_state[0:2] = current_state[0:2] + yout * dt + noise
  next_state[2] = yout[1]

  if next_state[1] < 0:
    next_state[1] = current_state[1]

  return next_state


def get_initial_state(nTraj, s0,l):
  y0 = np.zeros((nTraj, 3))
  R = 0.1

  for n in range(nTraj):
    noise = np.random.multivariate_normal([0,0],R**2*np.eye(2))

    y0[n][0] = (s0 + l + 2) + noise[0]
    y0[n][1] = 120/36 +  noise[1]

  return y0

def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
  low = 0.9*mean
  upp = 1.1*mean
  return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

def get_driver_idm_params(a_in, b_in, T_in, v0_in, l_in, s0_in, s1_in):

  delt = 4
  sig = 0.01
  a = get_truncated_normal(a_in, sig).rvs()
  b = get_truncated_normal(b_in, sig).rvs()
  T = get_truncated_normal(T_in, sig).rvs()
  v0 = get_truncated_normal(v0_in, sig).rvs()
  l = get_truncated_normal(l_in, sig).rvs()
  s0 = get_truncated_normal(s0_in, sig).rvs()
  s1 = get_truncated_normal(s1_in, sig).rvs()

  return a, b, T, v0, l, s0, s1, delt



