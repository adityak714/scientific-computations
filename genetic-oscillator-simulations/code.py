from scipy.integrate import solve_ivp
import numpy as np
from matplotlib import pyplot as plt

## Deterministic
alpha_A = 50
alpha_A_prim = 500
alpha_R = 0.01
alpha_R_prim = 50
beta_A = 50
beta_R = 5
delta_MA = 10
delta_MR = 0.5
delta_A = 1
delta_R = 0.05
gamma_A = 1  # mol^-1 * hr^-1
gamma_R = 1
gamma_C = 2
theta_A = 50  # h^-1
theta_R = 100

Initial = [1, 1, 0, 0, 0, 0, 0, 0, 0]
t_steps = 400

def deterministic(t, y):
    D_A, D_R, D_A_prim, D_R_prim, M_A, A, M_R, R, C = y
    yprime = np.zeros(9)
    yprime[0] = theta_A * D_A_prim - gamma_A * D_A * A  # dD_A/dt
    yprime[1] = theta_R * D_R_prim - gamma_R * D_R * A  # dD_R/dt
    yprime[2] = gamma_A * D_A * A - theta_A * D_A_prim  # dD_A_prim/dt
    yprime[3] = gamma_R * D_R * A - theta_R * D_R_prim  # dD_R_prim/dt
    yprime[4] = alpha_A_prim * D_A_prim + alpha_A * D_A - delta_MA * M_A  # dMA/dt
    yprime[5] = (
        beta_A * M_A
        + theta_A * D_A_prim
        + theta_R * D_R_prim
        - (A * (gamma_A * D_A + gamma_R * D_R + gamma_C * R + delta_A))
    )  # dA/dt
    yprime[6] = alpha_R_prim * D_R_prim + alpha_R * D_R - delta_MR * M_R  # dMR/dt
    yprime[7] = beta_R * M_R - gamma_C * A * R + delta_A * C - delta_R * R  # dR/dt
    yprime[8] = gamma_C * A * R - delta_A * C  # dC/dt
    return yprime

teval = np.linspace(0, t_steps, 1000)  # a fine evaluation time samples
sol = solve_ivp(deterministic, [0, t_steps], Initial, t_eval=teval, method="RK45")

plt.figure(figsize=(6, 3))
plt.plot(sol.t, sol.y[5], linestyle="solid", color="blue", label="A")
plt.xlabel("Time (hr)")
plt.ylabel("Number of molecules")
plt.title("Deterministic solution using RK45")
plt.legend(loc="upper right")
plt.show()

plt.figure(figsize=(6, 3))
plt.plot(sol.t, sol.y[7], linestyle="solid", color="red", label="R")
plt.xlabel("Time (hr)")
plt.ylabel("Number of molecules")
plt.title("Deterministic solution using RK45")
plt.legend(loc="upper right")
plt.show()

## Stochastic

## 16 reactions x 9 states
StateChangeMatrix = np.array([
    [-1,-1,0,0,0,0,0,0,1],
    [-1,0,0,0,0,0,0,0,0],
    [0,1,0,0,0,0,0,0,-1],
    [0,-1,0,0,0,0,0,0,0],
    [-1,0,-1,1,0,0,0,0,0],
    [-1,0,0,0,-1,1,0,0,0],
    [1,0,1,-1,0,0,0,0,0],
    [0,0,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,-1,0,0],
    [1,0,0,0,0,0,0,0,0],
    [1,0,0,0,1,-1,0,0,0],
    [0,0,0,0,0,0,0,1,0],
    [0,0,0,0,0,0,0,1,0],
    [0,0,0,0,0,0,0,-1,0],
    [0,1,0,0,0,0,0,0,0]
])

def Propensityfunction(state, reactions):
    w = np.zeros(reactions)
    w[0] = gamma_C*state[0]*state[1]
    w[1] = delta_A*state[0]
    w[2] = delta_A*state[8]
    w[3] = delta_R*state[1]
    w[4] = gamma_A*state[2]*state[0]
    w[5] = gamma_R*state[4]*state[0] 
    # [A,R,D_A,D_Aprime,D_R,D_Rprime,M_A,M_R,C]
    w[6] = theta_A*state[3]
    w[7] = alpha_A*state[2]
    w[8] = alpha_A_prim*state[3]
    w[9] = delta_MA*state[6]
    w[10] = beta_A*state[6]
    w[11] = theta_R*state[5]
    w[12] = alpha_R*state[4]
    w[13] = alpha_R_prim*state[5]
    w[14] = delta_MR*state[7]
    w[15] = beta_R*state[7]
    return w # return propensity


## Exponential (for sampling the timestep tau)
def Exponential(lam):
    u = np.random.rand(1)
    tau = (-1/lam)*np.log(1-u)
    return tau # time sample from exponential function

## Reaction Choice (sampling which reaction occurs)
def Discrete(x,p):
    cdf = np.cumsum(p)
    u = np.random.rand(1)
    idx = np.searchsorted(cdf,u)
    return x[idx]

def StochasticSimulation(initial, StateChangeMatrix, FinalTime):
    reactions, states = StateChangeMatrix.shape
    ReactNum = np.array(range(reactions))
    allStates= []
    allTimes = []
    allStates.append(initial)
    state = initial
    allTimes.append(0)

    k = 0
    t = 0
    while True:
        propensity = Propensityfunction(state.flatten(), reactions)
        a = np.sum(propensity)
        tau = Exponential(a) ## time
        t = t + tau
        if t > FinalTime:
            break
        k = k + 1
        which = Discrete(ReactNum, propensity/a) ## which state
        state = state + StateChangeMatrix[which.item(),:] ## update
        allStates.append(state)
        allTimes.append(t)
    return allStates, allTimes

## simulate for 400 timesteps (hours)
FinalTime = 400
allStates, allTimes = StochasticSimulation(Initial, StateChangeMatrix,
FinalTime)

def plot_simulation(allStates, allTimes):
    times = [allTimes[i] if i == 0 else allTimes[i][0] for i in range(len(allTimes))] ## timestamp
    state_a = [allStates[i][0] for i in range(len(allStates))]
    state_r = [allStates[i][1] for i in range(len(allStates))]
    
    fig, axes = plt.subplots(nrows=2,ncols=1,figsize=(15,10))

    axes[0].plot(times,state_a,linestyle='-', color='blue', label='Protein A')
    axes[0].set_xlabel('Time(hours)')
    axes[0].set_ylabel('Count of Proteins')
    axes[0].set_title('Simulation of Protein A')
    axes[1].plot(times, state_r, linestyle='-', color='red', label='Protein R')
    axes[1].set_xlabel('Time(hours)')
    axes[1].set_ylabel('Count of Proteins')
    axes[1].set_title('Simulation of Protein R')

plot_simulation(allStates, allTimes)

## Stochastic simulation - if delta_r were to be set to 0.05
## What would happen? 
# - Stochastic simulation: number of R molecules flatlined after roughly 50 timesteps
# - Deterministic: oscillatory behaviour sustained

# delta_r = 0.05
# allStates, allTimes = StochasticSimulation(Initial, StateChangeMatrix, FinalTime)
# plot_simulation(allStates, allTimes)