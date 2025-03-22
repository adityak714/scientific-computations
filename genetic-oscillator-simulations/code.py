from scipy.integrate import solve_ivp
import numpy as np
from matplotlib import pyplot as plt

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