# %%
# Cell 1: Imports and Data Loading
# Import libraries
import numpy as np
import cvxpy as cp


# %%
# Cell 2: MODEL
# ==========================================
# 1. DEFINE SYSTEM PARAMETERS
# ==========================================
T = 24  # 24-hour scheduling horizon

# Battery Limits (using rough Port/1MW scale estimates)
E_MAX = 5000.0       # Max battery capacity in kWh
E_MIN = 0.0          # Min battery capacity in kWh
P_BAT_MAX = 1000.0   # Max charge/discharge rate in kW
SOC_INIT = 2500.0    # State of Charge at the start of the day

# Cost Weights (Level 1 - Day Ahead)
c_q_pos, c_l_pos = 0.1, 0.5  # Penalties for importing power (positive prosumption)
c_q_neg, c_l_neg = 0.1, 0.5  # Penalties for exporting power (negative prosumption)

# Cost Weights (Level 2 - Real Time Imbalance)
c_q_imb, c_l_imb = 0.05, 0.3 # Penalties for deviating from schedule

# Dummy Data: 24 hours of Net Prosumption (Load - Solar)
np.random.seed(42)
# The day-ahead forecast we will use to plan
forecast_prosumption = np.random.uniform(-500, 1500, T) 
# The actual reality we face when the day arrives
actual_prosumption = forecast_prosumption + np.random.normal(0, 200, T) 

# ==========================================
# 2. LEVEL 1: DAY-AHEAD SCHEDULING
# ==========================================
print("--- Running Level 1: Day-Ahead Optimization ---")

# Define Variables
P_s = cp.Variable(T)             # Battery power (positive=charge, negative=discharge)
E_s = cp.Variable(T + 1)         # Battery state of energy
P_g = cp.Variable(T)             # Total grid power (Prosumption + Battery)

# To penalize positive and negative grid power differently, we split P_g
P_g_pos = cp.Variable(T, nonneg=True)
P_g_neg = cp.Variable(T, nonneg=True)

# Define Constraints
constraints_L1 = [
    E_s[0] == SOC_INIT,          # Set initial state of charge
    P_g == P_g_pos - P_g_neg,    # P_g is the difference of its pos/neg parts
    P_s <= P_BAT_MAX,            # Battery charge limit
    P_s >= -P_BAT_MAX,           # Battery discharge limit
    E_s <= E_MAX,                # Max capacity limit
    E_s >= E_MIN                 # Min capacity limit
]

# Add dynamic equations for all 24 hours
for k in range(T):
    # Energy conservation: Next State = Current State + Power
    # (Assuming Delta t = 1 hour, and ignoring efficiency loss for MVP clarity)
    constraints_L1.append(E_s[k+1] == E_s[k] + P_s[k])

    # Power balance: Grid Power = Battery Power + Forecasted Prosumption
    constraints_L1.append(P_g[k] == P_s[k] + forecast_prosumption[k])

# Define Objective: Minimize quadratic and linear costs of grid power
cost_L1 = cp.sum(
    c_q_pos * cp.square(P_g_pos) + c_l_pos * P_g_pos + 
    c_q_neg * cp.square(P_g_neg) + c_l_neg * P_g_neg
)

# Solve Level 1
prob_L1 = cp.Problem(cp.Minimize(cost_L1), constraints_L1)
prob_L1.solve(solver=cp.GUROBI)

# Extract the locked-in schedule for the day
scheduled_P_g = P_g.value
print("Level 1 Complete. Schedule locked.\n")

# ==========================================
# 3. LEVEL 2: REAL-TIME OPERATION
# ==========================================
print("--- Running Level 2: Real-Time Iteration ---")

# Variables to track reality as we step through the hours
actual_battery_power = []
actual_imbalance = []
current_soc = SOC_INIT

for k in range(T):
    # Define variables for this SINGLE hour
    P_s_rt = cp.Variable(1)      # Real-time battery adjustment
    E_next_rt = cp.Variable(1)   # End-of-hour battery state
    dP_g = cp.Variable(1)        # The imbalance (deviation from schedule)
    P_g_rt = cp.Variable(1)      # The actual realized grid power
    
    # Define Constraints for this single hour
    constraints_L2 = [
        E_next_rt == current_soc + P_s_rt,
        P_g_rt == P_s_rt + actual_prosumption[k],
        P_g_rt == scheduled_P_g[k] + dP_g,   # Reality = Schedule + Deviation
        
        # Physical constraints
        P_s_rt <= P_BAT_MAX,
        P_s_rt >= -P_BAT_MAX,
        E_next_rt <= E_MAX,
        E_next_rt >= E_MIN
    ]
    
    # Define Objective: Minimize the Imbalance Cost
    # Note: cvxpy handles absolute value inherently as a convex function
    cost_L2 = c_q_imb * cp.square(dP_g) + c_l_imb * cp.abs(dP_g)
    
    # Solve Level 2 for this specific hour
    prob_L2 = cp.Problem(cp.Minimize(cost_L2), constraints_L2)
    prob_L2.solve(solver=cp.GUROBI)
    
    # Record the actions and update the physical battery state for the next hour
    actual_battery_power.append(P_s_rt.value[0])
    actual_imbalance.append(dP_g.value[0])
    current_soc = E_next_rt.value[0]

print("Level 2 Complete. End of simulated day.\n")

# ==========================================
# 4. REVIEW RESULTS
# ==========================================
print(f"Total Imbalance Volume for the day: {np.sum(np.abs(actual_imbalance)):.2f} kWh")
print(f"Final Battery SOC at end of day:    {current_soc:.2f} kWh")


# %%
