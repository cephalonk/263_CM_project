"""
Initial parameter values for geyser_simulator.py (SI units throughout).
 
Keys match the `pars` dict expected by the simulator, and are exactly the
parameters listed in the model PDF. Values for the Rotorua geyser reservoir
come from the check-in 1 parameter table (the PDF gives only T_c = 30 C for
this reservoir). Where the table gives a range, the initial value is the
midpoint (geometric mean for k, which spans an order of magnitude). Ranges
are in the comments for later use as optimiser bounds.
 
The recharge term in the model equations is the COLD recharge path, so rho,
mu, k and L below are the check-in 1 "cold" values (rho_cold, mu_cold,
k_cold, L_cold), not the hot-recharge ones.
 
NOTE: A = 5.0e7 m^2 in the model PDF belongs to the BENCHMARK data (a
separate reservoir, the London chalk aquifer), not to the geyser reservoir.
It is only for the validation run, see pars_benchmark_area below.
"""
 
# Intermediate constants, only used to build the dict below
k_therm = 2.5       # W m^-1 K^-1, thermal conductivity (table)
L_cond = 200.0      # m, conduction distance, table range 100-300
A_area = 2.15e7     # m^2, reservoir area, table range 1.5-2.8e7 (midpoint)
 
pars_initial = {
    # --- given in the model PDF ---
    "Tc":    30.0,        # C, shallow cold-water temperature
 
    # --- from the check-in 1 table ---
    "g":     9.81,        # m s^-2
    "A_res": A_area,      # m^2, reservoir area, range 1.5-2.8e7
    "A":     A_area,      # m^2, recharge area, taken equal to A_res
    "rho":   995.65,      # kg m^-3, water at 30 C (rho_cold)
    "mu":    7.97e-4,     # Pa s, water at 30 C (mu_cold)
    "c":     3750.0,      # J kg^-1 K^-1, specific heat of brine (FLAG 2)
    "phi":   0.2,         # porosity, range 0.1-0.3
    "k":     5.5e-14,     # m^2, permeability, range 15-200e-15 (geometric mean)
    "L":     200.0,       # m, cold recharge distance, range 100-300
    "P0":    1.4e6,       # Pa, natural pressure, table range 1.4-2e6
    "M0":    5.0e11,      # kg, reservoir mass, table range 1e11-1e12
 
    # --- read from the data / derived (FLAGS) ---
    "T_hot":  150.0,                         # C (FLAG 3)
    "Rth":   L_cond / (k_therm * A_area),   # K W^-1, ~3.7e-6 (FLAG 1)
}
 
# FLAG 1 (Rth): the table has no direct value. This is conduction only,
#   L_cond / (k_therm * A). The model's Rth also includes steady hot upflow,
#   which the table does not cover, so treat it as a rough starting point.
#
# FLAG 2 (c): the PDF says specific heat of water; the table value is for
#   brine. It is fixed in the calibration, so it is absorbed into M0 and Rth.
#
# FLAG 3 (Thot): not in the table as a reservoir temperature (the table's
#   200 C is the deep hot recharge temperature). 150 C is read from the
#   temperature data, which starts near 148-150 C.
#
# NOTE (M0): not given in the PDF. With A = 2.15e7 the table's formula
#   A * phi * P0 / g gives ~6e11 kg, inside the table range and close to 5e11.