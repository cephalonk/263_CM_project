# Intermediate constants, only used to build the dict below
k_therm = 2.5       # W m^-1 K^-1, thermal conductivity (table)
L_cond = 200.0      # m, conduction distance, table range 100-300
A_area = 5.0e7      # m^2, from the model PDF (table range 1.5-2.8e7 not used)
 
pars_initial = {
    # --- from the model PDF ---
    "Tc":    30.0,        # C, cold groundwater temperature
    "A_res": A_area,      # m^2, reservoir area
    "A":     A_area,      # m^2, recharge cross-sectional area
    "Thot":  150.0,       # C, natural reservoir temperature (read from the
                          #    PDF's temperature plot, which starts ~149 C)
 
    # --- from the check-in 1 table ---
    "g":     9.81,        # m s^-2
    "rho":   995.65,      # kg m^-3, water at 30 C (rho_cold)
    "mu":    7.97e-4,     # Pa s, water at 30 C (mu_cold)
    "c":     3750.0,      # J kg^-1 K^-1, specific heat of brine
    "phi":   0.2,         # porosity, range 0.1-0.3
    "k":     5.5e-14,     # m^2, permeability, range 15-200e-15 (geometric mean)
    "L":     200.0,       # m, cold recharge distance, range 100-300
    "P0":    1.4e6,       # Pa, natural pressure, table range 1.4-2e6
    "M0":    5.0e11,      # kg, reservoir mass, table range 1e11-1e12
 
    # --- derived (FLAG) ---
    "Rth":   L_cond / (k_therm * A_area),   # K W^-1, ~1.6e-6
}
 
# FLAG (Rth): the PDF gives no value and the table has no direct one. This is
#   conduction only, L_cond / (k_therm * A). The model's Rth also includes
#   steady hot upflow, which the table does not cover, so treat it as a rough
#   starting point.
#
# NOTE (M0): not given in the PDF. With A = 5.0e7 the table's formula
#   A * phi * P0 / g would give ~1.4e12 kg, above the table range, so M0 is a
#   good candidate to let the calibration move.