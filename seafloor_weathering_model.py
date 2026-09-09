"""
seafloor_weathering_model.py

A simplified physical model linking oceanic crustal age to basalt
weathering intensity, built around three well-established building
blocks from marine geophysics and geochemistry:

    1. Half-space cooling  -> how oceanic crust temperature evolves with age
    2. Spreading rate       -> converts age into distance from the ridge
    3. Arrhenius kinetics   -> converts temperature into a weathering rate

The purpose of this script is NOT to reproduce a published, calibrated
model. It is a compact, fully transparent toy model showing how the
governing physics (heat conduction) and governing chemistry (reaction
kinetics) can be coupled to produce a testable prediction: seafloor
weathering intensity should fall off with distance from the ridge axis,
and that fall-off should depend on spreading rate.

Author: Ayobami Isola
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

# ---------------------------------------------------------------------
# 1. Physical constants and model parameters
# ---------------------------------------------------------------------

SECONDS_PER_MYR = 3.15576e13      # seconds in one million years
KM_PER_CM_YR_MYR = 10.0           # 1 cm/yr * 1 Myr = 10 km (unit convenience)

# --- Half-space cooling parameters (Turcotte & Schubert, standard values)
KAPPA = 8.0e-7        # thermal diffusivity of oceanic lithosphere, m^2/s
T_MANTLE = 1350.0      # ridge-axis / mantle temperature, deg C
T_SEAWATER = 2.0        # bottom seawater temperature, deg C
Z_REACTION = 500.0     # representative depth of the hydrothermal
                        # reaction zone within the upper crust, m
T_REACTION_CAP = 400.0  # deg C -- above this, seawater-derived fluid is
                        # supercritical/boiling and does not drive normal
                        # aqueous basalt weathering, so we cap the
                        # "reaction temperature" here rather than letting
                        # it follow the crust all the way to magmatic
                        # temperatures near the ridge axis

# --- Arrhenius kinetics parameters (order-of-magnitude, illustrative;
#     basaltic glass dissolution activation energies reported in the
#     weathering literature are commonly in the ~50-60 kJ/mol range)
EA = 55_000.0          # activation energy, J/mol
R_GAS = 8.314          # universal gas constant, J/mol/K
A_PREFACTOR = 1.0       # pre-exponential factor; the model is normalised
                        # so absolute units drop out (see normalise_rate)

# --- Spreading rates used for the fast vs slow ridge comparison
SLOW_HALF_RATE_CM_YR = 1.5   # e.g. Mid-Atlantic-Ridge-like, cm/yr
FAST_HALF_RATE_CM_YR = 6.0   # e.g. East-Pacific-Rise-like, cm/yr


# ---------------------------------------------------------------------
# 2. Step 1 - Half-space cooling model
# ---------------------------------------------------------------------

def crustal_temperature(age_myr, z=Z_REACTION,
                         kappa=KAPPA, t_mantle=T_MANTLE, t_sw=T_SEAWATER):
    """
    Temperature (deg C) at depth z within crust of a given age.

    This is the classical half-space cooling solution: newly formed
    crust starts at mantle temperature and cools as heat conducts
    upward into the ocean. At a fixed depth z, temperature decreases
    as the crust ages, because the cooling front migrates downward
    over time.

    age_myr : crustal age in millions of years (must be > 0)
    z       : depth below seafloor, metres
    """
    age_s = np.asarray(age_myr, dtype=float) * SECONDS_PER_MYR
    # Guard against age = 0 (division by zero)
    age_s = np.maximum(age_s, 1e-6)
    eta = z / (2.0 * np.sqrt(kappa * age_s))
    return t_sw + (t_mantle - t_sw) * erf(eta)


def heat_flux(age_myr, kappa=KAPPA, t_mantle=T_MANTLE, t_sw=T_SEAWATER,
              thermal_conductivity=3.0):
    """
    Conductive surface heat flux (W/m^2) as a function of crustal age.
    Included as a secondary, bonus diagnostic: heat flux is the classic
    observable geophysicists actually measure to test cooling models.
    """
    age_s = np.asarray(age_myr, dtype=float) * SECONDS_PER_MYR
    age_s = np.maximum(age_s, 1e-6)
    return thermal_conductivity * (t_mantle - t_sw) / np.sqrt(np.pi * kappa * age_s)


def seafloor_depth(age_myr):
    """
    Bathymetry (depth below sea surface, m) as a function of crustal age,
    using the empirical Parsons & Sclater (1977) relation for young
    seafloor. This is a bonus diagnostic connecting the model to
    something directly observable/mappable, and is not used elsewhere
    in the weathering calculation.
    """
    age_myr = np.asarray(age_myr, dtype=float)
    return 2500.0 + 350.0 * np.sqrt(np.maximum(age_myr, 0.0))


# ---------------------------------------------------------------------
# 3. Step 2 - Convert between crustal age and distance from the ridge
# ---------------------------------------------------------------------

def age_from_distance(distance_km, half_spreading_rate_cm_yr):
    """Crustal age (Myr) at a given distance (km) from the ridge axis."""
    rate_km_per_myr = half_spreading_rate_cm_yr * KM_PER_CM_YR_MYR
    return distance_km / rate_km_per_myr


def distance_from_age(age_myr, half_spreading_rate_cm_yr):
    """Distance (km) from the ridge axis for a given crustal age (Myr)."""
    rate_km_per_myr = half_spreading_rate_cm_yr * KM_PER_CM_YR_MYR
    return age_myr * rate_km_per_myr


# ---------------------------------------------------------------------
# 4. Step 3 - Temperature-dependent weathering rate (Arrhenius kinetics)
# ---------------------------------------------------------------------

def arrhenius_rate(temperature_c, ea=EA, a_prefactor=A_PREFACTOR):
    """
    Relative basalt dissolution / weathering rate, following an
    Arrhenius temperature dependence: rate = A * exp(-Ea / R*T).

    This is the standard way weathering kinetics are related to
    temperature in the geochemistry literature (warmer reactions
    proceed faster, following an exponential rather than linear law).
    """
    temperature_k = np.asarray(temperature_c, dtype=float) + 273.15
    return a_prefactor * np.exp(-ea / (R_GAS * temperature_k))


def reaction_temperature(age_myr, z=Z_REACTION, cap=T_REACTION_CAP):
    """
    Effective temperature (deg C) driving aqueous weathering reactions.

    Equal to the raw half-space cooling temperature, but capped at
    T_REACTION_CAP: above that temperature, circulating fluid is
    supercritical or boiling rather than liquid water, so it stops
    behaving like a normal aqueous weathering agent. This keeps the
    model in the physically sensible regime (liquid-water basalt
    alteration) instead of extrapolating Arrhenius kinetics into
    magmatic temperatures where the reaction is a different process.
    """
    raw_t = crustal_temperature(age_myr, z=z)
    return np.minimum(raw_t, cap)


def normalised_weathering_rate(age_myr, z=Z_REACTION, ea=EA, cap=T_REACTION_CAP):
    """
    Weathering rate as a function of crustal age, normalised so that
    the rate at the ridge axis (age -> 0) equals 1.0. Normalising
    removes the arbitrary pre-exponential factor and lets us focus on
    the SHAPE of the decline with age, which is the testable, physically
    meaningful prediction of this toy model.
    """
    t_at_age = reaction_temperature(age_myr, z=z, cap=cap)
    t_at_ridge = cap  # by construction, the ridge-axis reaction temperature is the cap
    rate = arrhenius_rate(t_at_age, ea=ea)
    rate_ridge = arrhenius_rate(np.array([t_at_ridge]), ea=ea)[0]
    return rate / rate_ridge


# ---------------------------------------------------------------------
# 5. Build the figures
# ---------------------------------------------------------------------

def make_plots(output_dir="."):
    # Finer resolution near the axis (where the rate changes fast) and
    # coarser resolution further out (where it barely changes at all).
    age_near = np.geomspace(0.001, 2.0, 300)
    age_far = np.linspace(2.0, 70.0, 300)[1:]
    age_myr = np.concatenate([age_near, age_far])

    # --- Figure 1: temperature vs age -----------------------------------
    temp_raw = crustal_temperature(age_myr)
    temp_capped = reaction_temperature(age_myr)
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(age_myr, temp_raw, color="lightcoral", linewidth=1.5,
             linestyle="--", label="Raw half-space cooling temperature")
    ax1.plot(age_myr, temp_capped, color="firebrick", linewidth=2.5,
             label=f"Reaction temperature (capped at {T_REACTION_CAP:.0f}°C)")
    ax1.axhline(T_REACTION_CAP, color="gray", linewidth=0.8, linestyle=":")
    ax1.set_ylim(-20, 450)
    ax1.set_xlabel("Crustal age (Myr)")
    ax1.set_ylabel(f"Temperature at {Z_REACTION:.0f} m depth (°C)")
    ax1.set_title("Step 1: Half-space cooling — crust cools with age")
    ax1.legend()
    ax1.grid(alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(f"{output_dir}/fig1_temperature_vs_age.png", dpi=150)
    plt.close(fig1)

    # --- Figure 2: weathering rate vs age --------------------------------
    rate = normalised_weathering_rate(age_myr)
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.semilogy(age_myr, rate, color="seagreen", linewidth=2)
    ax2.axvline(1.0, color="gray", linewidth=0.8, linestyle=":")
    ax2.text(1.1, 0.5, "~1 Myr: axial\nregime ends", fontsize=8, color="gray")
    ax2.set_xlabel("Crustal age (Myr)")
    ax2.set_ylabel("Relative weathering rate (ridge axis = 1.0, log scale)")
    ax2.set_title("Step 3: Weathering rate collapses within ~1 Myr of the axis,\nthen persists at a low level")
    ax2.grid(alpha=0.3, which="both")
    fig2.tight_layout()
    fig2.savefig(f"{output_dir}/fig2_weathering_rate_vs_age.png", dpi=150)
    plt.close(fig2)

    # --- Figure 3: weathering rate vs distance, slow vs fast ridge -------
    dist_slow = distance_from_age(age_myr, SLOW_HALF_RATE_CM_YR)
    dist_fast = distance_from_age(age_myr, FAST_HALF_RATE_CM_YR)
    fig3, ax3 = plt.subplots(figsize=(7, 5))
    ax3.semilogy(dist_slow, rate, label=f"Slow ridge ({SLOW_HALF_RATE_CM_YR} cm/yr half-rate)",
              color="steelblue", linewidth=2)
    ax3.semilogy(dist_fast, rate, label=f"Fast ridge ({FAST_HALF_RATE_CM_YR} cm/yr half-rate)",
              color="darkorange", linewidth=2)
    ax3.set_xlim(0, 100)
    ax3.set_xlabel("Distance from ridge axis (km)")
    ax3.set_ylabel("Relative weathering rate (log scale)")
    ax3.set_title("Step 2+3: A fast-spreading ridge stretches the same\nweathering history over a wider band of seafloor")
    ax3.legend()
    ax3.grid(alpha=0.3, which="both")
    fig3.tight_layout()
    fig3.savefig(f"{output_dir}/fig3_weathering_rate_vs_distance.png", dpi=150)
    plt.close(fig3)

    # --- Figure 4 (bonus): seafloor depth vs age --------------------------
    depth = seafloor_depth(age_myr)
    fig4, ax4 = plt.subplots(figsize=(7, 5))
    ax4.plot(age_myr, depth, color="navy", linewidth=2)
    ax4.invert_yaxis()
    ax4.set_xlabel("Crustal age (Myr)")
    ax4.set_ylabel("Seafloor depth below sea surface (m)")
    ax4.set_title("Bonus: Parsons & Sclater (1977) age–depth relation")
    ax4.grid(alpha=0.3)
    fig4.tight_layout()
    fig4.savefig(f"{output_dir}/fig4_bonus_depth_vs_age.png", dpi=150)
    plt.close(fig4)

    return {
        "age_myr": age_myr,
        "temperature_c": temp_capped,
        "relative_weathering_rate": rate,
        "distance_slow_km": dist_slow,
        "distance_fast_km": dist_fast,
        "depth_m": depth,
    }


def print_summary(results):
    age = results["age_myr"]
    rate = results["relative_weathering_rate"]

    def age_at_fraction(frac):
        # first age at which the (monotonically declining) rate drops
        # below `frac` of its ridge-axis value
        below = np.where(rate < frac)[0]
        return age[below[0]] if len(below) else np.nan

    print("=" * 60)
    print("SEAFLOOR WEATHERING MODEL — SUMMARY")
    print("=" * 60)
    print(f"Reaction-zone depth assumed:      {Z_REACTION:.0f} m")
    print(f"Reaction temperature cap:         {T_REACTION_CAP:.0f} deg C (liquid-water regime)")
    print(f"Activation energy (Arrhenius):     {EA/1000:.0f} kJ/mol")
    print(f"Raw crust temp at ridge axis:      {crustal_temperature(np.array([1e-6]))[0]:.0f} deg C (capped to {T_REACTION_CAP:.0f})")
    print(f"Reaction temp at 70 Myr crust:     {reaction_temperature(np.array([70.0]))[0]:.0f} deg C")
    print("-" * 60)
    a50 = age_at_fraction(0.5)
    a10 = age_at_fraction(0.1)
    a01 = age_at_fraction(0.01)
    print(f"Age/distance where rate first falls below 50% of axis value:")
    print(f"    {a50:.2f} Myr  ({distance_from_age(a50, SLOW_HALF_RATE_CM_YR):.1f} km slow-ridge / "
          f"{distance_from_age(a50, FAST_HALF_RATE_CM_YR):.1f} km fast-ridge)")
    print(f"Age/distance where rate first falls below 10% of axis value:")
    print(f"    {a10:.2f} Myr  ({distance_from_age(a10, SLOW_HALF_RATE_CM_YR):.1f} km slow-ridge / "
          f"{distance_from_age(a10, FAST_HALF_RATE_CM_YR):.1f} km fast-ridge)")
    print(f"Age/distance where rate first falls below 1% of axis value:")
    print(f"    {a01:.2f} Myr  ({distance_from_age(a01, SLOW_HALF_RATE_CM_YR):.1f} km slow-ridge / "
          f"{distance_from_age(a01, FAST_HALF_RATE_CM_YR):.1f} km fast-ridge)")
    print("-" * 60)
    print("Interpretation: the model predicts intense, high-temperature")
    print("weathering confined to a narrow band within ~1 Myr of the ridge")
    print("axis, followed by a much weaker but long-lived 'tail' out to old")
    print("crust. This qualitatively matches the real observation that high-")
    print("temperature hydrothermal vent fields cluster near spreading")
    print("centres, while cooler, lower-temperature ridge-flank alteration")
    print("(not captured by this single-regime model) is understood to")
    print("continue for tens of millions of years afterward — a natural next")
    print("extension of this model (see README, 'Next steps').")
    print("=" * 60)


if __name__ == "__main__":
    results = make_plots(output_dir=".")
    print_summary(results)
