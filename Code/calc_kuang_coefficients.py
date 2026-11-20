"""Project the w and 500-hPa humidity Jacobians onto Kuang's heating modes.

Based on Calc_Kuang's_coefficient.ipynb. Edit the settings below, then run:
    python Code/calc_kuang_coefficients.py

For w, reproduce the notebook's two density-weighted velocity modes exactly.
For Q, perturb ONLY the 500-hPa input level, holding all other levels fixed.
The humidity calculation therefore has one input and two output modes.

Each coefficient matrix has output modes in its rows and inputs in its columns:
    w: 2 x 2, using the notebook's velocity-mode normalization.
    Q: 2 x 1, per kelvin perturbation of model moisture q at 500 hPa.
The model uses q = (L_v / C_p) * specific_humidity. A unit model-moisture
perturbation therefore requires C_p / L_v kg/kg in the native humidity input.
Also save Q coefficients per kg/kg and per g/kg for comparison with the native
Jacobian. Humidity verification plots show a 1-K model-moisture perturbation.
"""

import csv
from pathlib import Path

import matplotlib
import numpy as np
from scipy.interpolate import interp1d

matplotlib.use("Agg")
from matplotlib import pyplot as plt


# ====================================================
# 1. Settings
# ====================================================

project_dir = Path(__file__).resolve().parents[1]
k_min, k_max = 1, 13
band_name = f"k={k_min}~{k_max}"
predictors = ("w", "Q")
jacobian_root = project_dir / "Files" / "Jacobian"
output_root = project_dir / "Files" / "Kuang_coefficients"
figure_root = project_dir / "Figure" / "Kuang_coefficients"

humidity_pressure_pa = 50000.0  # Perturb only the 500-hPa ERA5 humidity level.
humidity_plot_amplitude_K = 1.0  # Model q; affects the plot, not the coefficients.
make_plots = True

# Keep the atmosphere and height grid identical to the notebook.
z = np.linspace(0.0, 14000.0, 141)  # m
T0 = 300.0                         # K
Gamma_T = 0.0065                    # K/m
gravity = 9.81                     # m/s^2
heat_capacity = 1004.5             # J/(kg K)
latent_heat = 2.5e6                # J/kg; edit to match the model's L_v.
gas_constant = 287.5               # J/(kg K)
p0 = 1.0e5                         # Pa
seconds_per_day = 86400.0


# ====================================================
# 2. Background atmosphere and two heating basis functions
# ====================================================

if not np.isfinite(latent_heat) or latent_heat <= 0:
    raise ValueError("Latent heat must be finite and positive.")
if not np.isfinite(heat_capacity) or heat_capacity <= 0:
    raise ValueError("Heat capacity must be finite and positive.")
model_moisture_per_kgkg = latent_heat / heat_capacity  # K per kg/kg.
specific_humidity_per_K = heat_capacity / latent_heat  # kg/kg per K.

Gamma_d = gravity / heat_capacity
T = T0 - Gamma_T * z
p = p0 * (1.0 - Gamma_T * z / T0) ** (gravity / gas_constant / Gamma_T)
rho = p / gas_constant / T

G1 = np.pi / 2 * np.sin(np.pi * z / z.max())
G2 = np.pi / 2 * np.sin(2 * np.pi * z / z.max())
G_stack = np.stack([G1, G2], axis=1)
G_J_stack = G_stack * (Gamma_d - Gamma_T)


# ====================================================
# 3. Apply the Jacobians and project each radiative response
# ====================================================

for predictor in predictors:
    jacobian_dir = jacobian_root / predictor / band_name
    output_dir = output_root / predictor / band_name
    figure_dir = figure_root / predictor / band_name
    plev = np.loadtxt(jacobian_dir / "plev.txt", ndmin=1)
    if plev.size < 2 or not np.isfinite(plev).all() or not np.all(np.diff(plev) > 0):
        raise ValueError(f"{predictor}: pressure levels must be finite and strictly increasing.")
    if p.min() < plev.min() or p.max() > plev.max():
        raise ValueError(f"{predictor}: the pressure grid must cover the full Kuang height domain.")

    if predictor == "w":
        # G is a density-weighted velocity mode per day in the notebook.
        # Convert it to geometric velocity before applying the w Jacobian.
        w_modes = G_stack / (seconds_per_day * rho[:, None])
        input_ERA5 = interp1d(
            p, w_modes, axis=0, bounds_error=False, fill_value=0.0
        )(plev)
        input_labels = ("mode_1", "mode_2")
        normalization = "w(z) = G(z) / (86400 * rho(z)), in m/s"
        plot_scale = 1.0
        plot_title = "Vertical-velocity modes"
    elif predictor == "Q":
        # Native Q Jacobians expect kg/kg, while model q = (L_v/C_p)*Q is in K.
        # Use the chain rule: d(heating)/d(q) = (C_p/L_v) * d(heating)/d(Q).
        matches = np.flatnonzero(np.isclose(plev, humidity_pressure_pa, rtol=0, atol=1e-6))
        if matches.size != 1:
            raise ValueError(f"Q: require exactly one {humidity_pressure_pa / 100:g}-hPa input level.")
        humidity_index = int(matches[0])
        input_ERA5 = np.zeros((plev.size, 1))
        input_ERA5[humidity_index, 0] = specific_humidity_per_K
        input_labels = (f"q_{humidity_pressure_pa / 100:g}hPa_K",)
        normalization = (f"Unit derivative per K of model q=(L_v/C_p)*specific_humidity "
                         f"at {plev[humidity_index] / 100:g} hPa; "
                         f"1 K = {specific_humidity_per_K:g} kg/kg; all other input levels held fixed")
        plot_scale = humidity_plot_amplitude_K
        plot_title = (f"{plev[humidity_index] / 100:g}-hPa model moisture: {plot_scale:g} K\n"
                      f"({1000 * plot_scale * specific_humidity_per_K:g} g/kg specific humidity)")
    else:
        raise ValueError(f"Unknown predictor: {predictor}")

    output_dir.mkdir(parents=True, exist_ok=True)
    responses = {}
    reconstructions = {}
    coefficients = {}
    relative_errors = {}

    for radiation in ("LW", "SW"):
        jacobian = np.loadtxt(jacobian_dir / f"Jacob_{radiation}.txt", ndmin=2)
        if jacobian.shape != (plev.size, plev.size) or not np.isfinite(jacobian).all():
            raise ValueError(f"{predictor}/{radiation}: Jacobian must be finite and match plev.")

        # The heating Jacobians output K/day on ERA5 pressure levels.
        heating_ERA5 = jacobian @ input_ERA5
        heating_modes = interp1d(plev, heating_ERA5, axis=0, bounds_error=True)(p)

        # Same least-squares projection as the notebook:
        # rho * heating = (Gamma_d - Gamma_T) * [G1, G2] @ coefficient.
        coefficient = np.linalg.lstsq(G_J_stack, rho[:, None] * heating_modes, rcond=None)[0]
        reconstructed = G_J_stack @ coefficient / rho[:, None]
        residual_norm = np.linalg.norm(rho[:, None] * (reconstructed - heating_modes), axis=0)
        response_norm = np.linalg.norm(rho[:, None] * heating_modes, axis=0)
        relative_error = np.divide(residual_norm, response_norm,
                                   out=np.full(response_norm.shape, np.nan), where=response_norm > 0)

        coefficients[radiation] = coefficient
        responses[radiation] = heating_modes
        reconstructions[radiation] = reconstructed
        relative_errors[radiation] = relative_error
        header = (f"{radiation}: rows = output modes 1,2; columns = {','.join(input_labels)}\n"
                  f"{normalization}")
        np.savetxt(output_dir / f"{radiation}_coefficients.txt", coefficient, header=header)
        if predictor == "Q":
            native_coefficient = coefficient * model_moisture_per_kgkg
            np.savetxt(output_dir / f"{radiation}_coefficients_per_kgkg.txt", native_coefficient,
                       header="500-hPa specific-humidity coefficients per kg/kg; rows = output modes 1,2")
            np.savetxt(output_dir / f"{radiation}_coefficients_per_gkg.txt", native_coefficient / 1000.0,
                       header="500-hPa humidity coefficients per g/kg; rows = output modes 1,2")
        print(f"{predictor} / {radiation} coefficients (output mode x input):\n{coefficient}")
        print(f"Density-weighted relative projection error: {relative_error}")

    with (output_dir / "projection_errors.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("radiation", "input", "density_weighted_relative_error"))
        for radiation in ("LW", "SW"):
            writer.writerows((radiation, label, error)
                              for label, error in zip(input_labels, relative_errors[radiation]))

    # Retain the exact inputs and responses so normalization can be inspected.
    np.savez(output_dir / "mode_profiles.npz", z=z, p=p, rho=rho, G_stack=G_stack,
             G_J_stack=G_J_stack, plev=plev, input_ERA5=input_ERA5,
             heat_capacity=heat_capacity, latent_heat=latent_heat,
             specific_humidity_per_K=specific_humidity_per_K,
             LW_response=responses["LW"], SW_response=responses["SW"],
             LW_reconstructed=reconstructions["LW"], SW_reconstructed=reconstructions["SW"])
    with (output_dir / "normalization.txt").open("w") as stream:
        stream.write(f"Predictor: {predictor}\n{normalization}\n")
        stream.write("Coefficient rows: output modes 1,2; columns: input modes or q_500hPa.\n")
        stream.write("Heating reconstruction: G_J_stack @ coefficient / rho.\n")
        stream.write(f"T0={T0} K; Gamma_T={Gamma_T} K/m; gravity={gravity} m/s^2; ")
        stream.write(f"cp={heat_capacity}; R={gas_constant}; p0={p0} Pa.\n")
        if predictor == "Q":
            stream.write(f"L_v={latent_heat} J/kg; C_p={heat_capacity} J/(kg K).\n")
            stream.write(f"q_model=(L_v/C_p)*specific_humidity; L_v/C_p={model_moisture_per_kgkg:g}.\n")
            stream.write("*_coefficients.txt are per K of model q at 500 hPa.\n")
            stream.write("*_per_kgkg.txt and *_per_gkg.txt use native specific-humidity units.\n")
            stream.write(f"Verification plot amplitude: {humidity_plot_amplitude_K} K of model q.\n")
        stream.write("A relative projection error is NaN when the original response is zero.\n")

    if make_plots:
        figure_dir.mkdir(parents=True, exist_ok=True)
        n_input = input_ERA5.shape[1]
        fig, axes = plt.subplots(2, n_input, figsize=(5 * n_input, 9), sharey=True, squeeze=False)
        for row, radiation in enumerate(("LW", "SW")):
            for column, label in enumerate(input_labels):
                ax = axes[row, column]
                ax.plot(responses[radiation][:, column] * plot_scale, z, label="Jacobian response")
                ax.plot(reconstructions[radiation][:, column] * plot_scale, z, "--",
                        label="2-mode reconstruction")
                ax.set_title(f"{radiation}, {label}")
                ax.set_xlabel(r"Heating rate (K day$^{-1}$)")
                ax.grid(True, alpha=0.3)
        axes[0, 0].set_ylabel("Height (m)")
        axes[1, 0].set_ylabel("Height (m)")
        axes[0, 0].legend()
        fig.suptitle(f"{plot_title} ({band_name})")
        fig.tight_layout()
        for extension in ("png", "pdf"):
            fig.savefig(figure_dir / f"verification.{extension}", dpi=300, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved {predictor} coefficients to {output_dir}")
