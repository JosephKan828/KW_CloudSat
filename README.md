# KW_CloudSat: Kelvin-wave radiative coupling

This project identifies Kelvin-wave events from outgoing longwave radiation
(OLR), fits linear mappings from ERA5 vertical velocity or specific humidity to
CloudSat radiative heating, and projects those mappings onto two vertical modes.
The default analysis covers **2006-01-01 through 2017-12-31**, within **5°S–5°N**.

This README describes the files currently under `Code/`. Regression uses
**separate Ridge models for velocity and humidity**, each with longwave and
shortwave responses. Settings are edited at the top of each script; the current
directory has no pipeline runner or `--mode` interface.

## Files and workflow

| File | Purpose |
| --- | --- |
| [Code/KW_selection.py](Code/KW_selection.py) | Filter OLR in frequency–wavenumber space, select strong local minima, and save event indices and OLR plots. |
| [Code/select_valid_column.py](Code/select_valid_column.py) | Collocate event-centered atmospheric columns, select complete profiles, fit heating Jacobians, and compare predictions with saved composites. |
| [Code/calc_kuang_coefficients.py](Code/calc_kuang_coefficients.py) | Apply the Jacobians to two velocity modes or one 500-hPa humidity perturbation, then fit two heating-mode coefficients. |
| [Code/utils.py](Code/utils.py) | Provide optional plotting style, atmospheric profiles, and sine basis functions. The analysis scripts currently define their own settings and do not import this module. |
| [Code/tests/test_kw_selection_netcdf4.py](Code/tests/test_kw_selection_netcdf4.py) | Synthetic-wave checks for filtering, amplitude preservation, events, dates, and invalid OLR inputs. Contains a legacy script reference; see [Tests](#tests). |
| [Code/tests/test_select_valid_column.py](Code/tests/test_select_valid_column.py) | Check coordinate mapping, finite-pair metrics, separate regression masks, saved outputs, and rejection of inconsistent grids/dates. |
| [Code/tests/test_calc_kuang_coefficients.py](Code/tests/test_calc_kuang_coefficients.py) | Check modal projection, humidity scaling, normalization, and pressure validation. One check depends on a removed notebook; see [Tests](#tests). |

Processing order:

1. **OLR → event CSVs:** `KW_selection.py`.
2. **Events + ERA5 + CloudSat → Jacobians:** `select_valid_column.py`.
   Existing composite text files are also required for its diagnostics.
3. **Jacobians → modal coefficients:** `calc_kuang_coefficients.py`.

## Inputs and execution

Dependencies: Python, NumPy, SciPy, Matplotlib, netCDF4, and scikit-learn.
Plots use Matplotlib's noninteractive `Agg` backend. Data paths below are source
defaults and should be edited for another machine.

| Input | Default location | Variables and layout |
| --- | --- | --- |
| OLR anomalies | `/work/DATA/Satellite/OLR/olr_anomaly.nc` | `olr(time, lat, lon)`, normally W m⁻²; daily CF time coordinates. |
| ERA5 geometric vertical velocity | `/data92/b11209013/ERA5_GRIB/Data/tropical_-10_10/w.nc` | `w(time, plev, lat, lon)`, m s⁻¹; `plev` in Pa. |
| ERA5 specific humidity | `/data92/b11209013/ERA5_GRIB/Data/tropical_-10_10/Q.nc` | `Q(time, plev, lat, lon)`, kg kg⁻¹; same dates/grid as `w`. |
| Gridded CloudSat heating | `/data92/b11209013/CloudSat/DATA/QR_gridded_15layer_-5_5.nc` | `QLW`, `QSW`, each `(time, lev, latitude, longitude)`, K day⁻¹; `lev` in Pa. |
| Saved composites | `Files/{w,Q,LW,SW}_composite.txt`, plus `Files/plev.txt` | Composites are `(pressure level, relative longitude)` with native predictor units or K day⁻¹ heating; pressure is Pa. |

The scripts expect geometric `w` already. They do not convert pressure velocity
ω to `w`, regrid ERA5/CloudSat, or create the saved composite text files. Units
are assumed as above rather than converted from arbitrary unit strings.
Composites must already use event-centered longitude ordering and a compatible
event selection; only their shapes and pressure coordinates are checked.

**Connect the event paths before running.** The selector writes to
`Files/KW_events_netcdf4/`, but regression currently reads `Files/KW_events/`.
To use newly selected events, change `event_file` in `select_valid_column.py` to:

```python
event_file = project_dir / "Files" / "KW_events_netcdf4" / f"{band_name}.csv"
```

Keep regression's `event_start`/`event_end` consistent with the selector's
`start_date`/`end_date`. Choose the same `k_min, k_max` in the regression and
coefficient scripts. The selector produces seven bands; the other scripts
process only their configured band, initially `k=1~13`.

After editing settings, run from the repository root:

```bash
python Code/KW_selection.py
python Code/select_valid_column.py
python Code/calc_kuang_coefficients.py
```

## `KW_selection.py`: Kelvin-wave events

### Algorithm

1. Read the requested OLR date/latitude slab, including the entire end date.
   Require finite OLR, uninterrupted daily sampling, paired latitudes across
   the equator, and an ascending, evenly spaced global longitude grid without
   a duplicate cyclic endpoint.
2. Subtract one time–longitude mean at each latitude, symmetrize across the
   equator, and take the arithmetic latitude mean:

   $$
   A=\mathrm{OLR}-\langle\mathrm{OLR}\rangle_{t,\lambda}(\phi),\qquad
   A_s(t,\phi,\lambda)=\frac{A(t,\phi,\lambda)+A(t,-\phi,\lambda)}{2}.
   $$

   `olr_raw = mean(A_s, latitude)` has shape `(time, longitude)`. This uses
   equal latitude weights and removes no additional seasonal climatology.
3. Use FFT over time and IFFT over longitude. Under this convention eastward
   waves have frequency `f` and planetary wavenumber `k` of the same sign.
   Retain conjugate spectral pairs satisfying all of:

   $$
   fk>0,\qquad k_{\min}\le |k|\le k_{\max},\qquad
   \frac{1}{P_{\max}}\le |f|\le\frac{1}{P_{\min}},
   $$

   $$
   \frac{\sqrt{g h_{\min}}\,|k|\,86400}{2\pi R_E}
   \le |f|\le
   \frac{\sqrt{g h_{\max}}\,|k|\,86400}{2\pi R_E}.
   $$

   `f` is cycles/day, `k` counts waves around the globe, and `sqrt(g*h)` is
   phase speed in m s⁻¹. Inversion produces `kw_recon`; the longitude
   transform's `n_lon` factor cancels on inversion, preserving OLR amplitude
   and units. The FFT assumes periodic time; no taper is applied.
4. Select points equal to the local minimum and below or equal to
   `-sigma_multiplier * kw_recon.std()`. Standard deviation uses all selected
   times/longitudes and `ddof=0`. Longitude wraps during the minimum search;
   time does not. Equal minima are retained without tracking or clustering.
   A zero-variance field produces no events.
5. Average filtered and raw OLR at each event's fixed longitude over the
   configured lags. Out-of-range dates are excluded, so sample counts can vary
   with lag. These composites are plotted rather than saved as text arrays.

### Settings, variables, and outputs

| Name | Default / shape | Meaning or scaling |
| --- | --- | --- |
| `k_ranges` | `(1,13), (1,3), (3,5), (5,7), (7,9), (9,11), (11,13)` | Inclusive bands; adjacent narrow bands share endpoints. |
| `period_min`, `period_max` | `2.5`, `30.0` days | Frequency bounds are their reciprocals. |
| `depth_min`, `depth_max` | `8.0`, `90.0` m | Equivalent-depth bounds. |
| `earth_radius`, `gravity` | `6.37122e6` m, `9.80665` m s⁻² | Spectral-filter constants. |
| `sigma_multiplier` | `2.77` | Threshold in each band's filtered standard deviation; OLR itself is not divided by σ. |
| `time_window` | `7` samples | Local-minimum search over ±3 days. |
| `longitude_half_width` | `15.0` degrees | Radius becomes `floor(radius / spacing + 0.5)` grid points on either side; full window is `2 * half_width_points + 1`, not a fixed 31 points. |
| `lag_days` | `10` | Composite spans −10 through +10 days. |
| `olr_raw`, `kw_recon` | `(n_time, n_lon)` | Unfiltered/filtered anomalies, in input OLR units. |
| `spectrum`, `kelvin_mask` | `(n_time, n_lon)` | Complex transform and Boolean band selection. |
| `time_idx`, `lon_idx` | One pair per event | Zero-based indices into selected OLR dates and the original OLR longitude axis; not ERA5 indices. |

Per-band outputs are `Files/KW_events_netcdf4/k=....csv`, with header
`time_idx,lon_idx`, and, when `make_plots=True`,
`Figure/KW_olr_netcdf4/k=.../{reconstruct,histogram,composite}.png`.
Reconstruction plots show at most the first 365 days.

## `select_valid_column.py`: heating Jacobians

### Collocation and sample construction

`calendar_days()` accepts CF time or ERA5's `day as %Y%m%d.fraction` convention
and reduces timestamps to calendar days. Event indices resolve against the
selected OLR dates. `nearest_longitude()` maps each OLR longitude **value** onto
the target grid by minimizing cyclic distance:

$$
d(\lambda_1,\lambda_2)
=\left|((\lambda_2-\lambda_1+180)\bmod360)-180\right|.
$$

OLR may have a different longitude grid. ERA5 velocity, humidity, and CloudSat
must share pressure levels, longitudes, and selected latitude rows. Their
analysis dates must match OLR in increasing order; dates are matched rather
than assuming identical time origins.

`event_anomalies()` reads one pressure level at a time, converts masked values
to NaN, and removes the time–longitude mean at each level/latitude. At each
event date it rolls the entire longitude field so the event is at index
`n_lon // 2`. Regression uses no temporal lag stack or spatial smoothing.

| Variable | Shape | Role |
| --- | --- | --- |
| `LW_selected`, `SW_selected`, `predictor_selected` | `(L, N_lon, N_event, N_lat)` | Event-centered float32 anomalies; `L` is the pressure-level count. |
| `heating_valid` | `(N_lon, N_event, N_lat)` | Both LW and SW finite at every pressure level. |
| `valid_columns` | `(N_lon, N_event, N_lat)` | Complete heating and current predictor profiles. Humidity gaps do not remove velocity samples. |
| `predictor_valid`, `LW_valid`, `SW_valid` | `(L, N)` | Complete columns; each longitude/event/latitude combination becomes a sample. |
| `J_LW`, `J_SW` | `(L, L)` | Rows: output heating levels; columns: input predictor levels. |
| `relative_lon` | `(N_lon,)` | `(arange(N_lon) - N_lon // 2) * dlon`, degrees. |
| `pressure_hpa` | `(L,)` | Pressure in Pa divided by 100, for plotting only. |

### Regression and unit scaling

For each predictor (`w`, `Q`), fit separate LW/SW models with
`Ridge(alpha=1.0, solver="svd", fit_intercept=True)`. With samples in the rows
of `X` and `Y`, the objective is:

$$
\min_{J,b}\;\|Y-XJ^T-\mathbf{1}b^T\|_F^2+\alpha\|J\|_F^2.
$$

All complete event-centered columns enter the fit. Velocity and humidity are
fitted independently; their effects are not separated in a joint model.

| Quantity | Units / normalization |
| --- | --- |
| Velocity `w` | m s⁻¹; no standardization or density weighting during regression. |
| Specific humidity `Q` | Native kg kg⁻¹; no multiplication by 1000 or `L_v/C_p` during regression. |
| Heating `QLW`, `QSW` | K day⁻¹. |
| Velocity Jacobian | `(K day⁻¹) / (m s⁻¹)`. |
| Humidity Jacobian | `(K day⁻¹) / (kg kg⁻¹)`; a 1 g kg⁻¹ perturbation uses `J_Q * 0.001`. |

`ridge_alpha=1.0` applies in each predictor's native units. Without
standardization, it does not impose the same relative regularization strength
on velocity and humidity. Changing predictor units before fitting changes the
model unless the penalty is adjusted; converting an already fitted Jacobian's
units is a separate operation.

### Diagnostics and outputs

Composite predictions use `J @ predictor_composite`. Fitted intercepts are
saved but **omitted** from these predictions and subsequent modal responses.
There is no training/validation event split. Existing composites may use another
event catalogue, so these are composite diagnostics rather than independent
validation on held-out events.

Metrics use matching finite pairs across all pressure levels and **−50° to
+50°** relative longitude, with equal grid-point weights:

- `r`: centered Pearson pattern correlation.
- `amplitude_ratio`: standard deviation of prediction divided by observation.
- `r2`: `1 - sum((prediction-observation)^2) / sum((observation-mean(observation))^2)`;
  not generally `r**2`.
- `rmse`, `bias`: K day⁻¹; bias is prediction minus observation.
- `n_pairs`: finite-pair count. Undefined metrics are NaN.

Data go to `Files/Jacobian/{w,Q}/k=.../`:

- `Jacob_LW.txt`, `Jacob_SW.txt`, `plev.txt`.
- `intercept_LW.txt`, `intercept_SW.txt`, `fit_info.txt`.
- `event_grid_mapping.csv`, recording dates and OLR/ERA5/CloudSat indices.
- `LW_prediction.txt`, `SW_prediction.txt`, `verification_metrics.csv`.

Figures in `Figure/Jacobian/{w,Q}/k=.../` are `Longwave`, `Shortwave`,
`LW_SW_prediction_scatter`, and `Jacobian`, each in PNG and PDF. LW/SW overlays
share a symmetric prediction color scale within each predictor; Jacobian panels
share their own symmetric scale. Observed contours retain K day⁻¹ units.
Display scales do not change saved values or fitted coefficients.

## `calc_kuang_coefficients.py`: modal projection

### Atmosphere and basis

Use `z = linspace(0, 14000, 141)`: **141 heights at 100-m spacing**. Background
temperature, hydrostatic pressure, and density are:

$$
T(z)=T_0-\Gamma_T z,\qquad
p(z)=p_0\left(1-\frac{\Gamma_T z}{T_0}\right)^{g/(R\Gamma_T)},\qquad
\rho(z)=\frac{p(z)}{RT(z)}.
$$

| Setting | Default | Meaning |
| --- | --- | --- |
| `T0` | `300.0` K | Surface temperature. |
| `Gamma_T` | `0.0065` K m⁻¹ | Environmental lapse rate. |
| `gravity` | `9.81` m s⁻² | Gravity for the modal atmosphere. |
| `gas_constant` | `287.5` J kg⁻¹ K⁻¹ | Dry-air gas constant. |
| `p0` | `1.0e5` Pa | Surface pressure. |
| `heat_capacity` | `1004.5` J kg⁻¹ K⁻¹ | `C_p`; sets `Gamma_d = gravity / heat_capacity`. |
| `latent_heat` | `2.5e6` J kg⁻¹ | `L_v`; model-moisture conversion. |
| `humidity_pressure_pa` | `50000.0` Pa | Single perturbed humidity level. |
| `humidity_plot_amplitude_K` | `1.0` K | Model-moisture amplitude used only in plots. |

For `H = 14000 m`, define:

$$
G_1=\frac{\pi}{2}\sin\left(\frac{\pi z}{H}\right),\qquad
G_2=\frac{\pi}{2}\sin\left(\frac{2\pi z}{H}\right),\qquad
B=(\Gamma_d-\Gamma_T)[G_1,G_2].
$$

`G_stack = [G1, G2]` and `G_J_stack = B` both have shape `(141, 2)`.
The π/2, lapse-rate, and density factors define the coefficient normalization.

### Velocity normalization

The two velocity inputs follow the notebook's density-weighted convention:

$$
w_j(z)=\frac{G_j(z)}{86400\,\rho(z)},\qquad j=1,2.
$$

Density division and the day-to-second factor convert the prescribed modes
to geometric velocity in m s⁻¹ before applying the velocity Jacobian. Their peak
velocity is not normalized to 1 m s⁻¹. Linear interpolation maps them from
`p(z)` to ERA5 `plev`; input velocities outside the 0–14-km domain are zero.
`input_ERA5` has shape `(L, 2)`.

### Humidity normalization

`Q` is native ERA5 specific humidity; model moisture is
`q_model = (L_v / C_p) * Q`, expressed in kelvin. This `Q` is distinct from
radiative heating `QLW`/`QSW`.

Perturb **only the 500-hPa humidity input**, holding all other input levels
fixed. The grid must contain exactly one matching level. A unit model-moisture
perturbation requires:

$$
\delta Q_{500}=\frac{C_p}{L_v}\,(1\ \mathrm{K})
=0.0004018\ \mathrm{kg\,kg^{-1}}
=0.4018\ \mathrm{g\,kg^{-1}}.
$$

`input_ERA5` is `(L, 1)` with one nonzero entry. By the chain rule,
`d(heating)/d(q_model) = (C_p/L_v) * J_Q[:, i500]`. This is a single pressure-level
input rather than two humidity input modes.

### Projection and coefficient interpretation

1. Load `Files/Jacobian/{w,Q}/k=.../{plev,Jacob_LW,Jacob_SW}.txt`.
   Require finite square Jacobians, strictly increasing pressures in Pa, and
   pressure coverage of the full modal atmosphere.
2. Apply `heating_ERA5 = jacobian @ input_ERA5`, in K day⁻¹, then linearly
   interpolate the response to `p(z)` to obtain `heating_modes`.
3. Fit the density-weighted response with `np.linalg.lstsq` and reconstruct:

   $$
   C=\underset{C}{\arg\min}\;\|BC-\rho\,\mathrm{heating\_modes}\|_F^2,
   \qquad\widehat{\mathrm{heating}}=\frac{BC}{\rho}.
   $$

   All 141 equally spaced heights enter the fit. Density multiplies the
   response before fitting; no extra 86400 factor is applied to K day⁻¹ heating.
4. Save each input's relative projection error:

   $$
   \epsilon_j=
   \frac{\|\rho(\widehat{\mathrm{heating}}_j-\mathrm{heating}_j)\|_2}
        {\|\rho\,\mathrm{heating}_j\|_2}.
   $$

   Zero responses have undefined relative error, saved as NaN.

Coefficient **rows are output heating modes**. Velocity matrices are `(2, 2)`
with velocity input modes in columns; humidity matrices are `(2, 1)` with the
single 500-hPa moisture input in the column. These are amplitudes in the basis
`B/rho`, not heating rates at individual levels. For velocity input amplitudes
`a`, reconstruct `B @ C_w @ a / rho`. For a model-moisture perturbation
`delta_q_K`, reconstruct `B @ C_Q[:, 0] * delta_q_K / rho`, in K day⁻¹.

Humidity coefficient files use three input-unit conventions:

| File suffix | Scaling from default coefficient `C_K` |
| --- | --- |
| `*_coefficients.txt` | `C_K`: output-mode amplitudes per K of model moisture at 500 hPa. |
| `*_coefficients_per_kgkg.txt` | `C_K * (L_v / C_p)`: per kg kg⁻¹ specific humidity. |
| `*_coefficients_per_gkg.txt` | `C_K * (L_v / C_p) / 1000`: per g kg⁻¹. |

Increasing `latent_heat` reduces coefficients per model-moisture kelvin by the
chain rule and leaves velocity coefficients unchanged. Changing
`humidity_plot_amplitude_K` scales only plotted humidity responses and
reconstructions; saved coefficients and profile arrays retain unit amplitude.

Outputs in `Files/Kuang_coefficients/{w,Q}/k=.../` include LW/SW coefficient
files, `projection_errors.csv`, `normalization.txt`, and `mode_profiles.npz`.
The NPZ retains `z`, `p`, `rho`, `G_stack`, `G_J_stack`, `plev`, `input_ERA5`,
thermodynamic conversion constants, and LW/SW responses/reconstructions.
With `make_plots=True`, `verification.png` and `verification.pdf` go to
`Figure/Kuang_coefficients/{w,Q}/k=.../`.

## `utils.py`: helper definitions

- `set_matplotlib_style()` loads `~/KW_CloudSat/scientific.mplstyle`; the path
  uses that home-directory location rather than the script location.
- `get_background_profiles()` returns `(z, T, p, rho)` in m, K, Pa, and kg m⁻³,
  using the same lapse-rate atmosphere. Its grid has **71 heights at 200-m
  spacing**, compared with 141 heights at 100-m spacing in the coefficient script.
- `get_vertical_basis(z)` returns `G1, G2` with π/2 sine normalization and
  `H = z.max()`. It applies no density, time, or lapse-rate scaling itself.

## Computational scaling

OLR processing holds the selected latitude slab and time–longitude transforms
in memory. Each FFT filter costs approximately
`O(N_time * N_lon * (log N_time + log N_lon))`. Event extraction reads input
levels separately but retains `L * N_lon * N_event * N_lat` float32 values per
variable, approximately four bytes per value before masks and regression copies.
Memory therefore increases directly with event count and longitude resolution.
For `N` complete samples and `L` levels, with `N >= L`, each dense SVD Ridge
fit costs approximately `O(N * L^2 + L^3)`.

## Tests

The existing tests use `unittest` and temporary synthetic inputs:

```bash
python -m unittest discover -s Code/tests -v
```

Two legacy references need attention before the whole suite can pass with the
present file layout:

- `test_kw_selection_netcdf4.py` points `SCRIPT` at the absent
  `Code/KW_selection_netcdf4.py`; the current selector is `Code/KW_selection.py`.
  The selector's opening docstring retains the old execution filename too.
- `test_calc_kuang_coefficients.py` reads the absent
  `Code/Calc_Kuang's_coefficient.ipynb` in its notebook-equivalence check.
  Its other checks use synthetic Jacobians directly.
