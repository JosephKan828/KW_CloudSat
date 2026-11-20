"""Check notebook equivalence and the single-level humidity projection."""

import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.interpolate import interp1d


SCRIPT = Path(__file__).resolve().parents[1] / "calc_kuang_coefficients.py"
NOTEBOOK = SCRIPT.parent / "Calc_Kuang's_coefficient.ipynb"


class CoefficientTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.plev = np.array([10000, 15000, 20000, 25000, 30000, 40000, 50000,
                              60000, 70000, 80000, 85000, 90000, 92500, 95000, 100000])
        generator = np.random.default_rng(8)
        self.jacobians = {}
        for predictor in ("w", "Q"):
            directory = self.root / "Jacobians" / predictor / "k=1~13"
            directory.mkdir(parents=True)
            np.savetxt(directory / "plev.txt", self.plev)
            for radiation in ("LW", "SW"):
                jacobian = generator.normal(size=(15, 15))
                if predictor == "Q":
                    # Large other columns make accidental use of humidity modes obvious.
                    jacobian *= 1000
                    jacobian[:, 6] = np.linspace(-1, 2, 15)
                self.jacobians[predictor, radiation] = jacobian
                np.savetxt(directory / f"Jacob_{radiation}.txt", jacobian)

    def run_analysis(self, **settings):
        setup, analysis = SCRIPT.read_text().split(
            "# 2. Background atmosphere and two heating basis functions\n", 1)
        namespace = {"__file__": str(SCRIPT)}
        exec(compile(setup, str(SCRIPT), "exec"), namespace)
        namespace.update(jacobian_root=self.root / "Jacobians",
                         output_root=self.root / "coefficients", figure_root=self.root / "figures",
                         make_plots=False)
        namespace.update(settings)
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(analysis, str(SCRIPT), "exec"), namespace)
        return namespace

    def test_w_reproduces_notebook_calculation_and_both_predictors_make_figures(self):
        self.run_analysis(make_plots=True)
        reference = {"np": np, "interp1d": interp1d, "plev": self.plev,
                     "J_LW": self.jacobians["w", "LW"], "J_SW": self.jacobians["w", "SW"]}
        notebook = json.loads(NOTEBOOK.read_text())
        for index in (6, 8, 10, 11):
            exec("".join(notebook["cells"][index]["source"]), reference)
        for radiation in ("LW", "SW"):
            actual = np.loadtxt(self.root / "coefficients" / "w" / "k=1~13"
                                / f"{radiation}_coefficients.txt", ndmin=2)
            np.testing.assert_allclose(actual, reference[f"{radiation.lower()}_coeff"],
                                       rtol=1e-13, atol=1e-13)
        for predictor in ("w", "Q"):
            for extension in ("png", "pdf"):
                image = self.root / "figures" / predictor / "k=1~13" / f"verification.{extension}"
                self.assertGreater(image.stat().st_size, 0)

    def test_humidity_uses_500_hpa_model_kelvin_scaling_and_saves_native_units(self):
        result = self.run_analysis(predictors=("Q",))
        directory = self.root / "coefficients" / "Q" / "k=1~13"
        with np.load(directory / "mode_profiles.npz") as profiles:
            self.assertEqual(profiles["input_ERA5"].shape, (15, 1))
            self.assertEqual(np.count_nonzero(profiles["input_ERA5"]), 1)
            kgkg_per_K = result["heat_capacity"] / result["latent_heat"]
            self.assertAlmostEqual(profiles["input_ERA5"][6, 0], kgkg_per_K)
            for radiation in ("LW", "SW"):
                expected_response = interp1d(
                    self.plev, self.jacobians["Q", radiation][:, 6:7] * kgkg_per_K, axis=0
                )(profiles["p"])
                np.testing.assert_allclose(profiles[f"{radiation}_response"], expected_response)
                expected_coefficient = np.linalg.lstsq(
                    result["G_J_stack"], result["rho"][:, None] * expected_response, rcond=None)[0]
                model = np.loadtxt(directory / f"{radiation}_coefficients.txt", ndmin=2)
                native = np.loadtxt(directory / f"{radiation}_coefficients_per_kgkg.txt", ndmin=2)
                per_gkg = np.loadtxt(directory / f"{radiation}_coefficients_per_gkg.txt", ndmin=2)
                self.assertEqual(model.shape, (2, 1))
                np.testing.assert_allclose(model, expected_coefficient)
                np.testing.assert_allclose(native, model / kgkg_per_K)
                np.testing.assert_allclose(per_gkg, native * 0.001)
        self.assertIn("per K", (directory / "LW_coefficients.txt").read_text())

    def test_humidity_plot_amplitude_does_not_change_the_coefficients(self):
        self.run_analysis(predictors=("Q",), humidity_plot_amplitude_K=1.0)
        path = self.root / "coefficients" / "Q" / "k=1~13" / "LW_coefficients.txt"
        first = np.loadtxt(path)
        self.run_analysis(predictors=("Q",), humidity_plot_amplitude_K=2.0)
        np.testing.assert_array_equal(np.loadtxt(path), first)

    def test_latent_heat_changes_only_moisture_coefficients_by_the_chain_rule(self):
        first = self.run_analysis()
        original = {}
        for predictor in ("w", "Q"):
            for radiation in ("LW", "SW"):
                path = self.root / "coefficients" / predictor / "k=1~13" / f"{radiation}_coefficients.txt"
                original[predictor, radiation] = np.loadtxt(path, ndmin=2)
        self.run_analysis(latent_heat=2 * first["latent_heat"])
        for predictor in ("w", "Q"):
            for radiation in ("LW", "SW"):
                path = self.root / "coefficients" / predictor / "k=1~13" / f"{radiation}_coefficients.txt"
                expected = original[predictor, radiation] * (0.5 if predictor == "Q" else 1.0)
                np.testing.assert_allclose(np.loadtxt(path, ndmin=2), expected)

    def test_nonpositive_latent_heat_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Latent heat must be finite and positive"):
            self.run_analysis(latent_heat=0)

    def test_zero_response_has_zero_coefficients_and_undefined_relative_error(self):
        directory = self.root / "Jacobians" / "w" / "k=1~13"
        for radiation in ("LW", "SW"):
            np.savetxt(directory / f"Jacob_{radiation}.txt", np.zeros((15, 15)))
        self.run_analysis(predictors=("w",))
        output = self.root / "coefficients" / "w" / "k=1~13"
        np.testing.assert_array_equal(np.loadtxt(output / "LW_coefficients.txt"), np.zeros((2, 2)))
        with (output / "projection_errors.csv").open() as stream:
            errors = list(csv.DictReader(stream))
        self.assertEqual(len(errors), 4)
        self.assertTrue(all(np.isnan(float(row["density_weighted_relative_error"])) for row in errors))

    def test_humidity_requires_the_requested_500_hpa_level(self):
        altered = self.plev.copy()
        altered[6] = 50100
        np.savetxt(self.root / "Jacobians" / "Q" / "k=1~13" / "plev.txt", altered)
        with self.assertRaisesRegex(ValueError, "exactly one 500-hPa input level"):
            self.run_analysis(predictors=("Q",))

    def test_unsorted_pressure_levels_are_rejected(self):
        np.savetxt(self.root / "Jacobians" / "w" / "k=1~13" / "plev.txt", self.plev[::-1])
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            self.run_analysis(predictors=("w",))


if __name__ == "__main__":
    unittest.main()
