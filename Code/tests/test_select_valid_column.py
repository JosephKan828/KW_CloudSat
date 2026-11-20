"""Check longitude mapping and the notebook-to-script Jacobian workflow."""

import ast
import contextlib
import csv
import io
from pathlib import Path
import tempfile
import unittest

import netCDF4 as nc
import numpy as np
from sklearn.linear_model import Ridge


SCRIPT = Path(__file__).resolve().parents[1] / "select_valid_column.py"


class CoordinateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load only imports and reusable functions; do not run the analysis.
        tree = ast.parse(SCRIPT.read_text())
        nodes = [node for node in tree.body
                 if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))]
        cls.functions = {}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SCRIPT), "exec"),
             cls.functions)

    def test_longitude_mapping_uses_values_instead_of_reusing_indices(self):
        olr_longitude = np.arange(576) * 0.625 + 0.3125
        target_longitude = np.arange(1440) * 0.25
        indices = self.functions["nearest_longitude"](olr_longitude[[112, 129]],
                                                       target_longitude)
        np.testing.assert_array_equal(indices, [281, 324])

    def test_longitude_mapping_crosses_dateline_and_accepts_negative_longitude(self):
        target = np.array([0.0, 90.0, 180.0, 270.0])
        indices = self.functions["nearest_longitude"](
            np.array([359.9, -90.1, -179.9, 360.1]), target)
        np.testing.assert_array_equal(indices, [0, 3, 2, 0])

    def test_metrics_use_only_matching_finite_pairs(self):
        observed = np.array([1.0, 3.0, np.nan, 5.0])
        predicted = np.array([1.0, 3.0, 100.0, 5.0])
        result = self.functions["calculate_comparison_metrics"](predicted, observed)
        self.assertEqual(result["n_pairs"], 3)
        for name in ("r", "r2", "amplitude_ratio"):
            self.assertAlmostEqual(result[name], 1.0)
        self.assertEqual(result["rmse"], 0.0)
        self.assertEqual(result["bias"], 0.0)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.n_time, self.n_level, self.n_lat, self.n_lon = 8, 3, 3, 32
        self.pressure = np.array([10000.0, 50000.0, 100000.0])
        self.rows = np.array([[1, 0], [3, 3], [6, 7]])
        generator = np.random.default_rng(21)
        self.w = generator.normal(size=(8, 3, 3, 32)).astype(np.float32)
        self.true_lw = np.array([[1, 2, 0], [0, -1, 3], [2, 0, -2]], dtype=np.float32)
        self.true_sw = np.array([[2, 0, -1], [-1, 1, 0], [0, 2, 1]], dtype=np.float32)
        self.lw = np.einsum("ij,tjyx->tiyx", self.true_lw, self.w) + 20.0
        self.sw = np.einsum("ij,tjyx->tiyx", self.true_sw, self.w) - 5.0
        self.w[1, 0, 0, 0] = np.nan
        self.lw[3, 2, 2, 12] = np.nan
        self.sw[6, 1, 1, 28] = np.nan

        with nc.Dataset(self.root / "olr.nc", "w") as dataset:
            dataset.createDimension("time", 8)
            dataset.createDimension("lon", 8)
            time = dataset.createVariable("time", "f8", ("time",))
            time.units = "days since 2006-01-01"
            time[:] = np.arange(8)
            dataset.createVariable("lon", "f8", ("lon",))[:] = np.arange(8) * 45 + 0.3125

        for filename, dimensions, values in (
            ("w.nc", ("time", "plev", "lat", "lon"), {"w": self.w}),
            ("heating.nc", ("time", "lev", "latitude", "longitude"),
             {"QLW": self.lw, "QSW": self.sw}),
        ):
            with nc.Dataset(self.root / filename, "w") as dataset:
                for name, size in zip(dimensions, (8, 3, 3, 32)):
                    dataset.createDimension(name, size)
                time = dataset.createVariable("time", "f8", ("time",))
                if filename == "w.nc":
                    time.units = "day as %Y%m%d.%f"
                    time[:] = 20060101 + np.arange(8) + 0.375
                else:
                    time.units = "days since 2006-01-01"
                    time[:] = np.arange(8)
                dataset.createVariable(dimensions[1], "f8", (dimensions[1],))[:] = self.pressure
                dataset.createVariable(dimensions[2], "f8", (dimensions[2],))[:] = [5, 0, -5]
                dataset.createVariable(dimensions[3], "f8", (dimensions[3],))[:] = np.arange(32) * 11.25
                for name, array in values.items():
                    variable = dataset.createVariable(name, "f4", dimensions, fill_value=-9999)
                    variable[:] = np.where(np.isfinite(array), array, -9999)

        np.savetxt(self.root / "events.csv", self.rows, delimiter=",", fmt="%d",
                   header="time_idx,lon_idx", comments="")
        w_composite = generator.normal(size=(3, 32))
        for name, values in (
            ("w_composite", w_composite), ("LW_composite", self.true_lw @ w_composite),
            ("SW_composite", self.true_sw @ w_composite), ("plev", self.pressure),
        ):
            np.savetxt(self.root / f"{name}.txt", values)

    def run_analysis(self):
        setup, analysis = SCRIPT.read_text().split(
            "# 3. Read event indices and map them to the physical data grids\n", 1)
        namespace = {"__file__": str(SCRIPT)}
        exec(compile(setup, str(SCRIPT), "exec"), namespace)
        namespace.update(event_file=self.root / "events.csv", olr_file=self.root / "olr.nc",
                         w_file=self.root / "w.nc", heating_file=self.root / "heating.nc",
                         composite_dir=self.root, output_dir=self.root / "output",
                         figure_dir=self.root / "figures",
                         event_start=np.datetime64("2006-01-01", "D"),
                         event_end=np.datetime64("2006-01-08", "D"))
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(analysis, str(SCRIPT), "exec"), namespace)
        return namespace

    def test_end_to_end_matches_notebook_regression_and_writes_all_figures(self):
        result = self.run_analysis()
        centered = []
        for data in (self.w, self.lw, self.sw):
            anomaly = data - np.nanmean(data, axis=(0, -1), keepdims=True)
            samples = [np.roll(anomaly[time], 16 - lon * 4, axis=-1)
                       for time, lon in self.rows]
            centered.append(np.asarray(samples).transpose(1, 3, 0, 2))
        mask = np.all(np.isfinite(centered[0]) & np.isfinite(centered[1])
                      & np.isfinite(centered[2]), axis=0)
        for name, heating in zip(("LW", "SW"), centered[1:]):
            model = Ridge(alpha=1, solver="svd").fit(centered[0][:, mask].T,
                                                    heating[:, mask].T)
            actual = np.loadtxt(self.root / "output" / f"Jacob_{name}.txt")
            np.testing.assert_allclose(actual, model.coef_, atol=2e-5, rtol=2e-5)
        np.testing.assert_array_equal(result["w_time_idx"], self.rows[:, 0])
        np.testing.assert_array_equal(result["w_lon_idx"], self.rows[:, 1] * 4)
        with (self.root / "output" / "event_grid_mapping.csv").open() as stream:
            mappings = list(csv.DictReader(stream))
        self.assertEqual(mappings[1]["olr_lon_idx"], "3")
        self.assertEqual(mappings[1]["era5_lon_idx"], "12")
        self.assertEqual(mappings[1]["date"], "2006-01-04")
        for stem in ("Longwave", "Shortwave", "LW_SW_prediction_scatter", "Jacobian"):
            for extension in ("png", "pdf"):
                self.assertGreater((self.root / "figures" / f"{stem}.{extension}").stat().st_size, 0)
        self.assertTrue((self.root / "output" / "verification_metrics.csv").exists())

    def test_missing_target_date_is_rejected_before_fitting(self):
        with nc.Dataset(self.root / "heating.nc", "a") as dataset:
            dataset.variables["time"][-1] = 9
        with self.assertRaisesRegex(ValueError, "CloudSat must cover"):
            self.run_analysis()

    def test_out_of_bounds_olr_index_is_rejected(self):
        np.savetxt(self.root / "events.csv", [[1, 8]], delimiter=",", fmt="%d",
                   header="time_idx,lon_idx", comments="")
        with self.assertRaisesRegex(ValueError, "outside the selected OLR"):
            self.run_analysis()


if __name__ == "__main__":
    unittest.main()
