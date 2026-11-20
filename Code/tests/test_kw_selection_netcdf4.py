"""Run the linear analysis on small netCDF fixtures with known wave signals."""

import contextlib
from datetime import datetime
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
from netCDF4 import Dataset


SCRIPT = Path(__file__).resolve().parents[1] / "KW_selection_netcdf4.py"


def wave(k=3, period=10, eastward=True, n_time=360, n_lon=64):
    time = np.arange(n_time)[:, None]
    longitude = np.arange(n_lon)[None, :] / n_lon
    direction = -1 if eastward else 1
    return np.cos(2 * np.pi * (time / period + direction * k * longitude))


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "olr.nc"

    def write_olr(self, signal):
        with Dataset(self.path, "w") as dataset:
            n_time, n_lon = signal.shape
            for name, size in (("time", n_time), ("lat", 2), ("lon", n_lon)):
                dataset.createDimension(name, size)
            time = dataset.createVariable("time", "f8", ("time",))
            time.units = "hours since 2006-01-01 00:00:00"
            time[:] = np.arange(n_time) * 24
            dataset.createVariable("lat", "f8", ("lat",))[:] = [2.5, -2.5]
            dataset.createVariable("lon", "f8", ("lon",))[:] = np.arange(n_lon) * 360 / n_lon
            olr = dataset.createVariable("olr", "f8", ("time", "lat", "lon"),
                                         fill_value=-9999.0)
            olr.units = "W/m^2"
            olr[:] = signal[:, None, :] + np.array([10.0, 20.0])[None, :, None]

    def run_analysis(self, **settings):
        # Run the editable settings first, replace paths/settings for the small
        # fixture, then execute the actual analysis unchanged. This lets the
        # production file remain a plain script with no test-only interface.
        setup, analysis = SCRIPT.read_text().split("# 2. Load OLR data", 1)
        namespace = {"__file__": str(SCRIPT)}
        exec(compile(setup, str(SCRIPT), "exec"), namespace)
        namespace.update(olr_data_path=self.path, events_dir=self.root / "events",
                         figures_dir=self.root / "figures", make_plots=False,
                         k_ranges=[(1, 13)])
        namespace.update(settings)
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(analysis, str(SCRIPT), "exec"), namespace)
        return namespace

    def test_eastward_wave_retains_amplitude_and_background_is_removed(self):
        signal = wave()
        self.write_olr(signal)
        result = self.run_analysis()
        np.testing.assert_allclose(result["olr_raw"], signal, atol=1e-12)
        np.testing.assert_allclose(result["kw_recon"], signal, atol=1e-12)

    def test_westward_wave_is_rejected(self):
        self.write_olr(wave(eastward=False))
        result = self.run_analysis()
        np.testing.assert_allclose(result["kw_recon"], 0, atol=1e-12)

    def test_five_day_k1_wave_outside_depth_bounds_is_rejected(self):
        self.write_olr(wave(k=1, period=5))
        result = self.run_analysis()
        np.testing.assert_allclose(result["kw_recon"], 0, atol=1e-12)

    def test_wavenumber_and_period_limits(self):
        self.write_olr(wave())
        result = self.run_analysis(k_ranges=[(5, 7)])
        np.testing.assert_allclose(result["kw_recon"], 0, atol=1e-12)
        self.write_olr(wave(k=8, period=2))
        result = self.run_analysis()
        np.testing.assert_allclose(result["kw_recon"], 0, atol=1e-12)

    def test_mask_has_conjugate_symmetry_on_even_and_odd_grids(self):
        for n_time, n_lon in ((360, 64), (361, 65)):
            self.write_olr(wave(n_time=n_time, n_lon=n_lon))
            mask = self.run_analysis()["kelvin_mask"]
            reflected = mask[np.ix_((-np.arange(n_time)) % n_time,
                                    (-np.arange(n_lon)) % n_lon)]
            np.testing.assert_array_equal(mask, reflected)

    def test_constant_field_has_no_events_and_empty_composites(self):
        self.write_olr(np.zeros((360, 64)))
        result = self.run_analysis()
        self.assertEqual(result["time_idx"].size, 0)
        self.assertTrue(np.isnan(result["composite_filtered"]).all())
        self.assertTrue((result["events_per_lag"] == 0).all())

    def test_events_are_exported_and_composite_centers_match_them(self):
        signal = wave()
        self.write_olr(signal)
        result = self.run_analysis(sigma_multiplier=0.5)
        indices = np.loadtxt(self.root / "events" / "k=1~13.csv",
                             delimiter=",", skiprows=1, dtype=int)
        self.assertGreater(indices.shape[0], 0)
        np.testing.assert_array_equal(indices[:, 0], result["time_idx"])
        np.testing.assert_array_equal(indices[:, 1], result["lon_idx"])
        self.assertAlmostEqual(result["threshold"], -0.5 * signal.std())
        center = result["lag_days"]
        self.assertAlmostEqual(result["composite_filtered"][center],
                               signal[indices[:, 0], indices[:, 1]].mean())

    def test_date_selection_includes_the_end_day(self):
        self.write_olr(wave())
        result = self.run_analysis(start_date=datetime(2006, 2, 1),
                                   end_date=datetime(2006, 2, 3))
        self.assertEqual(result["olr_raw"].shape, (3, 64))
        self.assertEqual(result["time"][-1].isoformat(), "2006-02-03T00:00:00")

    def test_missing_olr_is_rejected(self):
        self.write_olr(wave())
        with Dataset(self.path, "a") as dataset:
            dataset.variables["olr"][0, 0, 0] = -9999
        with self.assertRaisesRegex(ValueError, "missing/non-finite"):
            self.run_analysis()

    def test_missing_day_is_rejected(self):
        self.write_olr(wave())
        with Dataset(self.path, "a") as dataset:
            dataset.variables["time"][-1] += 24
        with self.assertRaisesRegex(ValueError, "daily sampling"):
            self.run_analysis()


if __name__ == "__main__":
    unittest.main()
