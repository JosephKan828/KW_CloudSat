# ERA5-GRIB Kelvin-wave workflow

Run the scripts from the repository root in this order:

```bash
python Code/ERA5_GRIB/ERA5_composite.py
python Code/ERA5_GRIB/QR_composite.py
python Code/ERA5_GRIB/QR_w_Relation.py --data-type composite --components 3
```

The first two commands use the event files in `Files/ERA5_GRIB/KW_events` and
write aligned 15-level fields to:

```text
Files/ERA5_GRIB/composites/
├── coordinates.npz
└── k=<lower>~<upper>/
    ├── composite/
    │   ├── w.npy
    │   ├── temperature.npy
    │   ├── LW.npy
    │   └── SW.npy
    └── concat/                 # only with --save-concat
        ├── w.npy
        ├── temperature.npy
        ├── LW.npy
        └── SW.npy
```

Response matrices and metrics are written to
`Files/ERA5_GRIB/linear_response/<data-type>`. Figures are written under
`Figure/ERA5_GRIB`.

Event concatenations can be very large. Generate them only when needed:

```bash
python Code/ERA5_GRIB/ERA5_composite.py --save-concat
python Code/ERA5_GRIB/QR_composite.py --save-concat
python Code/ERA5_GRIB/QR_w_Relation.py --data-type concat --components 3
```
