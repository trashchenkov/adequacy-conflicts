# Data and third-party attribution

## Rye (CC BY 4.0; not MIT)

Per Aaslid (creator), Gro Klæboe (contributor), **Rye microgrid load and generation data, and meteorological forecasts**, Zenodo, 2021, DOI [10.5281/zenodo.4448894](https://zenodo.org/records/4448894). Licensed under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/). The record metadata and original CSV are preserved in `rye/`. No meteorological HDF5 files or unrelated Idjwi data are needed here.

`rye_generation_and_load.csv` is unchanged. `processed.csv` adds the net balance and missing-wind flag; 14 missing wind hours (12 November 2020) are set to zero, an explicit conditional assumption, **not measurements**. The code also interpolates internal gaps up to 6 h; the included data contain no such short gap. No scaling is applied. The observation window is 8280 h, 1 January 13:00 to 11 December 12:00 UTC, not a complete 2020 calendar year. Load peak 70.366622222 kW is rounded to 70 in the manuscript. `audit.json` records exact dates and treatment. Regenerate with `python numerics/rye/prepare_data.py`.

## Yakutia (PVGIS / ERA5 and author-defined transformations)

Weather source: European Commission, Joint Research Centre, **PVGIS 5.3 / ERA5**, 71.7775° N, 139.8471° E, 2005–2023. [PVGIS usage conditions](https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/general-information/usage-conditions-data-protection_en) state that the information is free and unrestricted for use (checked 22 October 2026). Underlying [ERA5 hourly single-level dataset](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview), DOI [10.24381/cds.adbb2d47](https://doi.org/10.24381/cds.adbb2d47), is distributed under the CC-BY licence. These data are not relicensed as MIT.

Generated using Copernicus Climate Change Service information. Neither the European Commission nor ECMWF is responsible for use of the Copernicus information or data contained here.

The included NPZs contain **derived model inputs**, not measured Yakutia loads: modeled wind (EWT DW54-500 curve, 3 MW) and solar (500 kW), assigned 1 MW peak load, February 29 removed from leap years. See `yakutia/source_DATA_PROTOCOL_RU.md`, `source_request.json` and `source_README_RU.md` for transformations. Annual arrays retain all 19 years; winter arrays retain the declared 1008 h windows. Inputs were missing from the supplied paper-code bundle and were recovered from the author's local research archive; **all six recovered files exactly match the hashes in the supplied practical/manifest.json** (`validation/recovered_inputs_sha256.json`). Original weather bulk files are not redistributed; the processed inputs suffice to rerun this paper's numerical study. `source_manifest.json` documents the upstream chain, not a claim that all upstream files are included.

Catalog summaries and equipment parameters are author-created research inputs; source references remain in their JSON records. No new data or parameter grid was collected for this packaging task.
