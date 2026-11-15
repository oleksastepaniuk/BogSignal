# Three-site provisional NISAR comparison, June–October 2026

This is an exploratory **NISAR-only seasonal sensitivity check** of the frozen
website model D. It is not a full seasonal imagery comparison or an evaluation
against independent wetland observations. The executable is
`ml/compare_site_provisional.py`; machine-readable counts and per-patch source
dates are in `data/inference/site_provisional_june_oct_2026/report.json`.

## Inputs and scope

- Study areas are the three bounding boxes in Argos
  `data/targets/polesia_restoration_sites_bbox.geojson`. They approximate Ramsar
  wetland envelopes, not engineering restoration plots.
- NISAR comes solely from the **provisional** `NISAR_L2_GCOV_PROVISIONAL_V1`
  collection, release P05023, version 1.0.11. Ascending acquisitions are
  2026-06-26 and 2026-10-12; descending acquisitions are 2026-06-27 and
  2026-10-13 for the compared patches. No May scene was used.
- The model D checkpoint, v4 normalization, Sentinel-2 and Sentinel-1 patch
  imagery, site geometry, and class decision stay fixed. Only four NISAR
  HH/HV channels change between June and October. Provisional linear power
  is converted to dB and clipped using the v4 policy.
- Counts use only pixels inside the site box and valid for all 14 channels in
  **both** runs. The v4 sampled patches cover portions of each box; the
  figures below do not describe the complete wetland. Each pixel is 20 m,
  or 0.0004 km².

| Site | Compared area | June Swamp/Marsh | October Swamp/Marsh | Net change | Pixels changing five-class prediction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cheremske | 29.87 km² | 7.57 km² | 7.21 km² | −0.36 km² | 8.10% |
| Somyne | 44.22 km² | 22.72 km² | 21.93 km² | −0.79 km² | 7.80% |
| Syra Pohonia | 197.94 km² | 109.16 km² | 108.44 km² | −0.72 km² | 5.56% |

The net figures conceal two-way transitions. In 20 m pixels, non-wetland to
Swamp/Marsh versus Swamp/Marsh to non-wetland were 91 versus 981 at Cheremske,
2,524 versus 4,498 at Somyne, and 9,525 versus 11,332 at Syra Pohonia.

## Interpretation limits

The checkpoint was trained on validated NISAR products, while these inputs are
provisional. A shift in processing or residual radar artifacts may influence
predictions. The June/October difference also reflects seasonal radar response;
it cannot establish hydrological or restoration change. Static Sentinel imagery
means these are **not** complete spring and autumn model predictions. The site
boxes include surrounding terrain, and the saved dataset patches are sampled,
not wall-to-wall coverage. No independent dated reference was scored. Results
should be presented as exploratory model behavior, with the provisional source
and compared footprint stated alongside any visualization.
