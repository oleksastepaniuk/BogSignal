# June–October full-input patch comparison

Three complete 256 × 256 patches on the frozen 20 m EPSG:3035 dataset grid were
run through the original model D checkpoint. All six Sentinel-2 reflectance,
four Sentinel-1 VV/VH and four provisional NISAR HH/HV channels change between
June and October. The checkpoint and v4 training-fold normalization do not.

This is an **exploratory prediction comparison**, not evidence of wetland
change, model accuracy or restoration effect. The model was trained on a
single autumn-centered mosaic; June imagery may be outside its training
distribution. The NISAR products are provisional, and the sensors were not
acquired simultaneously. The large class changes require field or independent
dated reference checks before ecological interpretation.

## Patch and source selection

For each site, an existing dataset-grid patch with both provisional NISAR
passes in June and October was selected by the largest number of pixels
intersecting the saved Argos site bounding box. Selection did not use model
predictions. The full patch is always displayed and scored, including land
outside the site box. The boxes are Ramsar wetland envelopes, not restoration
plot boundaries.

| Site | Patch ID | Portion of patch in site box | Jointly valid full patch |
| --- | --- | ---: | ---: |
| Cheremske | `r016896_c009984` | 50.3% | 26.21 km² / 100% |
| Somyne | `r016640_c014848` | 45.0% | 24.39 km² / 93.0% |
| Syra Pohonia | `r015360_c015616` | 100% | 26.21 km² / 100% |

Somyne's 93.0% common coverage is accepted for this exploratory full-patch
display at the user's direction. It is below the regional inference pipeline's
95% contextual validity threshold, so this patch should not be described as
passing that production policy. White pixels remain explicit.

Sentinel-2 uses the nearest clear observation per pixel within 11–30 June or
1–27 October, with the original six-band, SCL and Cloud Score Plus screening
policy. Sentinel-1 ascending and descending scenes are selected separately by
nearest valid acquisition within the same windows. NISAR uses one provisional
HH/HV scene per pass from the saved compatible P05023 series. Sentinel-1 and
NISAR linear power are converted to dB and clipped using v4 bounds.

Actual used source dates vary by pixel. In these patches, Sentinel-2 dates span
21–29 June and 1–12 October at Cheremske; 11–26 June and 1–21 October at
Somyne; and 23–28 June and 1–14 October at Syra Pohonia. Sentinel-1 scenes
are 24–27 June and 10–13 October, depending on site and pass. Provisional
NISAR ascending/descending scenes are 26/27 June and 12/13 October. Exact
scene IDs, per-pixel source-index rasters and SHA-256 checksums are saved in the
local `sentinel_manifest.json` files and NISAR source records. Public display
dates are in `web/public/seasonal_full/manifest.json`.

## Predicted differences

Values below use the **same jointly valid pixels** in each pair. Swamp and
Marsh probabilities are summed before five-class argmax, matching the website
decision rule. Pixel area is 0.0004 km².

| Full patch | June predicted Swamp/Marsh | October predicted Swamp/Marsh | New prediction | Lost prediction |
| --- | ---: | ---: | ---: | ---: |
| Cheremske | 0.03 km² | 2.92 km² | 7,224 pixels | 2 pixels |
| Somyne | 0.09 km² | 9.87 km² | 24,540 pixels | 83 pixels |
| Syra Pohonia | 4.30 km² | 11.21 km² | 17,750 pixels | 472 pixels |

The corresponding within-site-box counts are retained separately in the
machine-readable report. The box is **not** used to blank the displays. White
pixels represent inputs unavailable at one or both dates, primarily the
cloud-screened Sentinel-2 October chip at Somyne.

A diagnostic channel-swap rerun held June inputs fixed and replaced only one
sensor at a time with its October inputs. On the final common area, the
predicted wetland pixels were:

| Site | June | October Sentinel-2 only | October Sentinel-1 only | October NISAR only | All October inputs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cheremske | 77 | 16,875 | 1 | 725 | 7,299 |
| Somyne | 230 | 23,159 | 2,699 | 1,340 | 24,687 |
| Syra Pohonia | 10,746 | 26,757 | 9,454 | 15,364 | 28,024 |

This is a model sensitivity check with interacting inputs, not an additive
attribution. It shows that Sentinel-2 seasonality accounts for much of the
large prediction shift in isolation. Vegetation phenology, composite dates,
residual cloud effects and model generalization can all affect these maps.

## Saved artifacts

`data/inference/full_seasonal_june_oct_2026_v2/` contains one folder per site
and month. Each has the exact Sentinel-2 and two Sentinel-1 source chips,
provisional NISAR linear-power chip, 14-channel model input and validity
rasters, six probability bands, five-class prediction raster, provenance
manifest, and display PNGs for Sentinel-2 RGB, Sentinel-1 ascending VV,
NISAR HH, NISAR HV and an HH/HV false-color composite. Each site folder also
contains a joint-valid raster and prediction-change PNG. The root contains
`report.json` and `channel_swap_diagnostic.json`.

The HH/HV composite is a **visualization only**: red = ascending HH, green =
ascending HV, blue = HH−HV dB contrast. Fixed stretches are used in June and
October. The model receives the physical HH/HV channels, not this composite.
The public site includes only the display PNGs and manifest under
`web/public/seasonal_full/`; raw chips and model outputs remain local.
