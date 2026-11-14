# Argos — Current Dataset Design

**Updated:** 2026-11-14  
**Status:** Dataset specification agreed in principle; acquisition, alignment, labels, and training dataset assembly remain in progress.

## 1. Objective

Build a multi-sensor geospatial dataset for supervised semantic segmentation of wetlands in Polissia, with particular interest in forested wetlands and potentially degraded or drained areas.

The central research question is:

**Does NISAR L-band SAR improve wetland detection compared with Sentinel-2 optical and Sentinel-1 C-band observations?**

We distinguish three concepts:

- **Existing wetlands:** wetland ecosystems, including forested wetlands that may not show exposed water.
- **Flooded vegetation:** vegetation with detectable surface inundation, potentially identifiable using L-band radar and InSAR.
- **Former/degraded wetlands:** areas potentially suitable for restoration, requiring historical, peatland, drainage, and ecological evidence.

These are related but not interchangeable targets.

## 2. Geographic scope

**Primary AOI:** Polissia, covering parts of Ukraine, Poland, Belarus, and Russia.

A combined Polissia boundary has been prepared. Initial training and validation should focus on Ukrainian Polissia, with geographically separate examples from other areas where suitable reference data exist.

Priority Ukrainian case studies:

- Cheremske Bog
- Somyne Swamps
- Syra Pohonia Bog

The AOI boundary defines the geographic extent of analysis; it is not a wetland label.

## 3. Model input layers

The proposed input stack combines six Sentinel-2 optical bands with Sentinel-1 and NISAR backscatter acquired in ascending and descending geometries.

### Sentinel-2 L2A — optical

| Band | Wavelength (approx.) | Purpose |
|---|---|---|
| B2 — Blue | 490 nm | Water/atmospheric and visible-surface information |
| B3 — Green | 560 nm | Water and vegetation discrimination |
| B4 — Red | 665 nm | Chlorophyll absorption and vegetation condition |
| B8A — Narrow NIR | 865 nm | Vegetation structure and spectral response |
| B11 — SWIR1 | 1610 nm | Vegetation and soil moisture |
| B12 — SWIR2 | 2190 nm | Moisture stress and vegetation/soil discrimination |

**Selected: 6 optical channels.**

Source: `COPERNICUS/S2_SR_HARMONIZED`

Preferred preprocessing:

- Cloud and shadow masking
- Spring and summer composites
- Initially April–May and July–August
- Observation counts and validity masks
- Consistent temporal selection between experiments

Seasonal composites are preferable to relying on one optical acquisition, especially when identifying persistent wetland ecosystems.

### Sentinel-1 GRD — C-band SAR

| Geometry | Channels |
|---|---|
| Ascending | VV, VH |
| Descending | VV, VH |

**Proposed: 4 channels.**

Source: `COPERNICUS/S1_GRD`

Use calibrated backscatter (dB) and consistent acquisition mode, orbit geometry, and temporal windows. Ascending and descending observations provide complementary viewing geometries.

### NISAR L2 GCOV — L-band SAR

| Geometry | Channels |
|---|---|
| Ascending | HH, HV |
| Descending | HH, HV |

**Proposed: 4 channels.**

Primary collection: `NISAR_L2_GCOV_V1`

Use calibrated backscatter, preferably validated products. GCOV power values require appropriate conversion and normalization.

The actual availability of both polarizations and both viewing geometries must be confirmed for each AOI. Missing acquisitions must not be represented as real observations.

### Proposed combined model input

| Sensor | Channels |
|---|---:|
| Sentinel-2 | 6 |
| Sentinel-1 ascending | 2 |
| Sentinel-1 descending | 2 |
| NISAR ascending | 2 |
| NISAR descending | 2 |
| **Total** | **14** |

**14 channels is the proposed maximum stack, not a verified or assembled dataset.**

If the acquisition coverage is insufficient, the experimental configuration must use available, explicitly documented channels and consistent comparison areas.

## 4. Reference datasets and labels

OSM, GWL, and Ramsar have different roles. They are not interchangeable sources of ground truth, and they are not part of the initial 14 predictor channels.

| Dataset | Primary role | Limitation |
|---|---|---|
| **GWL_FCS30D** | Main source of historical weak wetland labels | Model-derived classification, not independent field truth |
| **OpenStreetMap (OSM)** | Supporting wetland/water polygons, geographic context and manual cross-checks | Inconsistent completeness and tagging |
| **Ramsar sites** | Known wetland complexes, case-study selection, expert review | Protected-area boundaries do not delineate all wetland pixels |
| GLWDv2 | Complementary wetland reference | Coarse or aggregated wetland information |
| European Wetland Map | Additional reference, particularly for Poland | Coverage, dates, and class compatibility require review |
| JRC Global Surface Water | Historical open-water occurrence and seasonality | Does not comprehensively map water beneath vegetation |
| ESA WorldCover / Dynamic World | Land-cover context and negative/control sampling | Different class definitions and potential classification errors |
| Historical peatland/drainage records | Former-wetland and degradation evidence | Availability and spatial accuracy vary |

### GWL_FCS30D

**Global annual wetland classification at 30 m, 2000–2022.**

Initial focus: 2020–2022, preserving original class codes.

Relevant classes include:

- `180` — Permanent water
- `181` — Swamp
- `182` — Marsh
- `183` — Flooded flat
- `184–187` — Additional wetland categories, to be reviewed individually

The original multiclass raster should be preserved. Binary masks can be derived later using an explicit, documented class mapping.

The community Earth Engine asset uses the identifier:

`projects/sat-io/open-datasets/GWL_FCS30`

The asset's available annual images and class coding must be inspected before export; the scientific products GWL_FCS30 and GWL_FCS30D are distinct.

**Important:** GWL is weak supervision, not unquestionable ground truth. OSM and Ramsar agreement can help prioritize review but cannot automatically establish independently validated labels.

## 5. Spatial and temporal organization

**Initial common analysis resolution: 30 m.**

- Preserve original source resolution and raw scientific values.
- Align all predictor and reference rasters to one explicit CRS, grid origin, extent, and pixel size.
- Resample categorical labels with nearest-neighbour interpolation.
- Apply appropriate resampling/aggregation to continuous imagery.
- Keep missing-data and label-validity masks separate.
- Track acquisition dates, product versions, polarizations, geometry, and source provenance.

Proposed patch specification:

`X: [C, 256, 256]`  
`Y: [K, 256, 256]`  
`Validity masks: [K, 256, 256]`

At 30 m, a 256 × 256 patch covers approximately 7.68 × 7.68 km.

Seasonal information is initially represented by stacked composites rather than a recurrent or temporal-sequence model. The 14-channel layout above does not yet include duplicated spring/summer optical channels.

## 6. Two complementary dataset configurations

### Historical development dataset — 2020

**Purpose:** Build and debug the segmentation pipeline using relatively complete historical data.

Inputs:

- Sentinel-2 spring/summer imagery
- Sentinel-1 C-band backscatter
- Optional PALSAR-2 annual L-band HH/HV

Labels:

- Audited GWL_FCS30/GWL_FCS30D 2020 wetland classes
- Supporting OSM, European Wetland Map, and manually reviewed references

This dataset provides initial training and weak-label benchmarking, **not proof of NISAR's contribution**.

### Contemporary NISAR dataset — 2026

**Purpose:** Test whether actual NISAR observations improve wetland segmentation.

Inputs:

- Sentinel-2 optical
- Sentinel-1 C-band SAR
- NISAR L-band GCOV

Labels:

- Independently reviewed, dated contemporary wetland annotations
- Historical inventories used as supporting evidence, not automatic current labels

Validated NISAR Level 0–2 products were publicly released on 2026-11-06, with validated acquisition coverage beginning 2026-10-21. Earlier provisional data may be useful but must be distinguished from validated observations.

A metadata catalogue and acquisition tools exist; the complete usable 2026 training coverage is not yet established.

## 7. Planned evaluation

Start with a U-Net segmentation baseline and evaluate the same architecture under controlled sensor ablations.

| Experiment | Inputs |
|---|---|
| A | Sentinel-2 |
| B | Sentinel-2 + Sentinel-1 |
| C | Sentinel-2 + NISAR |
| D | Sentinel-2 + Sentinel-1 + NISAR |

**Primary comparison: D versus B** — NISAR's additional contribution beyond the two Sentinel missions.

Use identical geographic splits, compatible observation windows, common valid-pixel coverage, and independently reviewed evaluation labels.

Report IoU, Dice/F1, precision, recall, PR-AUC and coverage, including performance on forested wetlands and difficult dry-land controls.

A later optional experiment can add GSLC-derived interferometric coherence, local phase homogeneity, and phase-discontinuity measures. These require compatible complex acquisition pairs, not GCOV backscatter alone.

## 8. Current state and remaining work

**Established or planned in Argos:**

- Combined Polissia geographic AOI
- NISAR metadata discovery, download and cropping utilities
- Raster alignment and patch-indexing utilities
- GWL_FCS30D selected as principal historical weak-label source
- Six Sentinel-2 spectral bands selected
- Sentinel-1 and NISAR ascending/descending stacks proposed
- U-Net selected as the first segmentation baseline



## 9. Core dataset principle

**Predictor imagery, historical reference maps, protected-area inventories, and independently reviewed ground truth must remain separate.**

Argos should demonstrate what L-band SAR adds to wetland mapping without confusing map agreement with ground-truth accuracy or a remotely detected wetland with a confirmed restoration opportunity.