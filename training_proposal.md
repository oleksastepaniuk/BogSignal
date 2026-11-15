# Argos: wetland segmentation and restoration-candidate training proposal

**Training handoff · 13 November 2026**

## Project context

**Argos is our NASA Space Apps project investigating how satellite imagery and machine learning can help locate wetlands and identify places worth assessing for restoration.** Our study area is Polissia (also called Polesia), a region spanning parts of Ukraine, Belarus, Poland and Russia; the first case studies focus on Ukrainian Polissia.

We want to answer two practical questions: **where are wetlands still present, including beneath forest cover, and where are degraded or drained former wetlands that might be restored?** These landscapes can look like ordinary forest or agricultural land in an optical image. We will combine present-day observations with historical wetland, peatland and drainage evidence. Documented restoration work at sites such as Somyne and Syra Pohonia gives us relevant case-study locations, although precise training annotations still need to be assembled. [FZS restoration context](https://ukraine.fzs.org/en/project/wetlands-restoration/).

The central research question is **whether NISAR adds useful information that established satellites miss**. Sentinel-2 provides optical information about vegetation and exposed water. Sentinel-1 provides C-band radar observations. NISAR provides longer-wavelength L-band radar, which may reveal complementary vegetation and inundation signals. We will start with radar return strength (backscatter), then optionally add measures derived from the phase relationship between two complex acquisitions. Their value for our sites must be demonstrated by controlled comparisons. [NISAR product overview](https://nisar-docs.asf.alaska.edu/products-overview/).

For the training work, this becomes **image segmentation**: aligned satellite layers go into a model, and it produces a probability for each map pixel. The intended deliverable is a map of existing wetlands and restoration candidates, accompanied by evidence, uncertainty and a test report showing what NISAR contributed. A restoration candidate still needs hydrological and expert review before being considered eligible for intervention. We have data-access and raster-preparation tools; the labelled benchmark and trained models are the next work.

**Direction:** train a U-Net segmentation baseline for Polissia, identifying existing swamps/wetlands and degraded or drained wetlands that could be restoration candidates. Measure what **actual NISAR L-band observations add** beyond Sentinel-2 optical and Sentinel-1 C-band data. Fine-tuned foundation models and a probability ensemble are the stretch goal. The architecture, label scheme and hyperparameters below are proposed starting choices, not completed experiments.

## 1. What we train

Predict **per-pixel probability maps**, rather than one class for an entire image patch. Start in Ukrainian Polissia around Cheremske, Somyne and Syra Pohonia, with additional geographically separate sites and nearby dry forest/agricultural controls. Expand across Polissia after the pilot.

Proposed U-Net outputs use two sigmoid heads, with separate annotation masks:

| Output | Positive supervision | Meaning |
| --- | --- | --- |
| Existing wetland | Dated wetland inventory plus reviewed imagery/observations; GWL swamp/marsh classes for initial weak supervision | Current wetland ecosystem, including forested wetlands where supported by evidence; not simply water visible on one date |
| Degraded/former wetland candidate | Documented drainage/degradation supported by peat or historical wetland evidence and current-condition review | A location to assess for restoration; can overlap an existing but degraded wetland |

**Restoration eligibility adds a screening step:** combine the candidate map with drainage/connectivity, terrain, land use, documented interventions and expert assessment. Record supporting evidence and unresolved constraints per polygon. “Historically wet,” “degraded,” and “eligible for restoration” are different labels. Train a direct eligibility output only once expert-reviewed eligibility annotations exist.

Unknown pixels are ignored in loss and evaluation, never silently treated as negatives. For the existing-wetland head, audited GWL class 0 can supply weak negatives, but does not mean “never a wetland.” For the second head, use reviewed intact wetlands and reviewed non-wetland controls as negatives. If candidate annotations are initially insufficient, train the existing-wetland head first while assembling them; retain restoration identification as a project objective.

## 2. Data to prepare

Use Earth Engine for clipped optical/reference layers and ASF/Earthdata for NISAR. Preserve original rasters, masks, class codes, dates, processing versions and provenance; build separate aligned analysis layers.

| Source | Use and initial selection | Reference / access |
| --- | --- | --- |
| Sentinel-2 L2A | Optical predictor: blue, green, red, narrow NIR, SWIR1/2; cloud/shadow-masked spring and summer composites, with observation counts. Match the label year. Add a near-NISAR-date composite for the 2026 comparison. | [`COPERNICUS/S2_SR_HARMONIZED`](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED) |
| Sentinel-1 GRD | C-band comparator: VV/VH, using consistent mode/orbit selection and comparable temporal windows | [`COPERNICUS/S1_GRD`](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD) |
| NISAR L2 GCOV | Main experimental predictor: available HH/HV backscatter; initially validated products. Record geometry, acquisition dates, CRID and quality, and inspect artifacts. | [Products](https://nisar-docs.asf.alaska.edu/products-overview/), [availability](https://nisar-docs.asf.alaska.edu/availability-overview/), [known issues](https://nisar-docs.asf.alaska.edu/nisar-known-issues/), [ASF search](https://search.asf.alaska.edu/) |
| NISAR L2 GSLC pairs | Optional complex-data predictors: derive interferometric coherence, local phase homogeneity and fringe/phase-discontinuity features from compatible HH acquisition pairs. Preserve pair dates, geometry, processing settings and validity. | [GSLC product guide](https://nisar-docs.asf.alaska.edu/gslc/); papers in Section 6 |
| GWL_FCS30D | Historical weak wetland labels, initially 2020; later 2021/2022 and selected earlier years. Audit classes 181/182 (swamp/marsh); keep permanent water and other classes distinct. Validate local bog representation. | [Annual-map paper](https://doi.org/10.1038/s41597-024-03143-0), [data](https://doi.org/10.5281/zenodo.10068479), [GEE access](https://gee-community-catalog.org/projects/gwl_fcs/) |
| Historical/peatland and restoration evidence | Essential for degraded/former-wetland labels: dated inventories, drainage/intervention polygons and expert/local observations. Polish EWM coverage is a transfer-data candidate after checking source dates and definitions. | [European Wetland Map](https://zenodo.org/records/15302184), [UNEP peatland mapping](https://www.unep.org/en/what-we-do/peatlands/peatland-mapping), [FZS restoration](https://ukraine.fzs.org/en/project/wetlands-restoration/), [Ramsar Ukraine](https://www.ramsar.org/country-profile/ukraine) |
| Supporting maps | JRC open-water history; WorldCover land cover; Dynamic World probabilities/counts. Use for reference review and stratification, initially outside model inputs. | [JRC v1.4](https://developers.google.com/earth-engine/datasets/catalog/JRC_GSW1_4_GlobalSurfaceWater), [WorldCover 2020](https://developers.google.com/earth-engine/datasets/catalog/ESA_WorldCover_v100), [Dynamic World](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_DYNAMICWORLD_V1) |
| Terrain | Optional elevation/slope and hydrological context; keep identical across sensor comparisons. GLO-30 is a surface model, so forest canopy limits fine drainage interpretation. | [Copernicus GLO-30 2024_1](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_DEM_GLO30_2024_1) |

GWL and the other predicted maps are **weak references**, not independent field truth. Map disagreement or loss of a wetland class does not by itself prove drainage. Ramsar boundaries and the existing local site bounding boxes are whole-site extents, not pixel annotations or restoration intervention polygons. Keep reference products used to construct labels out of predictor channels to avoid circular evaluation.

**Two date-consistent experiments:**

- **2020 development benchmark:** Sentinel-2 + Sentinel-1, optionally [PALSAR-2 annual HH/HV](https://developers.google.com/earth-engine/datasets/catalog/JAXA_ALOS_PALSAR_YEARLY_SAR_EPOCH), against audited 2020 GWL labels. This tests the pipeline and historical L-band utility. Annual PALSAR mosaics favor reduced moisture response and do not establish seasonal flooding.
- **2026 NISAR benchmark:** contemporary Sentinel-2/Sentinel-1/NISAR against dated, reviewed 2026 annotations. Older inventories guide annotation but are not current truth. PALSAR cannot be substituted for NISAR to claim a NISAR improvement; cross-sensor transfer needs adaptation and evaluation.

## 3. Models and training

**Baseline: compact multichannel U-Net**, implemented in PyTorch, with one shared encoder/decoder and the two output heads above. Begin with the existing-wetland head where weak labels are available. [U-Net paper](https://arxiv.org/abs/1505.04597).

Suggested first configuration:

- Common **30 m analysis grid** with fixed CRS, affine origin and extent; choose the grid explicitly for the study region. Retain native resolution in raw data. Categorical labels: nearest-neighbour; continuous predictors: documented aggregation/resampling.
- **256 × 256 windows**, stride 256 initially: `X[C,H,W]`, `Y[2,H,W]`, and per-head validity masks. At 30 m, a patch spans 7.68 km. Concatenate seasonal channels; a temporal sequence model is unnecessary initially.
- Apply source calibration before training. Local GCOV crops contain linear gamma-0 power: convert positive values with `10*log10(power)` if using dB. PALSAR DN uses its own calibration; GEE Sentinel-1 GRD is already in dB. Fit normalization on training regions only.
- Masked BCE + Dice per annotated head, AdamW, initial learning rate `3e-4`, batch size 4–8 subject to GPU memory, up to 50 epochs, early stopping after 10 epochs without validation improvement. Use joint spatial flips/rotations of inputs, labels and masks. These are tunable defaults.
- Include dry forests, wet agricultural fields and lowlands as difficult controls. Balance training-window sampling without balancing away the natural prevalence in evaluation. Exclude clouds, invalid radar samples, obvious RFI and uncertain labels; report how much area remains.

**Stretch: foundation-model segmentation and ensemble.** Fine-tune [TerraMind 1.0 Small](https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-small) with a segmentation decoder using supported Sentinel-2/Sentinel-1 modalities. Optionally add an optical/temporal [Prithvi-EO-2.0](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M) segmenter, selecting checkpoint size for available hardware. [TerraTorch](https://github.com/IBM/terratorch) provides fine-tuning tooling.

First train the decoder with the backbone frozen, then fine-tune the backbone at a lower learning rate. Follow each checkpoint's band order, units, normalization and input-size contract. TerraMind's Sentinel-1 pathway does not make NISAR HH/HV interchangeable with Sentinel-1; Prithvi's published EO-2.0 pretraining uses six optical HLS bands.

The simplest proposed fusion is a **validation-weighted average of aligned class probabilities** from the NISAR-capable U-Net and the fine-tuned foundation segmenters. Use matching output definitions, tune weights/calibration on validation data, and compare every member separately. A learned NISAR encoder/fusion adapter is a later alternative. An ensemble improvement alone does not isolate NISAR's contribution.

**Optional additional channels: complex-data / InSAR measures.** Start from processed **GSLC complex HH pairs**, not Level-0 raw echoes. GCOV HH/HV power cannot recover repeat-pass phase. The flooded-vegetation paper motivates local phase homogeneity and fringe-derived features; coherence is an additional proposed feature/quality measure, not a wetland label.

For same-track/frame, same-polarization pairs with compatible grids and processing, verify registration and masks, then form `I = z1 * conjugate(z2)`. Compute features on the complex/native grid before alignment to the analysis grid:

- Local interferometric coherence (0–1).
- Local phase homogeneity and fringe/phase-discontinuity measures, with the paper's definitions inspected before implementation.
- Optionally `sin(angle(I))` and `cos(angle(I))`, an experimental circular encoding rather than treating wrapped radians as an ordinary continuous band.

Choose averaging windows by ground footprint, record temporal separation, and preserve quality masks; low coherence is not proof of dry land. Use circular/complex averaging for phase resampling. Append derived rasters to the U-Net input stack; no phase unwrapping is required for this initial feature experiment. [openSEPPO coherence tooling](https://openseppo.readthedocs.io/en/latest/nisar_coherence_cli/) is an implementation candidate, not an installed Argos component. These measures may support flooded-vegetation/connectivity detection; restoration eligibility still needs separate evidence.

## 4. Required experiment: determine NISAR's contribution

Run the same U-Net and label scheme on the **same 2026 geographic splits, dates/windows and joint valid-pixel intersection**:

| Run | Predictor inputs | Question |
| --- | --- | --- |
| A | Sentinel-2 | Optical baseline |
| B | Sentinel-2 + Sentinel-1 | What does established C-band SAR add? |
| C | Sentinel-2 + NISAR | What does NISAR add to optical? |
| D | Sentinel-2 + Sentinel-1 + NISAR | Does NISAR add information beyond both Sentinel sensors? |
| E (optional) | D inputs + GSLC-derived InSAR channels | Does complex-phase information add value beyond backscatter? |

**Primary result: D minus B. Secondary: C minus A.** Report changes in IoU, Dice/F1, precision, recall and PR-AUC for each output, including forested-wetland and documented degraded-site subsets. Report percentage-point changes, errors and coverage, even if NISAR adds nothing. Keep terrain, optimization budget and sampling rules fixed; use matched seeds, ideally three, and uncertainty estimates by geographic block rather than treating neighboring pixels as independent.

For **E**, rerun D on the pair-available subset with GCOV backscatter from **both corresponding acquisition dates**, then add only the InSAR channels. Compare E against this matched D, on identical pixels and labels, so a second observation or reduced coverage is not mistaken for a phase benefit. Fit any learned feature extraction or tuned thresholds using training/validation regions only.

Assign whole sites/catchments to train, validation and untouched test sets **before window extraction**. Keep all overlapping windows and repeat dates of the same ground in one split; add separation at split boundaries. Tune thresholds and ensemble weights only on validation. Review test annotations independently of model predictions. Report weak-map agreement separately from reviewed-reference performance. NISAR's limited autumn validated record cannot establish year-round benefit.

## 5. Starting state and deliverables for the training owner

The Argos checkout already has NISAR discovery/download/cropping, an **offline** Earth Engine pilot planner, raster grid checks/alignment, window reading and patch indexing. Live GEE exports, final labels, split assignment and training remain to be implemented. The 10 × 10 km acquisition pilot tests engineering; it is not a sufficient training dataset. No measured model performance exists yet.

Read `docs/ground_truth.md`, `docs/raster_alignment.md`, `docs/raster_windows.md`, `docs/patch_index.md` and `configs/gee/pilot.toml` in Argos. Reuse `argos-align-raster`, `RasterStack` and `argos-index-patches`; the indexer's 95% valid threshold checks coverage, not label correctness. GEE's community asset is named `projects/sat-io/open-datasets/GWL_FCS30`; inspect actual years/bands because static GWL_FCS30 and annual GWL_FCS30D are distinct products.

Deliver in order:

1. A dataset manifest and label guide: source dates/versions, grid, channel order, units, per-head masks, label evidence/confidence, geographic split IDs and coverage/class counts.
2. Reproducible U-Net training/inference configs, checkpoints and probability GeoTIFFs, with a geographic test report.
3. The A–D comparison table, optional E phase-channel comparison, and a clear statement of NISAR's measured incremental contribution and uncertainty.
4. Restoration-candidate polygons with evidence and eligibility-review status; foundation-model/ensemble results if time permits.

## 6. Research and discussion provenance

- [Melnychenko & Solovey (2024)](https://doi.org/10.3390/w16030407): regional Sentinel-1/2 wetland mapping for Polish–Ukrainian basins; useful local methods reference.
- [Huebinger et al., *Exploring the potential of using L-band InSAR for mapping flooded vegetation in tropical wetlands* (2026)](https://doi.org/10.1016/j.rse.2025.115086): combines fringe detection and local phase homogeneity using **ALOS PALSAR-1, not NISAR**, in Colombia's Atrato floodplain. [Author-institution abstract](https://www.pnnl.gov/publications/exploring-potential-using-l-band-insar-mapping-flooded-vegetation-tropical-wetlands). Methods evidence for the optional channels; transfer to Polissia remains experimental.
- [Huebinger & Jaramillo, *Hydrological barriers compound pressure on the world's most threatened wetlands*, preprint v1](https://doi.org/10.21203/rs.3.rs-9855710/v1): includes **actual NISAR complex HH GSLC**, wrapped interferograms and phase discontinuities, with masking and 5 × 5 multilooking, without phase unwrapping. Supports the connectivity-feature experiment, not a trained Polissia segmentation model or restoration-eligibility result. Both papers are indexed in research catalogue Section 4.
- ChatGPT project discussions reviewed: **NISAR Ukrainian Swamps Feasibility**, **LightGBM First Model Comparison**, **Dataset strategy and sources**, **Plan Training Dataset**, **Plan NISAR Coverage**, **Summarize Today’s Plan**. Codex: **Plan GEE data acquisition**, plus current handoff/research documentation. This proposal incorporates the user's latest clarification: **segmentation; U-Net baseline; foundation-model ensemble stretch; existing and restoration-candidate wetlands; explicit NISAR contribution**.
- Extended sources and corrections: Argos `docs/research_link_catalogue.md`; research workspace `references/README.md`. Core dataset/model documentation rechecked on 13 November 2026; additional references come from the project catalogue/local papers. Actual asset coverage and label availability still require acquisition-stage verification.
