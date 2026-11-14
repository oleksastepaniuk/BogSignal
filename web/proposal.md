Argos: app and visualization proposal
Handoff for the visualization / frontend colleague · 13 November 2026

Project context
Argos is our NASA Space Apps project investigating how satellite imagery and machine learning can help locate wetlands and identify places worth assessing for restoration. We focus on Polissia (Polesia), spanning parts of Ukraine, Belarus, Poland and Russia, with initial case studies in Ukrainian Polissia around Cheremske, Somyne and Syra Pohonia.

We want to show where wetlands still exist, including beneath forest cover, and where degraded or drained wetlands might be restoration candidates. An optical image can show vegetation while missing water beneath it; a former wetland may now resemble forest or farmland. We therefore combine present-day observations with historical wetland, peatland and drainage evidence. Local restoration context: FZS.

Our main scientific question is what NISAR adds beyond established satellites. Sentinel-2 supplies optical imagery; Sentinel-1 supplies C-band radar; NISAR supplies longer-wavelength L-band radar. The training colleague will develop per-pixel wetland probability maps using a U-Net baseline, with foundation models/ensembles as a stretch goal. Optional inputs from complex radar pairs may add information about flooded vegetation and hydrological connectivity. Their benefit remains to be measured.

Your role is to turn the evidence and model results into an understandable, inspectable experience. A visitor should understand the problem, explore one real location, compare results with and without NISAR, and see what supports a restoration candidate. We have data-access and raster-preparation tools; trained maps, validated findings and the public app are still to be produced. The design can proceed with clearly labelled reference layers and sample interfaces while those outputs are prepared.

1. Proposed experience: a short story leading into a map
Build a website with two connected modes: “Understand the project” and “Explore the findings.” The first supports a quick presentation; the second lets a colleague or reviewer inspect the evidence.

Suggested story:

The landscape: locate Polissia and explain the existing-wetland / restoration problem using one local example.
Why radar: compare an optical view with radar and a simple explanation of what each measures.
Our experiment: introduce the same segmentation model tested with different sensor inputs.
What we found: show measured performance and one successful or unsuccessful case, with uncertainty.
Explore: open that location in the interactive map and inspect its evidence.
Keep this general enough for the final findings to change. A small or negative NISAR improvement is a valid result to present.

2. What should be in the app
Component	Proposed contents and interaction
Main map	Polissia outline, named case-study locations, zoom/pan, scale, basemap attribution and visible analysis coverage. Open on one prepared site.
Layer controls	Optical imagery, NISAR backscatter, reference wetland/history layers, existing-wetland probability, and restoration-candidate probability/polygons. Include clear legends, dates and opacity controls.
Comparison view	Synchronized maps or a swipe slider: optical versus radar; predictions without versus with NISAR; before/after only where suitable dated observations exist. Keep geography and zoom identical.
Finding sidebar	Clicking a map polygon selects the same item in a short findings list. Show the location, area, observation dates, model version, supporting imagery/references, review status and remaining questions.
NISAR contribution panel	Show optical-only, optical + Sentinel-1, optical + NISAR, and all-sensor results on the same test area. Include wetland IoU/F1, precision/recall, valid area and improvement in percentage points.
Restoration evidence	For each candidate, distinguish historical/peatland evidence, present condition, drainage/connectivity clues and expert eligibility review. Explain why the location deserves investigation.
Methods and sources	A short explanation of labels, geographic test regions, limitations, data dates and links. Add downloadable result GeoJSON/CSV and a shareable case-study view when practical.
Make NISAR's contribution visible in two ways: a quantitative comparison and a local example where the prediction improves, remains unchanged or fails. A difference map shows where predictions changed; independent annotations show whether those changes were correct. Display both when available. For the optional phase-channel experiment, compare against backscatter from the same two dates and the same valid area, as specified in the training proposal.

Use plain labels such as “Existing wetland,” “Restoration candidate” and “Needs review.” The two prediction layers can overlap: an existing wetland may also be degraded. A model probability is not automatically a calibrated confidence or an approval for restoration.

3. Visual and scientific presentation
Give the map most of the screen, with a compact control panel and an evidence sidebar. Keep the opening view understandable without reading the methods page.
Use a consistent probability scale across compared predictions, readable units and a colourblind-friendly palette. Reserve a separate style for historical/reference data and hatching or transparency for unavailable observations.
Distinguish not observed, not analysed, low probability, and no findings. Also handle loading, failed layers and unavailable comparable dates explicitly.
Put sensor and acquisition dates beside every image. If dates differ, say so. Use common radar display ranges for before/after comparisons so separate contrast stretches do not create apparent changes.
Radar brightness, a false-colour composite and a wetland prediction need different legends. If phase/coherence is shown, give it a short explanation; GCOV measures backscatter power, while GSLC retains complex phase. Fringe colours are not land-cover classes.
Label restoration outputs as candidates until eligibility is reviewed. Whole reserve boundaries or site bounding boxes do not mean every enclosed pixel is wetland or a restoration plot.
Label placeholders as “Illustrative” and source-only layers as “Reference data.” Replace them with measured results when available; do not present example accuracy numbers as findings.
4. How we can build it
Recommended starting point: keep the frontend in Argos under web/, using HTML/CSS/JavaScript with Leaflet, and use FastAPI/Python to serve the page and prepared data. This follows our earlier UI discussion and keeps the scientific processing with the Python team. Leaflet supports clickable GeoJSON and georeferenced image overlays; FastAPI supports serving static files. A framework such as React is reasonable if it fits the frontend colleague's experience; it is not required by the current scope.

Start with precomputed case studies: map-ready images/tiles, vector findings and metrics. For a small demo, a static website can consume these directly; add the Python API when querying or processing is needed. Large rasters should be served as tiles rather than loaded whole into the browser. Reproject imagery appropriately for the web map; assigning geographic bounds to an unwarped projected image is not sufficient.

The browser handles presentation, selection and comparisons. The data/training team supplies predictions, validity masks, statistics and evidence. Live global downloads, on-demand model inference and arbitrary area/date analysis can follow once the prepared examples work. An optional explanation assistant can summarize supplied evidence later, with links back to it.

Agree on a small case-study data contract early: site ID/name/bounds; layer URLs, sensor/product, dates, units, legend and masks; finding GeoJSON with evidence/review fields; and metrics JSON with model/input versions, test-region identity and coverage. Preserve numeric rasters separately from display images so any quantitative values come from scientific outputs. This lets interface work proceed while training continues.

5. Visualization examples and ideas to borrow
Start with NASA's NISAR StoryMaps collection. It links directly to the stories below. The collection was updated before NISAR launched, so treat these as application storytelling references and check each story's sensor credits before describing its imagery as actual NISAR data.

Example	Sensor / context	Proposed inspiration for Argos
Glacial Motion StoryMap	Linked from NASA's NISAR applications collection; verify individual image credits	A guided explanation of an initially unfamiliar radar measurement. This is the glacier story linked from the page discussed above.
Floods StoryMap	Same applications collection	Structure our “why radar for wetlands?” introduction around a concrete place and question.
Land Subsidence StoryMap	Same applications collection	Connect a measured map to a practical environmental problem.
Hubbard Glacier: landslide and avalanche debris	Actual NISAR GSLC-derived imagery, 26 Nov / 8 Dec 2025; backscatter/roughness changes, not a glacier-velocity map	Particularly useful UI reference: curtain, toggle and two-up comparison, with dates and explanatory captions.
Mexico City subsidence	Actual NISAR, preliminary Oct 2025–Jan 2026 measurements	A thematic map with quantitative units, recognizable landmarks and explanation of residual noise. Our app needs similarly clear probability/metric legends.
Amazon flooded vegetation	Airborne UAVSAR, Peru, 2013; not NISAR. The accompanying optical image is from a different date.	Wetland-specific illustration of radar revealing different vegetation/water conditions. Borrow the explanatory approach; do not treat the images as a synchronized change pair.
Nepal earthquake on the radar	Sentinel-1A, 2015; not NISAR	Explain an interferogram and its colours before asking users to interpret phase-derived evidence. An educational reference, not an Argos result.
Venezuela earthquake displacement	Actual NISAR earthquake example	An alternative when we want an actual NISAR deformation example alongside the older Nepal illustration.
NISAR flooding in Mozambique — NASA SVS	Actual NISAR flood visualization	Inspiration for a short animated case-study sequence. Open-water flooding and forested-wetland mapping remain different tasks.
These are proposed design inspirations, not a requirement to reproduce every example or broaden Argos beyond wetlands. Source pages and StoryMap destinations were checked on 13 November 2026; interactive stories were not exercised end to end. Most examples are also indexed in docs/research_link_catalogue.md.

6. Suggested first deliverable
Produce a short project introduction and one working, prepared case study: a map, selectable evidence layers, a comparison view, one finding card, and a methods/source panel. Add the NISAR contribution results when the training team supplies them. Until then, show the available imagery/reference evidence with explicit result status.

Validate layer alignment and masks against QGIS, check map/list selection and legends, and rehearse the presentation with a cached data package. Keep the core story readable on a smaller screen, with keyboard-accessible controls and captions for images. A later pass can add more sites, a date timeline, exports and live analysis.

Read the companion training_proposal.md in Argos (TRAINING_PROPOSAL.md in the research workspace) for output definitions and experiments. This document proposes the visual experience; the colleague can choose the layout and frontend tooling that best deliver it.