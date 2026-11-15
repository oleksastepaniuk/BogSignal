# BogSignal website: data and image credits

This educational NASA Space Apps research presentation contains modified source
data and derived overview visualizations. Source terms apply independently of
the repository code licence.

- NASA–ISRO NISAR GCOV: NASA/ISRO; accessed from the NASA ASF DAAC. https://nisar-docs.asf.alaska.edu/gcov/
- Copernicus Sentinel-1 and Sentinel-2: contains modified Copernicus Sentinel data (2026), accessed through Google Earth Engine. https://dataspace.copernicus.eu/terms-and-conditions
- Cloud Score+: Google, used for Sentinel-2 quality screening. https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_CLOUD_SCORE_PLUS_V1_S2_HARMONIZED
- Annual global 30 m wetland maps: Zhang, Liu and colleagues, 2022 reference layer from the 2000–2022 series. https://zenodo.org/records/10068479 ; cited source paper https://doi.org/10.5194/essd-15-265-2023
- OpenStreetMap: © OpenStreetMap contributors; ODbL 1.0. Water/wetland polygons and saved, display-simplified country boundaries. https://www.openstreetmap.org/copyright
- European Wetland Map v5: Tegetmeyer, Kaiser and Barthelmes; CC BY 4.0. Peatland evidence in Poland. https://zenodo.org/records/15302184
- Global Peatland Map 2.0: Greifswald Mire Centre / Global Peatlands Assessment; CC BY-NC-SA 4.0. Peat-dominated evidence outside Poland, aligned from approximately 1 km evidence. https://www.greifswaldmoor.de/global-peatland-database-en.html
- ESA WorldCover 2021 v200: © ESA WorldCover project 2021; CC BY 4.0. Land cover used to subdivide mapped peat. https://esa-worldcover.org/en/data-access
- Regional outline: approximate, project-adapted regional map trace and local polygons; reference https://savepolesia.org/. Not an authoritative boundary.
- Cover photograph: Ladislav Stercell, Unsplash. https://unsplash.com/photos/a-body-of-water-surrounded-by-trees-and-grass-kdIyhflzJrM
- Leaflet: Vladimir Agafonkin and contributors, BSD-2-Clause. https://leafletjs.com/

Reference layers are combined and modified during weak-label construction.
Model outputs are predictions, not independently verified habitat maps.
Website map PNGs are nearest-neighbour overview renders of local 20 m rasters.
Swipe RGB backgrounds are 10 m source-scene chips; tiny display gaps are filled
from neighbouring RGB pixels without altering source images or predictions.
Original rasters, checkpoints and native inference products are not included
in the deployed website.
