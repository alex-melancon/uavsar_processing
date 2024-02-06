After obtaining raw UAVSAR data, use polsarpro_t3_freeman.py to process the data and generate the three output files from the F-D Decomposition.
Then, use uavsar_geotiff.py to convert the outputs of polsarpro_t3_freeman.py, the individual polarization bands, and ancillary files (.inc, .hgt) to .tif format for use in GIS software/further analysis. Also generates a false-color RGB with the F-D components, with R = dbl, G = vol, and B = odd (sgl).

--------------------------------------------------------------------------------------------------------------------

rfc_copol_imp_19and23.py shows an example of a random forest classifier trained on UAVSAR and ancillary data for two dates (can also just use one)

INPUTS:

UAVSAR Freeman-Durden Decomposition layers (SGL, DBL, VOL)

UAVSAR co-pol bands (HH, VV)

Global Manmade Impervious Surface data

Rasterized training samples (create a polygon shapefile using GIS software and outline pixels you wish to use for training, then convert to a raster with the same resolution/dimensions as the UAVSAR swath)
