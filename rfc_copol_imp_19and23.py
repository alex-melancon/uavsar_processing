# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 11:31:59 2021

@author: amelanco

Try training a classifier for 9/19 and 9/23 to see what happens
"""

from osgeo import gdal
import numpy as np
from osgeo import osr
import gdalconst
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve
from sklearn.ensemble import RandomForestClassifier
import joblib


# ----------------------------------------------------------------------------
# DEFINE FUNCTIONS

#get_geo opens raster, gets raster size, transform, projection
def get_geo(filename, band=1):
    ds = gdal.Open(filename)
    cols = ds.RasterXSize
    rows = ds.RasterYSize
    img = ds.GetRasterBand(band).ReadAsArray(0,0,cols,rows)
    in_geo = ds.GetGeoTransform()
    projref = ds.GetProjectionRef()
    
    return img

# https://gis.stackexchange.com/questions/57834/how-to-get-raster-corner-coordinates-using-python-gdal-bindings
def GetExtent(gt,cols,rows):
    ''' Return list of corner coordinates from a geotransform

        @type gt:   C{tuple/list}
        @param gt: geotransform
        @type cols:   C{int}
        @param cols: number of columns in the dataset
        @type rows:   C{int}
        @param rows: number of rows in the dataset
        @rtype:    C{[float,...,float]}
        @return:   coordinates of each corner
    '''
    ext=[]
    xarr=[0,cols]
    yarr=[0,rows]

    for px in xarr:
        for py in yarr:
            x=gt[0]+(px*gt[1])+(py*gt[2])
            y=gt[3]+(px*gt[4])+(py*gt[5])
            ext.append([x,y])
            #print(x,y)
        yarr.reverse()
    return ext

# from https://github.com/jgomezdans/eoldas_ng_observations/blob/reference/eoldas_ng_observations/eoldas_observation_helpers.py#L29
def reproject_image_to_reference(reference, input_file, output_file, res=None): 
    input_file_ds = gdal.Open( input_file )
    if input_file_ds is None:
        print("GDAL could not open input_file file %s " % input_file)
    input_file_proj = input_file_ds.GetProjection()
    input_file_geotrans = input_file_ds.GetGeoTransform()
    data_type = input_file_ds.GetRasterBand(1).DataType
    n_bands = input_file_ds.RasterCount

    reference_ds = gdal.Open( reference )
    if reference_ds is None:
        print("GDAL could not open reference file %s " % reference)
    reference_proj = reference_ds.GetProjection()
    reference_geotrans = reference_ds.GetGeoTransform()
    w = reference_ds.RasterXSize
    h = reference_ds.RasterYSize
    if res is not None:
        reference_geotrans[1] = float( res )
        reference_geotrans[-1] = - float ( res )

    dst_filename = output_file #input_file.replace( ".tif", "_crop.vrt" )
    #dst_ds = gdal.GetDriverByName('VRT').Create(dst_filename,
    #                                           w, h, n_bands, data_type)
    dst_ds = gdal.GetDriverByName('GTiff').Create(dst_filename,
                                                w, h, n_bands, data_type)
    dst_ds.SetGeoTransform( reference_geotrans )
    dst_ds.SetProjection( reference_proj)

    gdal.ReprojectImage( input_file_ds, dst_ds, input_file_proj,
                         reference_proj, gdal.GRA_NearestNeighbour)
    dst_ds = None  # Flush to disk
    return dst_filename


# Function writes data to a .tif in WGS84 projection
def dump_geotiff_float(outf, arr, in_geo):
    
    format = 'GTiff'
    rows, cols = np.shape(arr)
    driver = gdal.GetDriverByName(format)
    out_ds = driver.Create(outf, cols, rows, 1, gdalconst.GDT_Float32)
    out_cs = osr.SpatialReference()
    out_cs.ImportFromEPSG(4326)
    out_ds.SetProjection(out_cs.ExportToWkt())
    out_ds.SetGeoTransform(in_geo)
    out_ds.GetRasterBand(1).WriteArray(arr)
    out_ds = None
    print('File generated:',outf)
    return outf

def bytescale(arr, cmin=0, cmax=1, low=0, high=255):
    # First, we need to use numpy to 'clip' our array to the range of values desired... this will set all values
    # within the 'clip' to that... for example, if I "clip" to 0 to 1, a value of 1.1 is replaced as 1, and a value
    # of -0.1 would be set to 0. 
    arr = np.clip(arr, cmin, cmax)
    # Calculate the 'slope' of the line: change in byte value (Y) per change in physical value (X)
    m = (high-low)/(cmax-cmin)
    # With 'm' defined, use algebra to solve for b: input values for xmax (cmax here) and ymax (high, here)
    b = high-(m*cmax)
    img = np.uint8((m*arr) + b)
    #print('range of img:',np.min(img),np.max(img))
    # Pass the image back to where it was called from
    return img

def dump_geotiff_byte(outf, arr, in_geo, projref):
    # Let's write this out as a GeoTIFF...  we won't burn in a color table but the output will have the values
    # of our 'classes' array corresponding to:
    format = 'GTiff'
    rows, cols = np.shape(arr)
    driver = gdal.GetDriverByName(format)
    out_ds = driver.Create(outf, cols, rows, 1, gdalconst.GDT_UInt8)
    out_cs = osr.SpatialReference()
    out_cs.ImportFromWkt(projref)
    out_ds.SetProjection(out_cs.ExportToWkt())
    out_ds.SetGeoTransform(in_geo)
    out_ds.GetRasterBand(1).WriteArray(arr)
    out_ds = None
    print('File generated:',outf)
    return outf

# ============================================================================
# Read in data
# ============================================================================ 

# =============================================================================
# # 9/19 -----------------------------------------------------------------------
# vol_19_f = "C:/Users/amelanco/ProjectData/florence/processed_uavsar/lumber_31509/919/31509_19_vol.tif"
# dbl_19_f = "C:/Users/amelanco/ProjectData/florence/processed_uavsar/lumber_31509/919/31509_19_dbl.tif"
# sgl_19_f = "C:/Users/amelanco/ProjectData/florence/processed_uavsar/lumber_31509/919/31509_19_sgl.tif"
# 
# vol_19 = get_geo(vol_19_f, band=1)
# dbl_19 = get_geo(dbl_19_f, band=1)
# sgl_19 = get_geo(sgl_19_f, band=1)
# 
# # Extent of the UAVSAR image?
# ds = gdal.Open(vol_19_f)
# cols = ds.RasterXSize
# rows = ds.RasterYSize
# gt = ds.GetGeoTransform()
# ext = GetExtent(gt, cols, rows)
# ul, ll, lr, ur = ext
# 
# 
# # Clip data to range of 0-1
# #   - Original data has nodata value of 1e-30, and some spikes/noise in bright areas
# #     that are much greater than 1
# vol_19 = np.clip(vol_19, 0, 1)
# dbl_19 = np.clip(dbl_19, 0, 1)
# sgl_19 = np.clip(sgl_19, 0, 1)
# 
# # Want to use HH/VV as well, so read those in too
# hh_19_f = "C:/Users/amelanco/ProjectData/florence/processed_uavsar/lumber_31509/919/uavsar_31509_19_HH.tif"
# vv_19_f = "C:/Users/amelanco/ProjectData/florence/processed_uavsar/lumber_31509/919/uavsar_31509_19_VV.tif"
# HH_19 = get_geo(hh_19_f, band=1)
# VV_19 = get_geo(vv_19_f, band=1)
# # Clip to 0-1
# HH_19 = np.clip(HH_19, 0, 1)
# VV_19 = np.clip(VV_19, 0, 1)
# 
# imp_19_f = "C:/Users/amelanco/ProjectData/florence/gmis_impervious_surface_percentage/uavsar_31509_imp_sfc.tif"
# imp_19 = get_geo(imp_19_f, band=1)
# # Clip impervious surface data to 0-100% by masking out areas above 100 (not impervious)
# imp_19_locs = np.where(imp_19 > 100)
# imp_19[imp_19_locs] = 0.0
# 
# #plt.imshow(imp_19, cmap='jet')
# #plt.colorbar()
# 
# #dbl_18 = dbl_18[:,0:19703]
# =============================================================================


# 9/23 -----------------------------------------------------------------------
# All swaths/arrays have been reprojected to match 9/19 for sanity
vol_23_rpj = r"C:\Users\amelanco\ProjectData\florence\processed_uavsar\lumber_31509\923\31509_23_vol_rpj.tif"
dbl_23_rpj = r"C:\Users\amelanco\ProjectData\florence\processed_uavsar\lumber_31509\923\31509_23_dbl_rpj.tif"
sgl_23_rpj = r"C:\Users\amelanco\ProjectData\florence\processed_uavsar\lumber_31509\923\31509_23_sgl_rpj.tif"
vol_23 = get_geo(vol_23_rpj, band=1)
dbl_23 = get_geo(dbl_23_rpj, band=1)
sgl_23 = get_geo(sgl_23_rpj, band=1)

# Clip data to range of 0-1
#   - Original data has nodata value of 1e-30, and some spikes/noise in bright areas
#     that are much greater than 1
vol_23 = np.clip(vol_23, 0, 1)
dbl_23 = np.clip(dbl_23, 0, 1)
sgl_23 = np.clip(sgl_23, 0, 1)

# Extent of the UAVSAR image?
ds = gdal.Open(vol_23_rpj)
cols = ds.RasterXSize
rows = ds.RasterYSize
gt = ds.GetGeoTransform()
ext = GetExtent(gt, cols, rows)
ul, ll, lr, ur = ext

# Want to use HH/VV as well, so read those in too
hh_23_rpj = r"C:\Users\amelanco\ProjectData\florence\processed_uavsar\lumber_31509\923\31509_23_HH_rpj.tif"
vv_23_rpj = r"C:\Users\amelanco\ProjectData\florence\processed_uavsar\lumber_31509\923\31509_23_VV_rpj.tif"
HH_23 = get_geo(hh_23_rpj, band=1)
VV_23 = get_geo(vv_23_rpj, band=1)
# Clip to 0-1
HH_23 = np.clip(HH_23, 0, 1)
VV_23 = np.clip(VV_23, 0, 1)

imp_23_f = "C:/Users/amelanco/ProjectData/florence/processed_uavsar/lumber_31509/923/31509_23_imp_rpj.tif"
imp_23 = get_geo(imp_23_f, band=1)
# Clip impervious surface data to 0-100% by masking out areas above 100 (not impervious)
imp_23_locs = np.where(imp_23 > 100)
imp_23[imp_23_locs] = 0.0


# ============================================================================
# Set up training samples
# ============================================================================ 

# 19th first
#print('Gathering samples for 19-SEP')
#print(' ')
# Build mask of valid data
validmask_23 = np.zeros( (rows,cols), 'uint8')
valid_23 = np.where( (dbl_23 > 0) & (vol_23 > 0) & (sgl_23 > 0) & (HH_23 > 0) & (VV_23 > 0) )
validmask_23[valid_23] = 1 

# =============================================================================
# 
# # Read in training data
# #classes_f = r"C:\Users\amelanco\ProjectData\florence\processed_uavsar\lumber_31509\31509_classes_v3.tif"
# classes_rpj = r"C:\Users\amelanco\ProjectData\florence\processed_uavsar\lumber_31509\uavsar_31509_classes_v3.tif"
# #result = reproject_image_to_reference(vol_f, classes_f, classes_rpj)
# classes = get_geo(classes_rpj, band=1)
# 
# 
# X = []
# Y = []
# 
# # ---------------------------------------------------------------------------
# print('Gathering samples for class: open water')
# class_r, class_c = np.where( (classes == 1) & (validmask_19 == 1) )
# nth = 1
# class_r = class_r[0::nth]
# class_c = class_c[0::nth]
# 
# classno = 0
# npoints = len(class_r)
# print('There were this many points retained for water:',npoints)
# 
# 
# dbl_class = dbl_19[class_r, class_c]
# vol_class = vol_19[class_r, class_c]
# sgl_class = sgl_19[class_r, class_c]
# HH_class = HH_19[class_r, class_c]
# VV_class = VV_19[class_r, class_c]
# imp_class = imp_19[class_r, class_c]
# 
# samples = list(zip(dbl_class, vol_class, sgl_class, HH_class, VV_class, imp_class))
# 
# # Append to main lists
# X = X + samples
# Y = Y + ([classno] * len(samples))
# print(' ')
# 
# # ----------------------------------------------------------------------------
# print('Gathering samples for class: dry forest')
# class_r, class_c = np.where( (classes == 2) & (validmask_19 == 1) )
# nth = 1
# class_r = class_r[0::nth]
# class_c = class_c[0::nth]
# 
# 
# classno = 1
# npoints = len(class_r)
# print('There were this many points retained for dry forest:',npoints)
# 
# 
# dbl_class = dbl_19[class_r, class_c]
# vol_class = vol_19[class_r, class_c]
# sgl_class = sgl_19[class_r, class_c]
# HH_class = HH_19[class_r, class_c]
# VV_class = VV_19[class_r, class_c]
# imp_class = imp_19[class_r, class_c]
# 
# samples = list(zip(dbl_class, vol_class, sgl_class, HH_class, VV_class, imp_class))
# 
# # Append to main lists
# X = X + samples
# Y = Y + ([classno] * len(samples))
# print(' ')
# 
# # ----------------------------------------------------------------------------
# print('Gathering samples for class: inundated')
# class_r, class_c = np.where( (classes == 3) & (validmask_19 == 1) )
# nth = 1
# class_r = class_r[0::nth]
# class_c = class_c[0::nth]
# 
# classno = 2
# npoints = len(class_r)
# print('There were this many points retained for inundated:',npoints)
# 
# 
# dbl_class = dbl_19[class_r, class_c]
# vol_class = vol_19[class_r, class_c]
# sgl_class = sgl_19[class_r, class_c]
# HH_class = HH_19[class_r, class_c]
# VV_class = VV_19[class_r, class_c]
# imp_class = imp_19[class_r, class_c]
# 
# samples = list(zip(dbl_class, vol_class, sgl_class, HH_class, VV_class, imp_class))
# 
# # Append to main lists
# X = X + samples
# Y = Y + ([classno] * len(samples))
# print(' ')
# 
# # ----------------------------------------------------------------------------
# print('Gathering samples for class: fields/non-forest veg')
# class_r, class_c = np.where( (classes == 4) & (validmask_19 == 1) )
# nth = 1
# class_r = class_r[0::nth]
# class_c = class_c[0::nth]
# 
# classno = 3
# npoints = len(class_r)
# print('There were this many points retained for fields/non-forest veg:',npoints)
# 
# 
# dbl_class = dbl_19[class_r, class_c]
# vol_class = vol_19[class_r, class_c]
# sgl_class = sgl_19[class_r, class_c]
# HH_class = HH_19[class_r, class_c]
# VV_class = VV_19[class_r, class_c]
# imp_class = imp_19[class_r, class_c]
# 
# samples = list(zip(dbl_class, vol_class, sgl_class, HH_class, VV_class, imp_class))
# 
# # Append to main lists
# X = X + samples
# Y = Y + ([classno] * len(samples))
# print(' ')
# 
# # ----------------------------------------------------------------------------
# print('Gathering samples for class: urban')
# class_r, class_c = np.where( (classes == 5) & (validmask_19 == 1) )
# nth = 1
# class_r = class_r[0::nth]
# class_c = class_c[0::nth]
# 
# classno = 4
# npoints = len(class_r)
# print('There were this many points retained for urban:',npoints)
# 
# 
# dbl_class = dbl_19[class_r, class_c]
# vol_class = vol_19[class_r, class_c]
# sgl_class = sgl_19[class_r, class_c]
# HH_class = HH_19[class_r, class_c]
# VV_class = VV_19[class_r, class_c]
# imp_class = imp_19[class_r, class_c]
# 
# samples = list(zip(dbl_class, vol_class, sgl_class, HH_class, VV_class, imp_class))
# 
# # Append to main lists
# X = X + samples
# Y = Y + ([classno] * len(samples))
# print(' ')
# 
# # ----------------------------------------------------------------------------
# 
# # Now 23rd
# print('Gathering samples for 23-SEP')
# print(' ')
# # Build mask of valid data
# validmask_23 = np.zeros( (rows,cols), 'uint8')
# valid_23 = np.where( (dbl_23 > 0) & (vol_23 > 0) & (sgl_23 > 0) & (HH_23 > 0) & (VV_23 > 0) )
# validmask_23[valid_23] = 1 
# 
# # ---------------------------------------------------------------------------
# print('Gathering samples for class: open water')
# class_r, class_c = np.where( (classes == 1) & (validmask_23 == 1) )
# nth = 1
# class_r = class_r[0::nth]
# class_c = class_c[0::nth]
# 
# classno = 0
# npoints = len(class_r)
# print('There were this many points retained for water:',npoints)
# 
# 
# dbl_class = dbl_23[class_r, class_c]
# vol_class = vol_23[class_r, class_c]
# sgl_class = sgl_23[class_r, class_c]
# HH_class = HH_23[class_r, class_c]
# VV_class = VV_23[class_r, class_c]
# imp_class = imp_23[class_r, class_c]
# 
# samples = list(zip(dbl_class, vol_class, sgl_class, HH_class, VV_class, imp_class))
# 
# # Append to main lists
# X = X + samples
# Y = Y + ([classno] * len(samples))
# print(' ')
# 
# # ----------------------------------------------------------------------------
# print('Gathering samples for class: dry forest')
# class_r, class_c = np.where( (classes == 2) & (validmask_23 == 1) )
# nth = 1
# class_r = class_r[0::nth]
# class_c = class_c[0::nth]
# 
# 
# classno = 1
# npoints = len(class_r)
# print('There were this many points retained for dry forest:',npoints)
# 
# 
# dbl_class = dbl_23[class_r, class_c]
# vol_class = vol_23[class_r, class_c]
# sgl_class = sgl_23[class_r, class_c]
# HH_class = HH_23[class_r, class_c]
# VV_class = VV_23[class_r, class_c]
# imp_class = imp_23[class_r, class_c]
# 
# samples = list(zip(dbl_class, vol_class, sgl_class, HH_class, VV_class, imp_class))
# 
# # Append to main lists
# X = X + samples
# Y = Y + ([classno] * len(samples))
# print(' ')
# 
# # ----------------------------------------------------------------------------
# print('Gathering samples for class: inundated')
# class_r, class_c = np.where( (classes == 3) & (validmask_23 == 1) )
# nth = 1
# class_r = class_r[0::nth]
# class_c = class_c[0::nth]
# 
# classno = 2
# npoints = len(class_r)
# print('There were this many points retained for inundated:',npoints)
# 
# 
# dbl_class = dbl_23[class_r, class_c]
# vol_class = vol_23[class_r, class_c]
# sgl_class = sgl_23[class_r, class_c]
# HH_class = HH_23[class_r, class_c]
# VV_class = VV_23[class_r, class_c]
# imp_class = imp_23[class_r, class_c]
# 
# samples = list(zip(dbl_class, vol_class, sgl_class, HH_class, VV_class, imp_class))
# 
# # Append to main lists
# X = X + samples
# Y = Y + ([classno] * len(samples))
# print(' ')
# 
# # ----------------------------------------------------------------------------
# print('Gathering samples for class: fields/non-forest veg')
# class_r, class_c = np.where( (classes == 4) & (validmask_23 == 1) )
# nth = 1
# class_r = class_r[0::nth]
# class_c = class_c[0::nth]
# 
# classno = 3
# npoints = len(class_r)
# print('There were this many points retained for fields/non-forest veg:',npoints)
# 
# 
# dbl_class = dbl_23[class_r, class_c]
# vol_class = vol_23[class_r, class_c]
# sgl_class = sgl_23[class_r, class_c]
# HH_class = HH_23[class_r, class_c]
# VV_class = VV_23[class_r, class_c]
# imp_class = imp_23[class_r, class_c]
# 
# samples = list(zip(dbl_class, vol_class, sgl_class, HH_class, VV_class, imp_class))
# 
# # Append to main lists
# X = X + samples
# Y = Y + ([classno] * len(samples))
# print(' ')
# 
# # ----------------------------------------------------------------------------
# print('Gathering samples for class: urban')
# class_r, class_c = np.where( (classes == 5) & (validmask_23 == 1) )
# nth = 1
# class_r = class_r[0::nth]
# class_c = class_c[0::nth]
# 
# classno = 4
# npoints = len(class_r)
# print('There were this many points retained for urban:',npoints)
# 
# 
# dbl_class = dbl_23[class_r, class_c]
# vol_class = vol_23[class_r, class_c]
# sgl_class = sgl_23[class_r, class_c]
# HH_class = HH_23[class_r, class_c]
# VV_class = VV_23[class_r, class_c]
# imp_class = imp_23[class_r, class_c]
# 
# samples = list(zip(dbl_class, vol_class, sgl_class, HH_class, VV_class, imp_class))
# 
# # Append to main lists
# X = X + samples
# Y = Y + ([classno] * len(samples))
# print(' ')
# 
# # ----------------------------------------------------------------------------
# 
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=42)
# 
# print('Building SAR classifier!')
# clf = RandomForestClassifier(n_estimators=20, max_depth=10, n_jobs=-1)
# clf.fit(X_train, Y_train)
# print('Classifier trained!')
# 
# =============================================================================


rfc_path = ("C:/Users/amelanco/ProjectData/florence/processed_uavsar/lumber_31509/rfc_31509_copol_imp_19and23.sav")

# Save classifier so we don't have to keep running/overwriting 
#joblib.dump(clf, rfc_path)
print("Classifier ready!")

# Load classifier for use
rfc = joblib.load(rfc_path)

# Get the data you want to build the 'image' for classification
# Remember that scikit-learn here prefers our ravel'd versions of the data
orig_rows, orig_cols = np.shape(dbl_23)
validmask = np.ravel(validmask_23)

valid = np.where(validmask == 1)
invalid = np.where(validmask == 0)
bands = ['DBL','VOL','SGL','HH','VV','IMP']
image = np.zeros( (len(valid[0]), len(bands)), 'float')
print('Number of valid pixels:', len(valid[0]))
print('Number of bands:',len(bands))
image[:,0] = np.ravel(dbl_23)[valid]
image[:,1] = np.ravel(vol_23)[valid]
image[:,2] = np.ravel(sgl_23)[valid]
image[:,3] = np.ravel(HH_23)[valid]
image[:,4] = np.ravel(VV_23)[valid]
image[:,5] = np.ravel(imp_23)[valid]

print('Performing classification:')
ids = rfc.predict(image)
print('Classification complete!')
print('Range of class values:',np.min(ids),np.max(ids))

importance = rfc.feature_importances_
print("Importances: ", importance)

# 'ids' is an array of class numbers that can be used to plot our results
# Recall that class '0' is supposed to be water
# First create a zero array that is 1D and the same set up as the output
ravel_dim = np.shape(validmask)
arr = np.zeros( (ravel_dim), 'float')
# Where the image was valid, put the class number that was predicted... add '1' since default
# array value is 0
arr[valid] = ids+1
# Reshape it into a 2D array for plotting
arr = arr.reshape(rows, cols)


# Let's see what it looks like?
class_labels = [0,1,2,3,4,5]
class_names = ['N/A',  'Open Water','Forest','Inundated Forest', 'Non-Forest Vegetation', 'Urban']
colors =      ['black','blue', 'green', 'darkorange', 'gold', 'red']
cmap = matplotlib.colors.ListedColormap(colors)

plt.figure(figsize=(10,10))
plt.imshow(arr, vmin=0, vmax=5, cmap=cmap)
plt.show()

# Export classified result to .tif
result = dump_geotiff_float("C:/Users/amelanco/ProjectData/florence/processed_uavsar/lumber_31509/923/31509_23_copol_imp_23and23_result.tif", arr, gt)

#=============================================================================
# Begin calculation of accuracy statistics
#=============================================================================

# Read in truth pixels for accuracy assessment
truthpoints_f = "C:/Users/amelanco/ProjectData/florence/processed_uavsar/lumber_31509/923/31509_23_truth_v4.tif"
uavsar_truthpoints = "C:/Users/amelanco/ProjectData/florence/processed_uavsar/lumber_31509/923/uavsar_31509_23_truth_v4.tif"
result = reproject_image_to_reference(vol_23_rpj, truthpoints_f, uavsar_truthpoints)
truths = get_geo(uavsar_truthpoints, band=1)

#plt.imshow(truths)

# Separate array into individual classes to get number of pixels for each
water_loc = np.where(truths == 1)
true_water_px = len(water_loc[1])

dry_forest_loc = np.where(truths == 2)
true_dry_forest_px = len(dry_forest_loc[1])

inun_forest_loc = np.where(truths == 3)
true_inun_forest_px = len(inun_forest_loc[1])

non_forest_loc = np.where(truths == 4)
true_non_forest_px = len(non_forest_loc[1])

urban_loc = np.where(truths == 5)
true_urban_px = len(urban_loc[1])

total_true_px = (true_water_px + true_dry_forest_px + true_inun_forest_px + true_non_forest_px + true_urban_px)

'''
print("Water Truth pixels:", true_water_px)
print("Dry Forest Truth pixels:", true_dry_forest_px)
print("Inundated Forest Truth pixels:", true_inun_forest_px)
print("Non-Forest Truth pixels:", true_non_forest_px)
print("Urban Truth pixels:", true_urban_px)
print("------------")
print("Total Truth pixels:", total_true_px)
print(' ')
'''

classed_img = arr

# CLASS 1: WATER --------------------------------------------------------------

classed_water_loc = np.where((truths > 0) & (classed_img == 1.0))
classed_water = len(classed_water_loc[0])

correct_water_loc = np.where((truths == 1) & (classed_img == 1.0))
correct_water = len(correct_water_loc[0])

# User accuracy: number of correctly classified points / number of points
#                designated as target class by classifier
water_UA = correct_water / classed_water 


# Producer accuracy: number of correctly classified points / number of truth 
#                    points for target class
water_PA = correct_water / true_water_px


# CLASS 2: DRY FOREST ---------------------------------------------------------

classed_forest_loc = np.where((truths > 0) & (classed_img == 2.0))
classed_forest = len(classed_forest_loc[0])

correct_forest_loc = np.where((truths == 2) & (classed_img == 2.0))
correct_forest = len(correct_forest_loc[0])

# User accuracy: number of correctly classified points / number of points
#                designated as target class by classifier
forest_UA = correct_forest / classed_forest 


# Producer accuracy: number of correctly classified points / number of truth 
#                    points for target class
forest_PA = correct_forest / true_dry_forest_px


# CLASS 3: INUNDATED FOREST ---------------------------------------------------

classed_inun_loc = np.where((truths > 0) & (classed_img == 3.0))
classed_inun = len(classed_inun_loc[0])

correct_inun_loc = np.where((truths == 3) & (classed_img == 3.0))
correct_inun = len(correct_inun_loc[0])

# User accuracy: number of correctly classified points / number of points
#                designated as target class by classifier
inun_UA = correct_inun / classed_inun 


# Producer accuracy: number of correctly classified points / number of truth 
#                    points for target class
inun_PA = correct_inun / true_inun_forest_px


# CLASS 4: NON FOREST ---------------------------------------------------------

classed_non_forest_loc = np.where((truths > 0) & (classed_img == 4.0))
classed_non_forest = len(classed_non_forest_loc[0])

correct_non_forest_loc = np.where((truths == 4) & (classed_img == 4.0))
correct_non_forest = len(correct_non_forest_loc[0])

# User accuracy: number of correctly classified points / number of points
#                designated as target class by classifier
non_forest_UA = correct_non_forest / classed_non_forest 


# Producer accuracy: number of correctly classified points / number of truth 
#                    points for target class
non_forest_PA = correct_non_forest / true_non_forest_px


# CLASS 5: URBAN --------------------------------------------------------------

classed_urban_loc = np.where((truths > 0) & (classed_img == 5.0))
classed_urban = len(classed_urban_loc[0])

correct_urban_loc = np.where((truths == 5) & (classed_img == 5.0))
correct_urban = len(correct_urban_loc[0])

# User accuracy: number of correctly classified points / number of points
#                designated as target class by classifier
urban_UA = correct_urban / classed_urban 


# Producer accuracy: number of correctly classified points / number of truth 
#                    points for target class
urban_PA = correct_urban / true_urban_px


# Overall Accuracy: correctly classified points / total points
correct = correct_water + correct_forest + correct_inun + correct_non_forest + correct_urban

overall_acc = correct / total_true_px

#=============================================================================
# Print/display stats
#=============================================================================

print(' ')
print('User Accuracies:')
print('------------------')
print(f"Water: {water_UA:,.4f}")
print(f"Dry Forest: {forest_UA:,.4f}")
print(f"Inundated Forest: {inun_UA:,.4f}")
print(f"Non-Forest: {non_forest_UA:,.4f}")
print(f"Urban: {urban_UA:,.4f}")
print(' ')

print('Producer Accuracies:')
print('------------------')
print(f"Water: {water_PA:,.4f}")
print(f"Dry Forest: {forest_PA:,.4f}")
print(f"Inundated Forest: {inun_PA:,.4f}")
print(f"Non-Forest: {non_forest_PA:,.4f}")
print(f"Urban: {urban_PA:,.4f}")
print(' ')

print(f"Overall Accuracy: {overall_acc:,.4f}")