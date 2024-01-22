#!/usr/local/anaconda3/bin/python


# Converts Freeman-Durden code  output to .tifs  

import sys
import os
import glob
import numpy as np
from osgeo import osr, gdal
from gdalconst import *

# Supporting function: bytescale of imagery from float value to byte value
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

# Read in various UAVSAR information necessary to convert downloaded files to GeoTIFF format...

# Path to the available UAVSAR data
p_uavsar = '/data/amelanco/uavsar/harvey/sabine01200'

# Reference the specific annotation file to help describe the available data
ann = os.path.join(p_uavsar, 'sabine_01200_17088_017_170901_L090_CX_01.ann')

# Get relevant parameters from the annotation file
f = open(ann, 'r')
searchlines = f.readlines()
f.close()

params = {}

for i, line in enumerate(searchlines):
  if line[0] == ';':
    continue

  parts = line.split()
  if 'grd_pwr.set_rows' in line:
    params['rows'] = np.uint16(parts[3])
  if 'grd_pwr.set_cols' in line:
    params['cols'] = np.uint16(parts[3])
  if 'grd_pwr.row_addr' in line:
    params['ul_lat'] = float(parts[3])
  if 'grd_pwr.col_addr' in line:
    params['ul_lon'] = float(parts[3])
  if 'grd_pwr.row_mult' in line:
    params['dy'] = float(parts[3])
  if 'grd_pwr.col_mult' in line:
    params['dx'] = float(parts[3])

  if 'grdHHHH' in line:
    params['grdHHHH'] = os.path.join(p_uavsar, parts[2])
  if 'grdHVHV' in line:
    params['grdHVHV'] = os.path.join(p_uavsar, parts[2])
  if 'grdVVVV' in line:
    params['grdVVVV'] = os.path.join(p_uavsar, parts[2])

  if 'grdHHHV' in line:
    params['grdHHHV'] = os.path.join(p_uavsar, parts[2])
  if 'grdHHVV' in line:
    params['grdHHVV'] = os.path.join(p_uavsar, parts[2])
  if 'grdHVVV' in line:
    params['grdHVVV'] = os.path.join(p_uavsar, parts[2])

#print params['rows']
#print params['cols']
#print params['ul_lat']
#print params['ul_lon']
#print params['dy']
#print params['dx']
#print params['grdHHHH']

rows = params['rows']
cols = params['cols']

# What are the files we expect to be present?  Use dictionary keys to reference them...
# Here, these sections will only handle conversions of the GRD data.
keys = ['grdHHHV','grdHHVV','grdHVVV']
for key in keys:
  with open(params[key], 'rb') as ff:
    data = np.fromfile(ff, dtype=np.complex64)
    arr = np.reshape(data, [rows,cols])
    #print(arr)
    arr_r = arr.real
    arr_i = arr.imag

    intensity = np.sqrt((arr_i*arr_i)+(arr_r*arr_r))
    amplitude = np.sqrt(intensity)
    #print(np.min(amplitude), np.max(amplitude))

    # GeoTIFF conversion
    in_geo = (params['ul_lon'], params['dx'], 0, params['ul_lat'], 0, params['dy'])
    outf = params[key]
    outf = outf.replace('.grd','.tif')
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(outf, \
                       np.int(cols), np.int(rows), 1, GDT_Float32)
    out_cs = osr.SpatialReference()
    out_cs.ImportFromEPSG(4326)
    out_ds.SetProjection(out_cs.ExportToWkt())
    out_ds.SetGeoTransform(in_geo)
    out_ds.GetRasterBand(1).WriteArray(amplitude)
    print(outf)

keys = ['grdHHHH','grdHVHV','grdVVVV']
for key in keys:
  with open(params[key], 'rb') as ff:
    data = np.fromfile(ff, dtype=np.float32)
    arr = np.reshape(data, [rows,cols])

    # GeoTIFF converstion
    in_geo = (params['ul_lon'], params['dx'], 0, params['ul_lat'], 0, params['dy'])
    outf = params[key]
    outf = outf.replace('.grd','.tif')
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(outf, \
                       np.int(cols), np.int(rows), 1, GDT_Float32)
    out_cs = osr.SpatialReference()
    out_cs.ImportFromEPSG(4326)
    out_ds.SetProjection(out_cs.ExportToWkt())
    out_ds.SetGeoTransform(in_geo)
    out_ds.GetRasterBand(1).WriteArray(arr)
    print(outf)
    out_ds = None

# What about handling the Freeman-Durden decomposition files?
# We will assume that they have the same rows/cols and projection, etc. as the source GRD files
# First, let's guess at the file names...
base = os.path.basename(ann).split('.')[0]
keys = ['_vol.bin', '_odd.bin', '_dbl.bin']
for key in keys:
  f = os.path.join(p_uavsar, base+key)
  with open(f, 'rb') as ff:
    data = np.fromfile(ff, dtype=np.float32)
    arr = np.reshape(data, [rows,cols])

    # GeoTIFF converstion
    in_geo = (params['ul_lon'], params['dx'], 0, params['ul_lat'], 0, params['dy'])
    outf = f.replace('.bin','.tif')
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(outf, \
                       np.int(cols), np.int(rows), 1, GDT_Float32)
    out_cs = osr.SpatialReference()
    out_cs.ImportFromEPSG(4326)
    out_ds.SetProjection(out_cs.ExportToWkt())
    out_ds.SetGeoTransform(in_geo)
    out_ds.GetRasterBand(1).WriteArray(arr)
    print(outf)
    out_ds = None

# What about handling the incidence angle and DEM (height) files?
base = os.path.basename(ann).split('.')[0]
f = os.path.join(p_uavsar, base+'.inc')
with open(f, 'rb') as ff:
  data = np.fromfile(ff, dtype=np.float32)
  inc = np.reshape(data, [rows,cols])
  # GeoTIFF converstion
  in_geo = (params['ul_lon'], params['dx'], 0, params['ul_lat'], 0, params['dy'])
  outf = f.replace('.inc','_inc.tif')
  driver = gdal.GetDriverByName('GTiff')
  out_ds = driver.Create(outf, \
                     np.int(cols), np.int(rows), 1, GDT_Float32)
  out_cs = osr.SpatialReference()
  out_cs.ImportFromEPSG(4326)
  out_ds.SetProjection(out_cs.ExportToWkt())
  out_ds.SetGeoTransform(in_geo)
  out_ds.GetRasterBand(1).WriteArray(inc)
  print(outf)
  out_ds = None

base = os.path.basename(ann).split('.')[0]
f = os.path.join(p_uavsar, base+'.hgt')
with open(f, 'rb') as ff:
  data = np.fromfile(ff, dtype=np.float32)
  arr = np.reshape(data, [rows,cols])

  # GeoTIFF converstion
  in_geo = (params['ul_lon'], params['dx'], 0, params['ul_lat'], 0, params['dy'])
  outf = f.replace('.hgt','_hgt.tif')
  driver = gdal.GetDriverByName('GTiff')
  out_ds = driver.Create(outf, \
                     np.int(cols), np.int(rows), 1, GDT_Float32)
  out_cs = osr.SpatialReference()
  out_cs.ImportFromEPSG(4326)
  out_ds.SetProjection(out_cs.ExportToWkt())
  out_ds.SetGeoTransform(in_geo)
  out_ds.GetRasterBand(1).WriteArray(arr)
  print(outf)
  out_ds = None

# Read in the above-generated decomposition components and create a false color composite
# For example, stretch each of the below from 0-0.5 scaled as byte values 0-255
# r = bytescale(dbl, cmin=0, cmax=0.5, low=0, high=255)
# g = bytescale(vol, cmin=0, cmax=0.5, low=0, high=255)
# b = bytescale(odd, cmin=0, cmax=0.5, low=0, high=255)

outf = os.path.join(p_uavsar, base+'_rgb.tif')
driver = gdal.GetDriverByName('GTiff')
out_ds = driver.Create(outf, \
                       np.int(cols), np.int(rows), 3, GDT_Byte)
out_cs = osr.SpatialReference()
out_cs.ImportFromEPSG(4326)
out_ds.SetProjection(out_cs.ExportToWkt())
out_ds.SetGeoTransform(in_geo)

# Here, read in each of the input bands and then scale them/write them back out...
keys = ['_dbl.tif', '_vol.tif', '_odd.tif']
i = 1
for key in keys:
  f = os.path.join(p_uavsar, base+key)
  print(f) 
  ds = gdal.Open(f, GA_ReadOnly)
  cols = ds.RasterXSize
  rows = ds.RasterYSize
  img = ds.GetRasterBand(1).ReadAsArray(0,0,cols,rows)
  print('Range of data for:',f)
  print(np.min(img), np.max(img))
  # set all of the data to scale as 1-255 across values of 0-1
  arr = bytescale(img, cmin=0, cmax=1.0, low=1, high=255)
  # use the previous incidence angle data to find where data are invalid (-10000.)
  missing = np.where(inc == -10000.)
  # set to black before writing (0,0,0)
  arr[missing] = 0
  
  out_ds.GetRasterBand(i).WriteArray(arr)
  i = i + 1
out_ds = None
print(outf)




