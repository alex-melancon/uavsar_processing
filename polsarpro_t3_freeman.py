#!/usr/local/bin/python

import sys
import os
import glob
import numpy as np

# Basic unpolished script that uses polsarpro software to produce Freeman-Durden decomposition
# components.  Data were provided by UAVSAR / JPL portal.

# Required inputs are the *.ann file -- this is the download of the Metadata: Text Annotation File to plain text file
# along with all of the corresponding orthorectified/GRD products.

# Check that all necessary files are decompressed and stored in the directory listed below
# Note that the directory will also be the location of any output files...

# Specify which of the annotation / metadata files will be used, as this contains all of the information
# needed to figure out which other files are expected to be present, their array sizes, etc.

# Create a path to where the source files and outputs will reside...
# This will likely vary by flight track, case, date, etc.

p_uavsar = '/data/amelanco/uavsar/harvey/sabine01200'

# Define a high level path to the PolSARPro v6.0 software
p_polsarpro = '/usr/local/tools/PolSARpro_v6.0/Soft/bin'

# Choose the annotation file (metadata) needed to parse and get information about necessary inputs
# This is specific to each of the flights and should reside in the same directory as above:

ann = os.path.join(p_uavsar, 'sabine_01200_17088_017_170901_L090_CX_01.ann')

# Open access to the text annotation / metadata file and read the information, then close access to the file
f = open( ann, 'r')
searchlines = f.readlines()
f.close()

# Create a dictionary that will go through and store key information searched for in the metadata file, including:
# -- various file names (grdHHHH, etc.)
# -- corresponding header information (hdr)
# -- file names for incidence angle (inc), slope (slope), and DEM (hgt)
# -- dimensions of the arrays contained in the grd files
params = {'grdHHHH':'', \
          'grdHVHV':'', \
          'grdVVVV':'', \
          'grdHHHV':'', \
          'grdHHVV':'', \
          'grdHVVV':'', \
          'hdr':'', \
          'hgt':'', \
          'inc':'', \
          'slope':'', \
          'grd_pwr.set_rows':0, \
          'grd_pwr.set_cols':0}

# Here, each line of the input annotation/metadata file is parsed and searched for keywords, when found,
# those key word details are stored for information.

for i, line in enumerate(searchlines):
  if line[0] == ';':
    continue

  parts = line.split()
  # This block basically says, for each line in the metadata file:
  # if (parameter) exists in the line:
  #   (we found something we want to keep...)
  #   Store that item in its corresponding dictionary entry... params['key'] = value
  if 'grdHHHH' in line:
    params['grdHHHH'] = parts[2]
  if 'grdHVHV' in line:
    params['grdHVHV'] = parts[2]
  if 'grdVVVV' in line:
    params['grdVVVV'] = parts[2]
  if 'grdHHHV' in line:
    params['grdHHHV'] = parts[2]
  if 'grdHHVV' in line:
    params['grdHHVV'] = parts[2]
  if 'grdHVVV' in line:
    params['grdHVVV'] = parts[2]
  if 'hgt' in line:
    params['hgt'] = parts[2]
  if 'grd_pwr.set_rows' in line:
    params['grd_pwr.set_rows'] = np.uint16(parts[3])
  if 'grd_pwr.set_cols' in line:
    params['grd_pwr.set_cols'] = np.uint16(parts[3])

# For some files, the file names are the same as the *.ann, but with a different suffix
# In that case, we don't have to search the metadata file, but instead just replace the
# suffix with a new one

params['ann'] = ann
params['inc'] = ann.replace('.ann', '.inc')
params['hgt'] = ann.replace('.ann', '.hgt')
params['slope'] = ann.replace('.ann', '.slope')
params['hdr'] = ann.replace('.ann', '.hdr')

# Now that key information has been stored in our dictionary (params), we ask that dictionary
# to give us back information that we use to set variables for the rest of the code

# Array dimensions
set_rows = params['grd_pwr.set_rows']
set_cols = params['grd_pwr.set_cols']

# Key file names
vv_vv = params['grdVVVV']
hv_vv = params['grdHVVV']
hv_hv = params['grdHVHV']
hh_vv = params['grdHHVV']
hh_hv = params['grdHHHV']
hh_hh = params['grdHHHH']
ann = params['ann']
inc = params['inc']
hdr = params['hdr']

# For whatever reason, polsarpro prefers to create a specific header file for its own processing.
# Here, we 'build' the command needed to run on the VM that will run 'uavsar_header.exe' for our case

# These executables may change their location based upon the machine on which they are running...

uavsar_header_exe = os.path.join(p_polsarpro, 'data_import', 'uavsar_header.exe')

cmd = [uavsar_header_exe, \
       '-hf', \
       ann, \
       '-id', \
       p_uavsar, \
       '-od', \
       p_uavsar, \
       '-df', \
       'grd', \
       '-tf', \
       hdr]
cmd = " ".join(cmd)
os.system(cmd)

# Similarly, we now set up and run the command that polsarpro needs to convert UAVSAR data 
# Run the 'uavsar_convert_MLC.exe' on a terminal prompt for more information on key words
uavsar_convert_exe = os.path.join(p_polsarpro, 'data_import', 'uavsar_convert_MLC.exe')

t3_cmd = [uavsar_convert_exe, \
          '-hf', os.path.join(p_uavsar, ann), \
          '-if1', os.path.join(p_uavsar, hh_hh), \
          '-if2', os.path.join(p_uavsar, hh_hv), \
          '-if3', os.path.join(p_uavsar, hh_vv), \
          '-if4', os.path.join(p_uavsar, hv_hv), \
          '-if5', os.path.join(p_uavsar, hv_vv), \
          '-if6', os.path.join(p_uavsar, vv_vv), \
          '-od', p_uavsar, \
          '-odf', 'T3', \
          '-inr', ("%d" % set_rows), \
          '-inc', ("%d" % set_cols), \
          '-ofr 0', \
          '-ofc 0', \
          '-fnr', ("%d" % set_rows), \
          '-fnc', ("%d" % set_cols), \
          '-nlr 1', \
          '-nlc 1', \
          '-ssr 1', \
          '-ssc 1']
t3_cmd = " ".join(t3_cmd)
os.system(t3_cmd)

# This command provides conversion of the UAVSAR DEM, though I don't believe it is currently necessary
# and some of the UAVSAR DEM data we have explored have been corrupt
uavsar_convert_exe = os.path.join(p_polsarpro, 'data_import', 'uavsar_convert_dem.exe')

t3_cmd = [uavsar_convert_exe, \
          '-hf', os.path.join(p_uavsar, ann), \
          '-if', inc, \
          '-od', p_uavsar, \
          '-inr', ("%d" % set_rows), \
          '-inc', ("%d" % set_cols), \
          '-ofr 0', \
          '-ofc 0', \
          '-fnr', ("%d" % set_rows), \
          '-fnc', ("%d" % set_cols), \
          '-nlr 1', \
          '-nlc 1', \
          '-ssr 1', \
          '-ssc 1']
t3_cmd = " ".join(t3_cmd)
os.system(t3_cmd)

# Finally, we build the command necessary for the Freeman-Durden decomposition.
# Other decompositions available in data_process_sngl could be performed similarly...
# Check on the meaning of specific key words (nwr, nwc, etc.) by running the command at a prompt
freeman_exe = os.path.join(p_polsarpro, 'data_process_sngl', 'freeman_decomposition.exe')
freeman_cmd = [freeman_exe, \
               '-id', p_uavsar, \
               '-od', p_uavsar, \
               '-iodf T3', \
               '-nwr 5', \
               '-nwc 5', \
               '-ofr 1', \
               '-ofc 1', \
               '-fnr', ("%d" % set_rows), \
               '-fnc', ("%d" % set_cols)]
freeman_cmd = " ".join(freeman_cmd)
os.system(freeman_cmd)

# By default, the outputs of this command are always:
# Freeman_Dbl.bin  dbl_f
# Freeman_Odd.bin  odd_f
# Freeman_Vol.bin  vol_f

# Create desired output names... here, the '*.ann' file name will have the suffix '.ann' replaced 
# with the corresponding component (dbl, odd, and volume)
dbl_f = ann.replace('.ann','_dbl.bin')
odd_f = ann.replace('.ann','_odd.bin')
vol_f = ann.replace('.ann','_vol.bin')

# Using the file names above, move the generic Freeman_Dbl.bin, etc. output names to something more useful:
cmd = ['/bin/mv', \
       os.path.join(p_uavsar, 'Freeman_Dbl.bin'), \
       dbl_f]
cmd = ' '.join(cmd)
os.system(cmd)

cmd = ['/bin/mv', \
       os.path.join(p_uavsar, 'Freeman_Odd.bin'), \
       odd_f]
cmd = ' '.join(cmd)
os.system(cmd)

cmd = ['/bin/mv', \
       os.path.join(p_uavsar, 'Freeman_Vol.bin'), \
       vol_f]
cmd = ' '.join(cmd)
os.system(cmd)

# What files do we need to try and clean up after the fact?
cleanups = ['T11.bin', \
            'T12_real.bin', \
            'T12_imag.bin', \
            'T13_real.bin', \
            'T13_imag.bin', \
            'T22.bin', \
            'T23_real.bin', \
            'T23_imag.bin', \
            'T33.bin', \
            'dem.bin']
for cleanup in cleanups:
  cmd = ['/bin/rm', \
         os.path.join(p_uavsar, cleanup)]
  cmd = ' '.join(cmd)
  os.system(cmd)



          
   
