#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert Paraview parallel rectilinear grid data (.pvtr) to rectilinear grid 
data (.vtr).  

@author: CHEN Yongxin
"""

import vtk
import sys
import os
import shutil
import numpy as np
from struct import pack
from bs4 import BeautifulSoup
from pathlib import Path

__all__ = ["pvtr2vtr", "write_vtr"]

def main():
    """Read pvd file to convert .pvtr files to .vtr files"""

    args = sys.argv.copy() # get command line arguments
    to_delete = False      # to delete original folders
    root      = '.'        # folder with .pvd file
    pvd       = None       # pvd file
    newpvd    = None       # new pvd file
    pvtrs     = []         # pvtr files
    times     = []         # time steps
    vtrs      = []         # vtr files
    
    i = 1
    while i < len(args):
        arg = args[i]
    
        if arg in ("-h", "--help"):
            print("""
pvtr2vtr - Convert multiple parallel vtr files to a single vtr file in batch.
Note you need to execute pvtr2vtr in the folder where contains the pvd file, or
specify the folder which contains the pvd file.

Calling examples: 
    (1) ./pvtr2vtr -p xxx.pvd -d -n newxxx.pvd         (full, Linux/Mac)
    (2) python pvtr2vtr -p xxx.pvd -d -n newxxx.pvd    (full, All platforms)
    (3) python pvtr2vtr                                (simple, All platforms)
    (4) python pvtr2vtr -d                             (delete old, save space)
    (5) python pvtr2vtr -f folder -p xxx.pvd           (full)

The following command line arguments can be specified:

    -h, --help:       Show this help documentation.
    
    -f, --folder:     Specify the relative path of folder of interest.
        
    -d, --delete:     Delete the original .vtr files from parallel computation.
    
    -p, --pvd:        Name of pvd file.
    
    -n, --new:        New a pvd file with a specified name.
        """)
            return
        elif arg in ("-d", "--delete"):
            to_delete = True
            i += 1
        elif arg in ("-f", "--folder"):
            root = args[i+1]
            i += 2
        elif arg in ("-p", "--pvd"):
            pvd = args[i+1]
            i += 2
        elif arg in ("-n", "--new"):
            newpvd = args[i+1]
            i += 2
        else:
            print("Abort! Unknown flag: "+arg)
            return
    
    # Get pvd file name
    if root != '.': os.chdir(root)
    if pvd is None:
        n = 0
        for file in os.listdir():
            parts = file.split(".")
            if len(parts) > 1:
                ext = parts[-1]
                if ext == 'pvd':
                    pvd = file
                    n += 1
        if n == 0:
            print("No pvd file found in current directory.")
            return
        if n > 1:
            print("Multiple pvd files found. You must specify one via -p/--pvd flag")
            return    
    else:
        if os.path.isfile(pvd) is False:
            print("The speicified pvd file {} not found.".format(pvd))
            return
    
    
    # Get info
    soup = BeautifulSoup(open(pvd), 'xml')
    datasets = soup.find_all('DataSet')
    n = len(datasets)                 # total number of pvtr files
    
    for entity in datasets:
        dataset = BeautifulSoup(str(entity), 'xml')
        pvtrs.append(dataset.DataSet['file'])
        times.append(dataset.DataSet['timestep'])
        
    for pvtr in pvtrs:
        folder = str(Path(pvtr).parent)
        vtrs.append(folder+".vtr")
    
    # Conversion
    print("Total {} pvtr files found. Conversion starts.\n".format(n))
    
    for i, pvtr, vtr in zip(range(n), pvtrs, vtrs):
        print("* Converting {} to {} ...".format(pvtr, vtr))
        pvtr2vtr(pvtr, vtr)
        print("Done. Conversion procedure {}/{} finished.\n".format(i+1, n))
        if to_delete:
            shutil.rmtree(str(Path(pvtr).parent))

    print("Conversion done.\n")
    
    # Write new pvd file
    if newpvd is None: 
        newpvd = ".".join(pvd.split(".")[:-1])+"_collective.pvd"
    
    with open(newpvd, 'w') as fh:
        fh.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
        fh.write('<Collection>\n')
        for time, vtr in zip(times, vtrs):
            fh.write('<DataSet timestep="{}" group="" part="0" file="{}"/>\n'.
                     format(time, vtr))
        fh.write('</Collection>\n')
        fh.write('</VTKFile>')
    
    if to_delete:
        os.remove(pvd)
        
    print("New pvd file {} created.".format(newpvd))
    print("Good day :)")


def pvtr2vtr(pvtr, vtr):
    """
    Convert .pvtr to .vtr.
    
    Parameters
    ----------
    pvtr: str
        Path to .pvtr file.
    vtr: str
        Path to .vtr file.
    """
    # Open file and get data
    reader = vtk.vtkXMLPRectilinearGridReader()
    reader.SetFileName(pvtr)
    reader.Update()
    data = reader.GetOutput()
        
    # Get grid
    x = np.array(data.GetXCoordinates())
    y = np.array(data.GetYCoordinates())
    z = np.array(data.GetZCoordinates())
    
    # Get data pointer
    point_data = data.GetPointData()
    cell_data = data.GetCellData()
        
    # Build output dicts
    point, cell = {}, {}
    for i in range(point_data.GetNumberOfArrays()):
        name = point_data.GetArrayName(i)
        array = np.array(point_data.GetAbstractArray(i))
        if array.ndim > 1:    # e.g velocity: (nx*ny*nz, 3)
            array = array.transpose()
        else:
            array = np.reshape(array, (1, array.size))
        point.update({name: array})
    
    for i in range(cell_data.GetNumberOfArrays()):
        name = cell_data.GetArrayName(i)
        array = np.array(cell_data.GetAbstractArray(i))
        if array.ndim > 1:
            array = array.transpose()
        else:
            array = np.reshape(array, (1, array.size))
        cell.update({name: array})
    
    # write .vtr file
    write_vtr(vtr, x, y, z, point, cell)
    
def write_vtr(name, x, y, z, point, cell):
    """
    Write rectilinear grid .vtr file in binary.

    Parameters
    ----------
    name: str
        File name.
    x, y, z: array-like, float, (N,)
        x, y, z axis 1D grid point.
    point, cell: dict
        Output fields dictionary object of point and cell data.
        Key: field's name.
        Value: numpy array, 4D. e.g. Value = np.zeros((ndim, nx, ny, nz)) or 
        2D array as Value = np.zeros((ndim, nx*ny*nz)). ndim is number of 
        components.
    """
    def encode(string): 
        return str.encode(string)
    
    nx, ny, nz = x.size, y.size, z.size          # dimensions
    off = 0                                      # offset
    ise, jse, kse = [1, nx], [1, ny], [1, nz]    # start and ending indices

    with open(name, 'wb') as fh:
        fh.write(encode( '<VTKFile type="RectilinearGrid" version="0.1" byte_order="LittleEndian">\n'))
        fh.write(encode(f'<RectilinearGrid WholeExtent="{ise[0]} {ise[1]} {jse[0]} {jse[1]} {kse[0]} {kse[1]}">\n'))
        fh.write(encode(f'<Piece Extent="{ise[0]} {ise[1]} {jse[0]} {jse[1]} {kse[0]} {kse[1]}">\n'))
        fh.write(encode( '<Coordinates>\n'))
        fh.write(encode(f'<DataArray type="Float32" Name="x" format="appended" offset="{off}" NumberOfComponents="1"/>\n'))
        off += nx*4 + 4
        fh.write(encode(f'<DataArray type="Float32" Name="y" format="appended" offset="{off}" NumberOfComponents="1"/>\n'))
        off += ny*4 + 4
        fh.write(encode(f'<DataArray type="Float32" Name="z" format="appended" offset="{off}" NumberOfComponents="1"/>\n'))
        off += nz*4 + 4
        fh.write(encode( '</Coordinates>\n'))
        
        # additional info for fields
        if len(point) > 0:
            fh.write(encode('<PointData>\n'))
            for key, value in point.items():
                ndim = value.shape[0]
                fh.write(encode('<DataArray type="Float32" Name="{}" format="appended" offset="{}" NumberOfComponents="{}"/>\n'.
                                format(key, off, ndim)))
                off += value.size*4 + 4
            fh.write(encode('</PointData>\n'))
        
        if len(cell) > 0:
            fh.write(encode('<CellData>\n'))
            for key, value in cell.items():
                ndim = value.shape[0]
                fh.write(encode('<DataArray type="Float32" Name="{}" format="appended" offset="{}" NumberOfComponents="{}"/>\n'.
                                format(key, off, ndim)))
                off += value.size*4 + 4
            fh.write(encode('</CellData>\n'))

        fh.write(encode('</Piece>\n'))
        fh.write(encode('</RectilinearGrid>\n'))
        fh.write(encode('<AppendedData encoding="raw">\n'))
        fh.write(encode('_'))
        fh.write(pack("i",    4*nx))
        fh.write(pack("f"*nx,   *x))
        fh.write(pack("i",    4*ny))
        fh.write(pack("f"*ny,   *y))
        fh.write(pack("i",    4*nz))
        fh.write(pack("f"*nz,   *z))

        # write fields if present
        if len(point) > 0:
            for value in point.values():
                fh.write(pack("i", 4*value.size))
                fh.write(pack("f"*value.size, *(value.flatten(order='F'))))
        
        if len(cell) > 0:
            for value in cell.values():
                fh.write(pack("i", 4*value.size))
                fh.write(pack("f"*value.size, *(value.flatten(order='F'))))
                
        fh.write(encode('\n'))
        fh.write(encode('</AppendedData>\n'))
        fh.write(encode('</VTKFile>'))
        
if __name__ == "__main__":
    main()