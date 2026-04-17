#!/bin/bash
#
# Serial helper to run the grounding-line starter example.
#

set -e

currentdir=$(pwd)

mkdir -p OUTPUT_FILES bin
rm -rf OUTPUT_FILES/*

cd "$currentdir/bin"
rm -f xmeshfem2D xspecfem2D
ln -s ../../../bin/xmeshfem2D
ln -s ../../../bin/xspecfem2D
cd "$currentdir"

# Rebuild the receiver list from the editable parameter block in DATA/make_stations.py.
python3 DATA/make_stations.py

cp DATA/Par_file OUTPUT_FILES/
cp DATA/SOURCE OUTPUT_FILES/
cp DATA/STATIONS OUTPUT_FILES/

NPROC=$(grep ^NPROC DATA/Par_file | cut -d = -f 2 | cut -d \# -f 1 | tr -d ' ')

if [ "$NPROC" -eq 1 ]; then
  ./bin/xmeshfem2D
  ./bin/xspecfem2D
else
  mpirun -np "$NPROC" ./bin/xmeshfem2D
  mpirun -np "$NPROC" ./bin/xspecfem2D
fi

cp DATA/*SOURCE* DATA/*STATIONS* OUTPUT_FILES/

echo
echo "Finished. Inspect OUTPUT_FILES/ for seismograms and wavefield snapshots."
