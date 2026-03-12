#!/bin/bash
# 04_organize_processed_data.sh - Compile and run file organizer

echo "=== Compiling File Organizer ==="
g++ -o organize_files organize_files.cpp -std=c++17

if [ $? -eq 0 ]; then
    echo "Compilation successful!"
    echo ""
    echo "=== Running File Organizer ==="
    ./organize_files
else
    echo "Compilation failed!"
    exit 1
fi

echo ""
echo "Done! Files organized."
echo ""
echo "Organized structure:"
echo "  processed_data/"
echo "    ├── thickness_maps/    <- From DRIVE + IOSTAR"
echo "    └── orientation_maps/  <- From DRIVE + IOSTAR"
echo "  results/logs/"
echo "    ├── od_png/"
echo "    ├── vessel_png/"
echo "    ├── thickness_png/"
echo "    ├── orientation_png/"
echo "    ├── pixelwise_png/"
echo "    ├── treewise_png/"
echo "    └── topology_png/"
