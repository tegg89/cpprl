#!/bin/bash
if ! [ -d build ]; then
    mkdir build
fi

cd build
rm -rf ./*

cmake -DCMAKE_PREFIX_PATH=/Users/teggsung/code/libtorch ..
cmake --build . --config Release

echo "Done!"