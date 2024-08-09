#!/bin/bash

find . -name '*.cu' -exec bash -c 'cp "$0" "${0%.cu}.cpp"' "{}" \;
make -f x86.mk
