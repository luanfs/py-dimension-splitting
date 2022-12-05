#!/bin/bash

# Creates a tarball

date=` date +%F `
version=` date +%y.%m.%d `
echo "Today: " $date

sourcefiles="src/*.py"

parfiles="par/*.par "

scripts="sh/*.sh "

others="main.py\
 README.* \
 backup.sh "

files="$sourcefiles $parfiles $scripts $others"

output="py-dimsplit$version.tar.bz2"

tar cjfv $output $files

echo "File " $output " ready!"
echo

echo "-------------------------------------------"
