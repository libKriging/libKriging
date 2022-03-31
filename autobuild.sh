#!/bin/bash

inotifywait -q -m -e close_write $1 |
while read -r filename event; do
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> building... "` date +"%m-%d-%y"`

./mybuild.sh

echo "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
done
