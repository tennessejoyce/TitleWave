#!/bin/bash

#First command line argument is the name of the Stack Exchange forum.

# Make the directory to hold the data for that forum.
if [ ! -d "$1" ]; then
  mkdir $1
fi

cd $1

dataset_dir="dataset"

# Download and extract the dataset from archive.org.
if [ $1 == "overflow" ]; then
  # Special naming convention for StackOverflow, the biggest forum.
  if [ ! -d "$dataset_dir" ]; then
    mkdir "$dataset_dir"
  fi
  cd "$dataset_dir"
  if [ ! -f "Posts.7z" ]; then
    wget -O "Posts.7z" "https://archive.org/download/stackexchange/stackoverflow.com-Posts.7z"
  fi
  if [ ! -d "Posts.7z" ]; then
    7z e "Posts.7z"
  fi
else
  if [ ! -f "$dataset_dir.7z" ]; then
    wget -O "$dataset_dir.7z" "https://archive.org/download/stackexchange/$1.stackexchange.com.7z"
  fi
  if [ ! -d "$dataset_dir" ]; then
    7z e "$dataset_dir.7z" -o"$dataset_dir"
  fi
fi
# Start the MongoDB service
if ! systemctl is-active --quiet mongod; then
  echo "Starting Mongo..."
  sudo systemctl start mongod
fi
