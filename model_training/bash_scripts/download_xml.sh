#!/bin/bash

# First command line argument is the name of the Stack Exchange forum.
forum="$1"

# Download and extract the dataset from archive.org.
if [ $forum == "overflow" ]; then
  # Special naming convention for StackOverflow, the biggest forum.
  if [ ! -f "Posts.7z" ]; then
    wget -O "Posts.7z" "https://archive.org/download/stackexchange/stackoverflow.com-Posts.7z"
  fi
  if [ ! -d "Posts.7z" ]; then
    7z e "Posts.7z"
  fi
else
  if [ ! -f "$forum.7z" ]; then
    wget -O "$forum.7z" "https://archive.org/download/stackexchange/$forum.stackexchange.com.7z"
  fi
  if [ ! -d "$forum" ]; then
    7z e "$forum.7z" -o"$forum"
  fi
fi

# Start the MongoDB service
if ! systemctl is-active --quiet mongod; then
  echo "Starting Mongo..."
  sudo systemctl start mongod
fi
