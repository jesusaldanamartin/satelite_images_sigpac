#!/bin/bash

# Build docker image
docker build -t tfg:latest .

# Run docker image
docker run tfg

# Copy back to local the results
if [ "$(uname)" == "Darwin" ] || [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    # For Unix-based systems (Linux/MacOS)
    docker cp tfg:/app/data/results/ $HOME/Documents/results/
elif [ "$(expr substr $(uname -s) 1 10)" == "MINGW32_NT" ] || [ "$(expr substr $(uname -s) 1 10)" == "MINGW64_NT" ]; then
    # For Windows using Git Bash or similar
    docker cp tfg:/app/data/results/ /c/Users/$USER/Documents/results/
elif grep -q Microsoft /proc/version; then
    if grep -q "Microsoft" /proc/version && grep -q "WSL2" /proc/version; then
        # For Windows Subsystem for Linux (WSL 2)
        docker cp tfg:/app/data/results/ /mnt/c/Users/$USER/Documents/results/
    else
        # For Windows Subsystem for Linux (WSL 1)
        docker cp tfg:/app/data/results/ /mnt/c/Users/$USER/Documents/results/
    fi
fi
