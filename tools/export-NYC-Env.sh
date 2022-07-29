#!/bin/bash
cd `git rev-parse --show-toplevel` 
conda env export -n NYC-Env --no-build | grep -v "^prefix: " > ../.config/environment.yml

echo "Exported environment to '.config/environment.yml'"
