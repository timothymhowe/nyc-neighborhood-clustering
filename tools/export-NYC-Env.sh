#!/bin/bash
conda env export -n NYC-Env --no-build | grep -v "^prefix: " > ../.config/environments.yml

echo "Exported environment to '.config/environments.yml'"
