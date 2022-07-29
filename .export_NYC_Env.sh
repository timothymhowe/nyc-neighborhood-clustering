#!/bin/bash
conda activate NYC-Env
conda env export --no-build | grep -v "^prefix: " > environments.yml
