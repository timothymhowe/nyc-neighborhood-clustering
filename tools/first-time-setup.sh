#!/bin/bash
cd `git rev-parse --show-toplevel`
echo "configuring git hooks"
# cp .nyc_gitHooks/pre-commit .git/hooks
chmod a+x .nyc_gitHooks/*
# cp .nyc_gitHooks/*  .git/hooks
git config --local core.hooksPath .nyc_gitHooks
echo "...done!"
echo "Creating conda environment 'NYC-Env' from '.config/environment.yml'..."
conda env create --name NYC-Env --file=.config/environment.yml

echo "Use command 'conda activate NYC-Env' to activate the new environment."
