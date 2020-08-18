#!/usr/bin/env bash
set -e

git checkout beta
git merge dev --ff-only
git push