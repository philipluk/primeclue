#!/usr/bin/env bash

set -e
(cd backend && ./build.sh)
(cd frontend && ./build.sh)