#!/usr/bin/env bash

set -e

RUSTFLAGS="-C opt-level=3 -C debuginfo=0 -C target-cpu=x86-64"
export RUSTFLAGS
cd ..
cargo build --release --target=x86_64-unknown-linux-gnu
strip target/x86_64-unknown-linux-gnu/release/primeclue-api
cp target/x86_64-unknown-linux-gnu/release/primeclue-api ../docker/
cd ..
cd frontend
npm run build
cp -a dist ../docker/frontend
cd ../docker
docker build --tag primeclue -f Dockerfile .
rm -Rf frontend primeclue-api

#docker tag primeclue lukaszwojtow/primeclue
#docker push lukaszwojtow/primeclue
