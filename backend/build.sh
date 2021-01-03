#!/bin/bash

set -ex
cargo lichking check --all
cargo check --all-targets
cargo clippy
cargo test --release
USE_PREDICTABLE_RNG=1 cargo bench

