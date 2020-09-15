#!/usr/bin/env bash

run() {
  date
  cargo run --release --example gpw_median_check "../../../test_data/gpw/primeclue/$1" 120
}

set -e
run gpw_20_1
run gpw_30_1
run gpw_30_2
run gpw_50_2
run gpw_50_3
run gpw_50_4
run gpw_100_4
run gpw_100_6
run gpw_100_12
run gpw_200_12
run gpw_300_12
