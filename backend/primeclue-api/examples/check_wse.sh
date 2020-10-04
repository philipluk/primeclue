#!/usr/bin/env bash

run() {
  date
  cargo run --release --example wse_median_check "../../../test_data/wse/primeclue/$1" 60
}

set -e
run wse_20_1
run wse_30_1
run wse_30_2
run wse_50_2
run wse_50_3
run wse_50_4
run wse_100_4
run wse_100_6
