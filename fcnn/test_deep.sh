#!/usr/bin/env bash
set -euo pipefail

NAME="nn_ssfp"
SCHEME="ultra_honk"

timereal() {
  /usr/bin/time \
    --format=$'TIME: %e s\nMAX RAM: %M KiB' \
    "$@"
}

echo "*****************************************************************"

echo "COMPILE:_________________________________"
timereal nargo compile --silence-warnings --force > /dev/null
echo ""

echo "CIRCUIT INFO:"
nargo info --silence-warnings
echo ""

echo "BARRETENBERG GATES:________________________________________"
timereal bb gates \
  -s "$SCHEME" \
  -b "target/${NAME}.json"
echo ""

echo "EXECUTE (no compilation):___________________________________"
timereal nargo execute --silence-warnings > /dev/null
echo ""

echo "VK GEN:_____________________________________________________"
timereal bb write_vk -b target/${NAME}.json -o target/
echo ""

echo "PROVE:______________________________________________________"
timereal bb prove -b target/${NAME}.json -w target/${NAME}.gz -o target/
echo ""

echo "VERIFY:_____________________________________________________"
timereal bb verify -k target/vk -p target/proof

echo "*****************************************************************"
