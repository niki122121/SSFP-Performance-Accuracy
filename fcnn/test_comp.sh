#!/usr/bin/env bash
set -euo pipefail

NAME="fcnn"
SCHEME="ultra_honk"

timereal() {
  /usr/bin/time \
    --format=$'TIME: %e s\nMAX RAM: %M KiB' \
    "$@"
}

echo "COMPILE:_________________________________"
timereal nargo compile --silence-warnings --force > /dev/null
echo ""

echo "BARRETENBERG GATES:________________________________________"
timereal bb gates \
  -s "$SCHEME" \
  -b "target/${NAME}.json"
echo ""
