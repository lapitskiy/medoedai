#!/usr/bin/env bash
set -euo pipefail

cd /home/lapitsky/medoedai

docker compose logs -f medoedai | sed -E 's/^([^|]*\| )?[0-9]{4}-[0-9]{2}-[0-9]{2}T[^ ]+Z /\1/'
