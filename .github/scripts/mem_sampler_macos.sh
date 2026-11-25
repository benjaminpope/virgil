#!/bin/sh
# Log memory use every 10 s while the tests run (tests.yml, macOS): the VM
# page counts and the RSS of every python process (the xdist workers).
# Evidence for the macOS worker crashes; the log is uploaded as an artifact.
while true; do
  echo "[mem] $(date +%T) $(vm_stat | tr -d '.' | awk -F: '/^Pages (free|active|inactive|wired down|occupied by compressor|speculative)/{gsub(/ /,"",$2); printf "%s=%s ", $1, $2}') pressure=$(sysctl -n kern.memorystatus_vm_pressure_level 2>/dev/null)"
  ps -axo pid,rss,command | awk '/[p]ython/ {printf "[mem]   pid %s rss %dMB\n", $1, $2/1024}'
  sleep 10
done
