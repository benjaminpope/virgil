#!/bin/sh
# Log memory use every 10 s while the tests run (tests.yml, macOS): the VM
# page counts in MB, swap, the RSS of every Python process (the xdist
# workers; a framework build is spelled "Python") and the ten largest
# processes by phys_footprint (top's MEM column, which jetsam judges; RSS
# drops as pages are compressed). Evidence for the macOS worker crashes; the
# log is uploaded as an artifact.
ps_bytes=$(sysctl -n hw.pagesize)  # 16384 on Apple Silicon
echo "[mem] hw.pagesize=$ps_bytes hw.memsize=$(sysctl -n hw.memsize)"
while true; do
  echo "[mem] $(date +%T) $(vm_stat | tr -d '.' | awk -F: -v ps="$ps_bytes" '/^Pages (free|active|inactive|wired down|occupied by compressor|speculative)/{gsub(/ /,"",$2); printf "%s=%dMB ", $1, $2*ps/1048576}') pressure=$(sysctl -n kern.memorystatus_vm_pressure_level 2>/dev/null) swap: $(sysctl -n vm.swapusage 2>/dev/null)"
  ps -axo pid,rss,command | awk '/[Pp]ython/ {printf "[mem]   pid %s rss %dMB %s\n", $1, $2/1024, substr($0, index($0,$3), 60)}'
  top -l 1 -n 10 -o mem -stats pid,command,mem,rsize 2>/dev/null | awk '/^ *PID/{f=1} f' | sed 's/^/[mem]   top /'
  sleep 10
done
