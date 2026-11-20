nargo compile --force --benchmark-codegen > codegen.log 2>&1 &
PID=$!

pidstat -r -p "$PID" 1 > memory.log &
MONITOR_PID=$!

wait "$PID"
STATUS=$?

kill "$MONITOR_PID" 2>/dev/null
wait "$MONITOR_PID" 2>/dev/null

exit "$STATUS"