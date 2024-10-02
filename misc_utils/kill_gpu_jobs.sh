nvidia-smi | grep 'C\s' | awk '{print $5}' | while read pid; do
    if [[ $pid =~ ^[0-9]+$ ]]; then
        echo "Killing PID: $pid"
        kill -9 $pid
    else
        echo "Invalid PID: $pid"
    fi
done
