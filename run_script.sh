#!/bin/bash

# Application path
APP_PATH="/users/btech/spratham/Desktop/ScalablePFET/PFET_nov8/PFET"
DATA_FOLDER="/users/btech/spratham/Desktop/ScalablePFET/PFET_nov8/Vortex_data"

# IPM and HPCX directories
IPM_DIR="/users/btech/spratham/hpcx-v2.13.1-gcc-MLNX_OFED_LINUX-5-ubuntu22.04-cuda11-gdrcopy2-nccl2.12-x86_64/ompi/tests/ipm-2.0.6" # Set the IPM directory
HPCX_IPM_DIR="/users/btech/spratham/hpcx-v2.13.1-gcc-MLNX_OFED_LINUX-5-ubuntu22.04-cuda11-gdrcopy2-nccl2.12-x86_64/ompi/tests/ipm-2.0.6" # Adjust as needed
# Sizes and processor counts
SIZES=(128)
PROCESSES=(8)

find_divisors() {
    local proc=$1
    local cuberoot=$(echo "$proc" | awk '{printf "%d", $1^(1/3)}')
    for ((x=cuberoot; x>0; x--)); do
        if (( proc % x == 0 && x % 2 == 0 )); then  # Check if x is an even divisor
            for ((y=cuberoot; y>0; y--)); do
                if (( (proc / x) % y == 0 && y % 2 == 0 )); then  # Check if y is an even divisor
                    echo $x $y
                    return 0
                fi
            done
        fi
    done
}


# Main loop
for size in "${SIZES[@]}"; do
    CONFIG_FILE_PATH="${APP_PATH}/vorts.config" # Adjust the path to your config files

    for proc in "${PROCESSES[@]}"; do
        read x y <<< $(find_divisors $proc)
        z=$((proc / (x * y)))
        echo $proc
        echo $x
        echo $y
        echo $z
        for ((i=1; i<=5; i++)); do
            echo I am in run $i of $proc
            # Run the application
            mpiexec_path="/users/btech/spratham/hpcx-v2.13.1-gcc-MLNX_OFED_LINUX-5-ubuntu22.04-cuda11-gdrcopy2-nccl2.12-x86_64/ompi/bin/mpiexec -np 4 -x LD_PRELOAD=$IPM_DIR/lib/libipm.so:$IPM_DIR/lib/libipmf.so"
            # Run the application with the specified number of processors and configuration
            $mpiexec_path -np $proc -x LD_PRELOAD=$IPM_DIR/lib/libipm.so:$IPM_DIR/lib/libipmf.so -host csews1:8 --bind-to core:overload-allowed ./pfet $x $y $z $CONFIG_FILE_PATH
            echo mpiexec finished

            sleep 15
            
            # Parse the IPM output
            latest_ipm_xml=$(ls -t *.ipm.xml | head -n 1)
            if [[ -n "$latest_ipm_xml" ]]; then
                $HPCX_IPM_DIR/bin/ipm_parse -html "$latest_ipm_xml"
                
                recent_directory=$(ls -td -- */ | head -n 1)
                directory_name="${DATA_FOLDER}/profiler_reports/${size}_${proc}_${x}_${y}_${z}_run${i}"
                
                # Copy the IPM output to the named directory
                cp -r "$recent_directory" "$directory_name"
                echo copying done
            fi
        done
    done
done