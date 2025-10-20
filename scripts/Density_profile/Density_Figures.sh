# List of scripts to run
scripts=("DP_3D_V0.py" "Transfer_common.py" "Slrun_Execute.py" "vdk_data.py")

# Run all scripts with the parameter
for script in "${scripts[@]}"; do
    echo "Running $script with argument $1..."
    output=$(python3 ./$script "$1"  "$2")
    if [ $? -ne 0 ]; then
        echo "$script failed. Exiting."
        exit 1
    fi
    echo "$script output: $output"


done

echo "All scripts executed successfully."
