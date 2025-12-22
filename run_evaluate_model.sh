
# Directory to iterate through
DIR="pruned_weights"

# Python script to call
PYTHON_SCRIPT="evaluate_one_pruned_model.py"

# Check if directory exists
if [ ! -d "$DIR" ]; then
    echo "Error: Directory $DIR does not exist"
    exit 1
fi

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script $PYTHON_SCRIPT does not exist"
    exit 1
fi

# Iterate through files in the directory
for file in "$DIR"/*; do
    # Check if it's a file (not a directory)
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        echo "Processing: $filename"
        python3 "$PYTHON_SCRIPT" "$filename"
        
        # Optional: Check if Python script succeeded
        if [ $? -ne 0 ]; then
            echo "Error processing $file"
        fi
    fi
done

echo "Done processing all files"