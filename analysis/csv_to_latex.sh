#!/bin/bash

# Check if filename is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <input_csv_file>"
    exit 1
fi

# Input file
INPUT_FILE=$1

# Output file (same name but with .txt extension)
OUTPUT_FILE="${INPUT_FILE%.*}.txt"

# Convert commas to ampersands AND add two backslashes at the end of each line
sed 's/,/\&/g; s/$/\\\\/' "$INPUT_FILE" > "$OUTPUT_FILE"

echo "Conversion complete. Output saved to $OUTPUT_FILE"