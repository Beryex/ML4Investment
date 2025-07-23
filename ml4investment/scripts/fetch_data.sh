#!/bin/bash

# --- Script Start ---

# The 'until' loop executes its body repeatedly as long as its command fails
# (exits with a non-zero status). It stops once the command succeeds 
# (exits with 0), which is perfect for our use case.
until python fetch_data.py; do
    # $? is a special shell variable that holds the exit code of the last executed command.
    # A non-zero exit code indicates failure.
    echo "Script failed with exit code $?. Retrying in 5 seconds..."
    
    # Wait for 5 seconds to avoid overwhelming the network or a remote server
    # with rapid retries.
    sleep 5
done

# When the loop finishes, it means 'python fetch_data.py' has executed successfully.
echo "Script executed successfully!"

# --- Script End ---