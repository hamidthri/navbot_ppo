#!/bin/bash
# Entrypoint script to install missing dependencies on first run

# Check if dependencies are already installed
if ! python3 -c "import tensorboard" 2>/dev/null; then
    echo "Installing missing dependencies..."
    pip3 install --no-cache-dir tensorboard==2.11.0
fi

# Execute the main command
exec "$@"
