#!/bin/bash

# Script to set up conda configuration for a new user
# This should be run when creating new user accounts

if [ $# -eq 0 ]; then
    echo "Usage: $0 <username>"
    echo "Example: $0 newuser"
    exit 1
fi

USERNAME=$1

echo "Setting up conda configuration for user: $USERNAME"

# Create .conda directory structure
sudo mkdir -p /home/$USERNAME/.conda/envs
sudo chown -R $USERNAME:$USERNAME /home/$USERNAME/.conda

# Create .condarc file
sudo tee /home/$USERNAME/.condarc > /dev/null << EOF
channels:
  - defaults

auto_activate: true

# Ensure environments are created in user's home directory
envs_dirs:
  - /home/$USERNAME/.conda/envs
  - /opt/miniconda3/envs
EOF

# Set proper permissions
sudo chown $USERNAME:$USERNAME /home/$USERNAME/.condarc
sudo chmod 644 /home/$USERNAME/.condarc

echo "✓ Conda configuration set up for $USERNAME"
echo "User should restart their shell or run 'source ~/.bashrc' for changes to take effect."
