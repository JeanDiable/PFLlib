#!/bin/bash

# Script to fix conda environment directory configuration for all users
# This ensures that conda creates environments in user home directories instead of global directory

echo "Fixing conda environment configuration for all users..."

# Get list of users (excluding system users)
USERS=$(ls /home/ | grep -v lost+found)

for USER in $USERS; do
    echo "Configuring conda for user: $USER"
    
    # Create .conda directory if it doesn't exist
    sudo mkdir -p /home/$USER/.conda/envs
    sudo chown -R $USER:$USER /home/$USER/.conda
    
    # Create or update .condarc file
    sudo tee /home/$USER/.condarc > /dev/null << EOF
channels:
  - defaults

auto_activate: true

# Ensure environments are created in user's home directory
envs_dirs:
  - /home/$USER/.conda/envs
  - /opt/miniconda3/envs
EOF
    
    # Set proper permissions
    sudo chown $USER:$USER /home/$USER/.condarc
    sudo chmod 644 /home/$USER/.condarc
    
    echo "✓ Configured conda for $USER"
done

echo ""
echo "Configuration complete! Now conda will create environments in user home directories."
echo "Users may need to restart their shell or run 'source ~/.bashrc' for changes to take effect."
