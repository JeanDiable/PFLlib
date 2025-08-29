# Conda Environment Directory Fix

## Problem
Users were experiencing an issue where conda environments were being created in the global `/opt/miniconda3/envs` directory instead of their home directories (`/home/username/.conda/envs`).

## Root Cause
The issue was that users didn't have proper `.condarc` configuration files in their home directories, causing conda to use the global configuration and create environments in the system-wide directory.

## Solution Applied

### 1. Fixed Existing Users
The script `fix_conda_envs.sh` was run to configure conda for all existing users:
- Created `.conda/envs` directories in each user's home directory
- Created proper `.condarc` files with correct environment directory configuration
- Set appropriate ownership and permissions

### 2. Configuration Details
Each user now has a `.condarc` file with:
```yaml
channels:
  - defaults

auto_activate: true

# Ensure environments are created in user's home directory
envs_dirs:
  - /home/username/.conda/envs
  - /opt/miniconda3/envs
```

This configuration ensures that:
- New environments are created in the user's home directory first
- Falls back to global directory if needed
- Maintains compatibility with existing environments

## Files Created

1. **`fix_conda_envs.sh`** - Script to fix conda configuration for all existing users
2. **`setup_new_user_conda.sh`** - Script to set up conda configuration for new users
3. **`CONDA_FIX_README.md`** - This documentation file

## Usage

### For Existing Users
The fix has already been applied. Users should:
1. Restart their shell or run `source ~/.bashrc`
2. Verify the fix by running `conda env list` - new environments should appear in `/home/username/.conda/envs/`

### For New Users
When creating a new user account, run:
```bash
sudo ./setup_new_user_conda.sh <username>
```

## Verification
To verify the fix is working:
```bash
# Check current environment directories
conda config --show envs_dirs

# Create a test environment
conda create -n test_env python=3.9 -y

# List environments (should show new env in home directory)
conda env list

# Clean up test environment
conda env remove -n test_env -y
```

## Notes
- The existing `pfllib` environment in `/opt/miniconda3/envs/pfllib` remains unchanged
- Users can still access and use existing environments in the global directory
- The fix ensures future environments are created in user home directories
- No data loss occurred during the fix process
