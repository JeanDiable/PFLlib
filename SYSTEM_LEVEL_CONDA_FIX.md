# System-Level Conda Environment Directory Fix

## Problem Solved
Users were experiencing an issue where conda environments were being created in the global `/opt/miniconda3/envs` directory instead of their home directories (`/home/username/.conda/envs`).

Additionally, new users were not automatically entering the conda base environment when logging in.

## Root Cause
The issue was that users didn't have proper `.condarc` configuration files in their home directories, causing conda to use the global configuration and create environments in the system-wide directory.

New users also lacked proper conda initialization in their shell configuration, preventing automatic activation of the base environment.

## System-Level Solution Implemented

### 1. User Template Configuration (`/etc/skel/`)
- **Created**: `/etc/skel/.condarc` - Template conda configuration file
- **Created**: `/etc/skel/.conda/envs/` - Template conda environment directory
- **Updated**: `/etc/skel/.bashrc` - Added conda initialization for automatic base environment activation
- **Effect**: Every new user automatically gets proper conda configuration and base environment activation

### 2. System-Wide Conda Configuration (`/etc/conda/`)
- **Created**: `/etc/conda/condarc` - System-wide conda configuration
- **Purpose**: Provides default settings for all users
- **Priority**: User-specific `.condarc` files override system defaults

### 3. Automatic Setup Script (`/etc/profile.d/`)
- **Created**: `/etc/profile.d/conda-setup.sh` - Automatic conda setup script
- **Function**: Automatically creates conda directories, configuration, and shell initialization for new users
- **Trigger**: Runs when users log in (interactive shells only)

## Configuration Details

### User Template (`.condarc`)
```yaml
channels:
  - defaults

auto_activate: true

# Ensure environments are created in user's home directory
envs_dirs:
  - ~/.conda/envs
  - /opt/miniconda3/envs
```

### User Template (`.bashrc`)
```bash
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/opt/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/opt/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/opt/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/opt/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
```

### System-Wide Configuration (`/etc/conda/condarc`)
```yaml
# System-wide conda configuration
channels:
  - defaults

auto_activate: true

# Environment directories - users can override this in their ~/.condarc
envs_dirs:
  - ~/.conda/envs
  - /opt/miniconda3/envs

# Package cache locations
pkgs_dirs:
  - ~/.conda/pkgs
  - /opt/miniconda3/pkgs
```

### Automatic Setup Script (`/etc/profile.d/conda-setup.sh`)
- Checks if user has `.conda` directory, creates if missing
- Checks if user has `.condarc` file, creates if missing
- Checks if user has conda initialization in `.bashrc`, adds if missing
- Only runs for interactive shells and when conda is available
- Provides user feedback when directories/files are created

## How It Works

### For New Users
1. When a new user is created, files from `/etc/skel/` are copied to their home directory
2. The user automatically gets:
   - `.condarc` file with proper environment directory configuration and `auto_activate: true`
   - `.conda/envs/` directory for storing environments
   - `.bashrc` with conda initialization for automatic base environment activation
3. When the user first logs in, the automatic setup script runs
4. Conda environments are automatically created in `/home/username/.conda/envs/`
5. Users automatically enter the conda base environment when they open a terminal

### For Existing Users
- The previous fix script (`fix_conda_envs.sh`) was already applied
- Users have proper `.condarc` files and `.conda/envs/` directories
- No additional action needed

### Configuration Priority
1. User-specific `.condarc` (highest priority)
2. System-wide `/etc/conda/condarc` (default fallback)
3. Global `/opt/miniconda3/.condarc` (lowest priority)

## Files Created/Modified

### System Files
- `/etc/skel/.condarc` - User template configuration
- `/etc/skel/.conda/envs/` - User template directory
- `/etc/skel/.bashrc` - User template with conda initialization
- `/etc/conda/condarc` - System-wide configuration
- `/etc/profile.d/conda-setup.sh` - Automatic setup script

### User Files (for existing users)
- `/home/username/.condarc` - User-specific configuration
- `/home/username/.conda/envs/` - User environment directory

## Verification

### Test New User Creation
```bash
# Create test user
sudo useradd -m -s /bin/bash testuser

# Verify template files were copied
ls -la /home/testuser/.conda/
cat /home/testuser/.condarc
tail -15 /home/testuser/.bashrc

# Test conda activation (simulate login)
sudo -u testuser bash -c "export PATH=/opt/miniconda3/bin:\$PATH && source ~/.bashrc && echo 'Current conda env: \$CONDA_DEFAULT_ENV'"

# Clean up
sudo userdel -r testuser
```

### Test Environment Creation
```bash
# Create test environment
conda create -n test_env python=3.9 -y

# Verify location
conda env list

# Clean up
conda env remove -n test_env -y
```

## Benefits

1. **Automatic**: No manual configuration needed for new users
2. **Persistent**: Survives system updates and reinstalls
3. **Flexible**: Users can still override settings in their `.condarc`
4. **Backward Compatible**: Existing environments remain accessible
5. **System-Wide**: Works for all users on the system
6. **Auto-Activation**: New users automatically enter conda base environment
7. **Complete Setup**: All necessary conda configuration is automatically provided

## Maintenance

### Adding New Users
- No special steps needed - automatic setup works for all new users
- Users can immediately create conda environments in their home directories
- Users automatically enter the conda base environment when they log in

### System Updates
- Template files in `/etc/skel/` persist through updates
- System-wide configuration in `/etc/conda/` persists through updates
- Automatic setup script in `/etc/profile.d/` persists through updates

### Troubleshooting
- Check user's `.condarc` file: `cat ~/.condarc`
- Check system-wide config: `sudo cat /etc/conda/condarc`
- Verify environment directories: `conda config --show envs_dirs`
- Check conda initialization: `grep -A 15 "conda initialize" ~/.bashrc`
- Test automatic setup: `source /etc/profile.d/conda-setup.sh`
- Verify auto-activation: `echo $CONDA_DEFAULT_ENV`

## Notes
- The existing `pfllib` environment in `/opt/miniconda3/envs/pfllib` remains unchanged
- Users can still access and use existing environments in the global directory
- The fix ensures future environments are created in user home directories
- New users automatically enter the conda base environment when they log in
- No data loss occurred during the fix process
- The solution is permanent and system-wide
