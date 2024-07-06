# Check if sufficient arguments are provided
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <username> <repository-name> <env-var-name-for-pat> <destination-path>"
    exit 1
fi

# Assign command-line arguments to variables
username="$1"
repo_name="$2"
env_var_name="$3"  # Captures the environment variable name for the PAT
destination_path="$4"  # Path where the repository will be cloned

# Ensure the environment variable for PAT is set
if [ -z "${!env_var_name}" ]; then
    echo "Error: Environment variable '${env_var_name}' is not set."
    exit 1
fi

# Configure Git with global settings
git config --global user.email "praful.mohanan@gmail.com"
git config --global user.name "$username"

# Construct the authenticated URL by injecting the PAT directly from the specified environment variable into the repository URL
repo_url="https://github.com/${username}/${repo_name}.git"
auth_url=$(echo "$repo_url" | sed "s|https://|https://${username}:${!env_var_name}@|")

# Ensure the destination directory exists or create it
mkdir -p "$destination_path"

# Full path where the repository will be cloned
full_clone_path="${destination_path}/${repo_name}"

# Clone the repository into the specified directory
if git clone "$auth_url" "$full_clone_path"; then
    echo "Repository cloned successfully at: $full_clone_path"
else
    echo "Failed to clone repository at: $full_clone_path"
    exit 1
fi
