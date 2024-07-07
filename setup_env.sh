# Run `conda init` first then run `source ./ezlife/setup_env.sh`

# 1. add conda to path
export PATH="~/.local/bin/:${PATH}"

# 2. Create a env
poetry config virtualenvs.in-project true
poetry install -E gpu --with gpu --sync