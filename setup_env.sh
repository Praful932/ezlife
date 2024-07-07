# Run `conda init` first then run `source ./ezlife/setup_env.sh`

# 1. add conda to path
export PATH="~/.local/bin/:${PATH}"

# 2. Install deps
poetry config virtualenvs.create false
poetry install
# poetry does not have the no build isolation param
pip install flash-attn==2.5.9.post1 --no-build-isolation
