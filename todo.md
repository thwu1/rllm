support passing an agent (how?)

support groupby some arg, and create multiple trajectories.

support emit stepwise reward.

two modes, broadcast or non 

broadcast: each step's reward is the same as the trajectory reward

non broadcast: each step's reward is it's own reward



Setup

websocket use websockets>=15.0.1

ray version 2.48.0


verl:
flash_attn: pip install flash-attn --no-build-isolation
bash scripts/install_verl.sh, and then pip install vllm==0.10.0

make sure when install verl, the torch version is 2.6.0, install vllm will bump it to 2.7.1

add pip install 
pip install 'litellm[proxy]'