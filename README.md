# DMAC
DMAC:  A Distributional Perspective on Multi-agent Cooperation with Deep Reinforcement Learning

This repository is based on the PRMARL (https://github.com/oxwhirl/pymarl). For the installation instructions, pls refer to PRMARL.

The codes related to DMAC are,

1) Config: /src/config/algs/qr.yaml 

2) Learner: /src/learners/qr_learner.py 

3) Individual Mixing Network: /src/modules/mixers/individual_mixer.py  


To run experiment, for example 3s5z, pls use the following command:

python3 src/main.py --config=qr --env-config=sc2 with env_args.map_name= 3s5z/


