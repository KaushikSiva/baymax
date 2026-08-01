# Third-party notices

This repository downloads, but does not redistribute, the following robot
assets at pinned revisions:

- Google DeepMind MuJoCo Menagerie, Unitree G1 model. See the upstream
  `unitree_g1/LICENSE` after running `scripts/download_g1_mjcf.sh`.
- Unitree Robotics `unitree_rl_gym`, including the G1 locomotion policy. See the
  upstream `LICENSE` after running `scripts/setup_unitree_policy.sh`.

The optional hospital, furniture, and patient meshes are prepared from files
supplied by the user in `~/Downloads/hospital_assets`. Those exports are ignored
by Git. Confirm every source asset's license before redistribution.
