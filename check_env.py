from stable_baselines3.common.env_checker import check_env
from hide_and_seek_env import HideAndSeekEnv


env = HideAndSeekEnv()
# It will check your custom environment and output additional warnings if needed
check_env(env)