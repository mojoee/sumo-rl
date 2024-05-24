import os
import shutil
import subprocess


import numpy as np
import supersuit as ss
import traci
from pyvirtualdisplay.smartdisplay import SmartDisplay
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecMonitor
from tqdm import trange

import sumo_rl


if __name__ == "__main__":
    RESOLUTION = (3200, 1800)

    env = sumo_rl.grid4x4(use_gui=True, out_csv_name="outputs/grid4x4/ppo_test", virtual_display=RESOLUTION)
    
    #max_time = env.sim_max_time
    #delta_time = env.delta_time

    env = ss.pad_observations_v0(env)
    env = ss.pad_action_space_v0(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)

    

    print("Environment created")

    env = ss.concat_vec_envs_v1(env, 2, num_cpus=31, base_class="stable_baselines3")
    env = VecMonitor(env)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=3,
        gamma=0.95,
        n_steps=256,
        ent_coef=0.0905168,
        learning_rate=0.00062211,
        vf_coef=0.042202,
        max_grad_norm=0.9,
        gae_lambda=0.99,
        n_epochs=5,
        clip_range=0.3,
        batch_size=256,
        tensorboard_log="./logs/grid4x4/ppo_test",
    )

    print("Starting training")
    model.learn(total_timesteps=50000)

    print("Training finished. Starting evaluation")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1)

    print(mean_reward)
    print(std_reward)

    #I edit this part

    max_time = 3600
    delta_time = 5
    # Maximum number of steps before reset, +1 because I'm scared of OBOE
    print("Starting rendering")
    num_steps = (max_time // delta_time) + 1

    obs = env.reset()

    #if os.path.exists("temp"):
    #    shutil.rmtree("temp")

    #os.mkdir("temp")
    # img = disp.grab()
    # img.save(f"temp/img0.jpg")

    #img = env.render()
    #for t in trange(num_steps):
    #    actions, _ = model.predict(obs, state=None, deterministic=False)
    #    obs, reward, done, info = env.step(actions)
    #    imgs = env.render()  # This returns a list of images
    #    for agent_index, img in enumerate(imgs):
    #        img.save(f"temp/img{t}_agent{agent_index}.jpg")

    #subprocess.run(["ffmpeg", "-y", "-framerate", "5", "-i", "temp/img%d.jpg", "output.mp4"])

    print("All done, cleaning up")
    shutil.rmtree("temp")
    env.close()
