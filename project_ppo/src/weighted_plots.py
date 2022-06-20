import os, glob
import torch
import matplotlib.pylab as plt
from net_actor import NetActor
from net_critic import NetCritic
import sys
import numpy as np
from natsort import natsorted

obs_dim = 16
act_dim = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize actor and critic networks
actor = NetActor(obs_dim, act_dim).to(device)  # ALG STEP 1
critic = NetCritic(obs_dim, 1).to(device)
# critic_path = sorted(glob.glob(f'../../../models/simple_env_10/ppo_critic_*.pth'))
critic_path = natsorted(glob.glob(f'/is/ps2/otaheri/hamid/repos/planner/catkin_ws/src/models/simple_env_10/ppo_critic_*.pth'))
actor_path = natsorted(glob.glob(f'/is/ps2/otaheri/hamid/repos/planner/catkin_ws/src/models/simple_env_10/ppo_actor_*.pth'))

# actor_path = sorted(glob.glob(f'../../../models/simple_env_10/ppo_actor_*.pth'))
if critic_path != []:
    print(f"Loading in {actor_path} and {critic_path}...", flush=True)
    delta_w1_sum_actor = []
    delta_w2_sum_actor = []
    delta_w3_sum_actor = []
    delta_w4_sum_actor = []
    delta_w1_sum_critic = []
    delta_w2_sum_critic = []
    delta_w3_sum_critic = []
    delta_w4_sum_critic = []
    for i in range(0, (len(actor_path) - 1)):
        # load Actor Network
        actor.load_state_dict(torch.load(actor_path[i]))
        #load parameters of Actor
        w1_res1_actor0 = actor.rb1.fc1.weight.data.clone()
        w2_res1_actor0 = actor.rb1.fc2.weight.data.clone()
        w1_res2_actor0 = actor.rb2.fc1.weight.data.clone()
        w2_res2_actor0 = actor.rb2.fc2.weight.data.clone()

        actor.load_state_dict(torch.load(actor_path[i + 1]))

        w1_res1_actor1 = actor.rb1.fc1.weight.data.clone()
        w2_res1_actor1 = actor.rb1.fc2.weight.data.clone()
        w1_res2_actor1 = actor.rb2.fc1.weight.data.clone()
        w2_res2_actor1 = actor.rb2.fc2.weight.data.clone()

        delta_w1_actor = w1_res1_actor1 - w1_res1_actor0
        delta_w2_actor = w2_res1_actor1 - w2_res1_actor0
        delta_w3_actor = w1_res2_actor1 - w1_res2_actor0
        delta_w4_actor = w2_res2_actor1 - w2_res2_actor0

        delta_w1_sum_actor.append((torch.sum(delta_w1_actor).item()))
        delta_w2_sum_actor.append(torch.sum(delta_w2_actor).item())
        delta_w3_sum_actor.append(torch.sum(delta_w3_actor).item())
        delta_w4_sum_actor.append(torch.sum(delta_w4_actor).item())


        w1_res1_critic0 = critic.rb1.fc1.weight.data.clone()
        w2_res1_critic0 = critic.rb1.fc2.weight.data.clone()
        w1_res2_critic0 = critic.rb2.fc1.weight.data.clone()
        w2_res2_critic0 = critic.rb2.fc2.weight.data.clone()

        critic.load_state_dict(torch.load(critic_path[i + 1]))
        w1_res1_critic1 = critic.rb1.fc1.weight.data.clone()
        w2_res1_critic1 = critic.rb1.fc2.weight.data.clone()
        w1_res2_critic1 = critic.rb2.fc1.weight.data.clone()
        w2_res2_critic1 = critic.rb2.fc2.weight.data.clone()

        delta_w1_critic = w1_res1_critic1 - w1_res1_critic0
        delta_w2_critic = w2_res1_critic1 - w2_res1_critic0
        delta_w3_critic = w1_res2_critic1 - w1_res2_critic0
        delta_w4_critic = w2_res2_critic1 - w2_res2_critic0
        delta_w1_sum_critic.append(torch.sum(delta_w1_critic).item())
        delta_w2_sum_critic.append(torch.sum(delta_w1_critic).item())
        delta_w3_sum_critic.append(torch.sum(delta_w1_critic).item())
        delta_w4_sum_critic.append(torch.sum(delta_w1_critic).item())


fig, axs = plt.subplots(2, 2)

axs[0, 0].plot(delta_w1_sum_actor)
axs[0, 0].set_title("deltaw1_Actor", fontsize=10)
axs[0, 1].plot(delta_w2_sum_actor)
axs[0, 1].set_title("deltaw2_Actor", fontsize=10)
axs[1, 0].plot(delta_w3_sum_actor)
axs[1, 0].set_title("deltaw3_Actor", fontsize=10)
axs[1, 1].plot(delta_w4_sum_actor)
axs[1, 1].set_title("deltaw4_Actor", fontsize=10)
# set the spacing between subplots
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)
plt.show()
fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(delta_w1_sum_critic)
axs[0, 0].set_title("deltaw1_Critic", fontsize=10)
axs[0, 1].plot(delta_w2_sum_critic)
axs[0, 1].set_title("deltaw2_Critic", fontsize=10)
axs[1, 0].plot(delta_w3_sum_critic)
axs[1, 0].set_title("deltaw3_Critic", fontsize=10)
axs[1, 1].plot(delta_w4_sum_critic)
axs[1, 1].set_title("deltaw4_Critic", fontsize=10)
# set the spacing between subplots
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)
plt.show()
print('a')