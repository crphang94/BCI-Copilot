
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 13:49:40 2023

@author: Phang
"""

import random
import tensorflow as tf
import keras
import numpy as np
from collections import deque
from keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Input
from tensorflow.keras.optimizers.legacy import Adam
import os
import matplotlib.pyplot as plt
TF_ENABLE_ONEDNN_OPTS=0
output_dir = "/predictions_and_weights/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
 
class TD3():
    def __init__(self):
        self.state_dim = 4
        self.action_size = 4
        self.memory = deque(maxlen=100000)
        self.batch_size=256
        self.gamma=0.99
        self.max_action=1
        self.policy_noise=0.25
        self.noise_clip=0.25
        self.tau = 0.005
        
        self.optimizer_actor = Adam(learning_rate=0.0003)
        self.optimizer_critic = Adam(learning_rate=0.0003)

        self.critic_1 = self.build_critic()
        self.critic_target_1 = self.build_critic()
        self.critic_2 = self.build_critic()
        self.critic_target_2 = self.build_critic()

        self.actor = self.build_actor()  
        self.actor_target = self.build_actor()    
        
    def build_critic(self):
        model = Sequential() 
        model.add(Dense(16))
        model.add(Activation('relu'))
        model.add(Dense(8))
        model.add(Activation('relu'))
        model.add(Dense(2))
        model.add(Activation('linear'))
        
        state = Input(shape=(self.state_dim,))
        action = Input(shape=(self.action_size,))
        model_input = keras.layers.concatenate([state, action])
        Q_value = model(model_input)
        print(model.summary())
        return Model([state, action], Q_value)

    def build_actor(self):
        model = Sequential() 
        model.add(Dense(16))
        model.add(Activation('relu'))
        model.add(Dense(8))
        model.add(Activation('relu'))
        model.add(Dense(self.action_size))
        model.add(Activation('softmax'))
        
        state = Input(shape=(self.state_dim,))
        action = model(state)
        print(model.summary())
        return Model(state, action)
    
    def remember(self, state, action, reward, next_state, done): 
        self.memory.append((state, action, reward, next_state, done))
        
    def train(self):
        minibatch = random.sample(self.memory, self.batch_size)
        state = np.array([i[0] for i in minibatch])
        action = np.array([i[1] for i in minibatch])
        reward = np.array([i[2] for i in minibatch])
        next_state = np.array([i[3] for i in minibatch])
        done = np.array([i[4] for i in minibatch])
        blockage = np.float64(np.multiply(done, 1))
        action = np.squeeze(action)

        # Select action according to policy and add clipped noise 
        noise = np.random.normal(0, self.policy_noise, size=(self.batch_size,self.action_size))
        noise = noise.clip(-self.noise_clip, self.noise_clip)
        next_action = (np.array(self.actor_target(next_state)) + noise).clip(-self.max_action, self.max_action)

        critic_input = np.concatenate((state, action),axis=1)
        target_Q1 = np.array(self.critic_target_1([next_state, next_action]))[:,0]
        target_Q2 = np.array(self.critic_target_2([next_state, next_action]))[:,0]
        target_Q = np.min([target_Q1, target_Q2],axis=0)
        target_R = np.asarray(target_Q)

        for i in range(target_Q.shape[0]):
            if done[i]:
                target_R[i] = reward[i]
            else:
                target_R[i] = reward[i] + (self.gamma * target_Q[i])
        
        target_RB = np.concatenate((target_R[:,np.newaxis], blockage[:,np.newaxis]),axis=1)
        with tf.GradientTape() as tape1:
            q1_values = self.critic_1([critic_input[:,0:self.state_dim],critic_input[:,self.state_dim:self.state_dim+self.action_size]], training=True)
            critic_loss_1 = tf.reduce_mean(tf.math.square(q1_values - target_RB))
        critic_grad_1 = tape1.gradient(critic_loss_1, self.critic_1.trainable_variables)  # compute critic gradient
        self.optimizer_critic.apply_gradients(zip(critic_grad_1, self.critic_1.trainable_variables))
		
        with tf.GradientTape() as tape2:
            q2_values = self.critic_2([critic_input[:,0:self.state_dim],critic_input[:,self.state_dim:self.state_dim+self.action_size]], training=True)
            critic_loss_2 = tf.reduce_mean(tf.math.square(q2_values - target_RB))
        critic_grad_2 = tape2.gradient(critic_loss_2, self.critic_2.trainable_variables)  # compute critic gradient
        self.optimizer_critic.apply_gradients(zip(critic_grad_2, self.critic_2.trainable_variables))
        self.critic_loss = float(min(critic_loss_1,critic_loss_2))
        
        with tf.GradientTape() as tape:
            actions = self.actor(state)
            actor_loss = -tf.reduce_mean(self.critic_target_1([critic_input[:,0:self.state_dim],actions])[:,0])
        actor_grad = tape.gradient(actor_loss,self.actor.trainable_variables)
        self.optimizer_actor.apply_gradients(zip(actor_grad,self.actor.trainable_variables))
        
        return actor_loss, critic_loss_1, critic_loss_2
    
    def model_update(self):
      
        # Update the target models
        weights, weights_t = self.actor.get_weights(), self.actor_target.get_weights()
        for i in range(len(weights)):
        	weights_t[i] = self.tau*weights[i] + (1-self.tau)*weights_t[i]
        self.actor_target.set_weights(weights_t)
        
        weights, weights_t = self.critic_1.get_weights(), self.critic_target_1.get_weights()
        for i in range(len(weights)):
        	weights_t[i] = self.tau*weights[i] + (1-self.tau)*weights_t[i]
        self.critic_target_1.set_weights(weights_t)
        
        weights, weights_t = self.critic_2.get_weights(), self.critic_target_2.get_weights()
        for i in range(len(weights)):
        	weights_t[i] = self.tau*weights[i] + (1-self.tau)*weights_t[i]
        self.critic_target_2.set_weights(weights_t)
    
action_size = 4
input_dim = 4
td3 = TD3()

game_space_x = 16
game_space_y = 16
total_step = 0
e = 0  
epi_reward = []
train_cnt = 0  
actor_loss = 0
max_action = 1
noise = 0.25
n_episodes = 500000000
env_plot = np.zeros((game_space_x,game_space_y))
blocker_loss=0
while e<=n_episodes:
    x1 = round(game_space_x/2)
    y1 = round(game_space_y/2)
    game_close = False
    score = 0

    targetx = int(random.choice(np.concatenate((range(0, x1), range(x1+1, game_space_x)))))
    targety = int(random.choice(np.concatenate((range(0, y1), range(y1+1, game_space_y)))))

    env_plot = np.zeros((game_space_x,game_space_y))
    env_plot[targetx,targety] = -1
    env_plot[x1,y1] = 1
    old_state = [x1/game_space_x,y1/game_space_y,(x1-targetx)/game_space_x,(y1-targety)/game_space_y]
    reward = 0
    dist_1 = pow(pow((x1-targetx)/game_space_x,2) + pow((y1-targety)/game_space_y,2),0.5)
    reward = 0
    while not (game_close):
        action = np.array(td3.actor_target(np.reshape(old_state,(1,4))))[0]
        
        if total_step < 50000:
            action=np.zeros(4)
            rand_cnt=random.randint(0, 3)
            action[rand_cnt]=1
        
        if noise != 0: 
            action = (action + np.random.normal(0, noise, size=action_size))
            action = action * max_action
            action = action.clip(-max_action, +max_action)
            
        sel_action = np.argmax(action)
        if sel_action==0:
            x1 = x1 - 1
            direction = 'UP'
            Head_LR = -1
        if sel_action==1:
            x1 = x1 + 1
            direction = 'DOWN'
            Head_LR = 1
        if sel_action==2:
            y1 = y1 - 1
            direction = 'LEFT'
            Head_UD = 1
        if sel_action==3:
            y1 = y1 + 1
            direction = 'RIGHT'
            Head_UD = -1     
            
        if x1 >= game_space_x or x1 < 0 or y1 >= game_space_y or y1 < 0:
            game_close = True
    
        if x1 != targetx or y1 != targety:        
            dist_2 = pow(pow((x1-targetx)/game_space_x,2) + pow((y1-targety)/game_space_y,2),0.5)
            if dist_2 - dist_1 < 0:
                reward = 0.5
            elif dist_2 - dist_1 > 0:
                reward = -0.5
            else:
                reward = 0
        dist_1 = np.copy(dist_2)
        if x1 >= game_space_x or x1 < 0 or y1 >= game_space_y or y1 < 0:
            game_close = True
            reward = -10
        
        if x1 == targetx and y1 == targety:
            targetx = int(random.choice(np.concatenate((range(0, x1), range(x1+1, game_space_x)))))
            targety = int(random.choice(np.concatenate((range(0, y1), range(y1+1, game_space_y)))))
            score += 1
            reward = 1
            
        new_state = [x1/game_space_x,y1/game_space_y,(x1-targetx)/game_space_x,(y1-targety)/game_space_y]     
        td3.remember(np.array(old_state), action, reward, np.array(new_state), game_close)
        del old_state
        old_state = np.copy(new_state)
        
        env_plot = np.zeros((game_space_x,game_space_y))
        env_plot[targetx,targety] = -1
        if game_close == False:
            env_plot[x1,y1] = 1
        total_step = total_step + 1
        if len(td3.memory) > 50000:
            actor_loss, critic_loss_1, critic_loss_2 = td3.train() 
            train_cnt = train_cnt + 1
            if train_cnt % 2 == 0:
                td3.model_update()
        
        if total_step>=50000 and train_cnt % 10000 == 0:
            td3.actor_target.save_weights(output_dir + "actor_tw" + "{:d}".format(train_cnt) + ".hdf5")
            td3.critic_target_1.save_weights(output_dir + "critic1_tw" + "{:d}".format(train_cnt) + ".hdf5")
            td3.critic_target_2.save_weights(output_dir + "critic2_tw" + "{:d}".format(train_cnt) + ".hdf5")
            
            td3.actor.save_weights(output_dir + "actor_w" + "{:d}".format(train_cnt) + ".hdf5")
            td3.critic_1.save_weights(output_dir + "critic1_w" + "{:d}".format(train_cnt) + ".hdf5")
            td3.critic_2.save_weights(output_dir + "critic2_w" + "{:d}".format(train_cnt) + ".hdf5")
       
        del new_state
            
    print("ep.: {}, total_step： {}， score: {}"
          .format(e, total_step, score))

    epi_reward.append(score)     
    e = e + 1

plt.plot(epi_reward)
plt.show()