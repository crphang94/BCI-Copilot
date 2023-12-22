# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 08:45:04 2023

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
import matplotlib.pyplot as plt
TF_ENABLE_ONEDNN_OPTS=0 

output_dir = "/predictions_and_weights/"

class TD3():
    def __init__(self):
        self.state_dim = 4
        self.action_size = 4
        self.memory = deque(maxlen=100000)
        self.batch_size=256
        self.gamma=0.99
        self.tau = 0.005

        self.optimizer_blocker = Adam(learning_rate=0.0003)
        self.critic_target = self.build_critic()
        self.actor_target = self.build_actor()  
        self.blocker = self.build_blocker()  
        self.blocker_target = self.build_blocker()  
        
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
    
    def build_blocker(self):
        model = Sequential() 
        model.add(Dense(16))
        model.add(Activation('relu'))
        model.add(Dense(8))
        model.add(Activation('relu'))
        model.add(Dense(2))
        model.add(Activation('softmax'))
        
        state = Input(shape=(self.state_dim,))
        action = Input(shape=(1,))
        model_input = keras.layers.concatenate([state, action])
        block_state = model(model_input)
        print(model.summary())
        return Model([state, action], block_state)
    
    def remember(self, state, action, done, target_Q): 
        self.memory.append((state, action, done, target_Q))
        
    def train(self):
        minibatch = random.sample(self.memory, self.batch_size)
        state = np.array([i[0] for i in minibatch])
        action = np.array([i[1] for i in minibatch])
        action = np.argmax(action,1)
        done = np.array([i[2] for i in minibatch])
        action = np.squeeze(action)
        done_onehot=tf.one_hot(done,2)
        
        with tf.GradientTape() as tape:
            b_values = self.blocker([state,action], training=True)
            blocker_loss = tf.reduce_mean(tf.math.square(b_values - done_onehot))
        blocker_grad = tape.gradient(blocker_loss, self.blocker.trainable_variables)  # compute critic gradient
        self.optimizer_blocker.apply_gradients(zip(blocker_grad, self.blocker.trainable_variables))
        
        return blocker_loss
    
    def model_update(self):
        # Update the target models
        weights, weights_t = self.blocker.get_weights(), self.blocker_target.get_weights()
        for i in range(len(weights)):
         	weights_t[i] = self.tau*weights[i] + (1-self.tau)*weights_t[i]
        self.blocker_target.set_weights(weights_t)


 
action_size = 4
input_dim = 4  
game_space_x = 16
game_space_y = 16
target_Q = 0

input_dir = "/predictions_and_weights/"
td3 = TD3()
td3.actor_target.load_weights(input_dir + "actor_tw340000.hdf5")
td3.critic_target.load_weights(input_dir + "critic1_tw340000.hdf5")

for random_generation in range (100000):
    game_close = 0
    y1 = random.randint(0,63)
    x1 = random.randint(0,63)
    foody = int(random.choice(np.concatenate((range(0, y1), range(y1+1, game_space_y)))))
    foodx = int(random.choice(np.concatenate((range(0, x1), range(x1+1, game_space_x)))))
    
    old_state = [x1/game_space_x,y1/game_space_y,(x1-foodx)/game_space_x,(y1-foody)/game_space_y]
    
    pre_action = np.array(td3.actor_target(np.reshape(old_state,(1,4))))[0]
    action = np.zeros(np.size(pre_action))
    sel_action = int(random.choice((0,1,2,3)))
    action[:]=min(pre_action)
    action[sel_action] = max(pre_action) 
    
    if sel_action==0:
        x1 = x1 - 1
    if sel_action==1:
        x1 = x1 + 1
    if sel_action==2:
        y1 = y1 - 1
    if sel_action==3:
        y1 = y1 + 1
    if x1 >= game_space_x or x1 < 0 or y1 >= game_space_y or y1 < 0:
        game_close = 1
    
    td3.remember(np.array(old_state), action, game_close, target_Q)
    del old_state


Blocking_Data = random.sample(td3.memory, len(td3.memory))
done = np.array([i[2] for i in Blocking_Data])
target_Q = np.array([i[3] for i in Blocking_Data])
state = np.array([i[0] for i in Blocking_Data])
action = np.array([i[1] for i in Blocking_Data])
action = np.argmax(action,1)

TQD=target_Q[done==1]
TQND=target_Q[done!=1]
done = np.array([i[2] for i in Blocking_Data])
Blocking_Data_train = []
Blocking_Data_test = []

cnt = 0
for i in range (len(done)):
    if done[i] == 1:
        if cnt >= len(TQD)-1000:
            Blocking_Data_test.append(Blocking_Data[i])  
        else:
            Blocking_Data_train.append(Blocking_Data[i])  
            cnt = cnt + 1

cnt = 0
for i in range (len(done)):
    if done[i] == 0:
        if cnt >= len(TQD)-1000:
            Blocking_Data_test.append(Blocking_Data[i])  
        else:
            Blocking_Data_train.append(Blocking_Data[i])  
        cnt = cnt + 1   
        if cnt == len(TQD):
            break

state_test = np.array([i[0] for i in Blocking_Data_test])
action_test = np.array([i[1] for i in Blocking_Data_test])      
action_test = np.argmax(action_test,1)
done_test = np.array([i[2] for i in Blocking_Data_test]) 
td3.memory = Blocking_Data_train  
blocker_train_agg = [] 
blocker_test_agg = [] 
train_step = 0
n100000 = 0

while n100000 <= 100000:
    blocker_train_loss = td3.train() 
    b_test = np.array(td3.blocker([state_test,action_test]))
    blocker_test_loss = np.mean(np.power(np.argmax(b_test,1) - done_test, 2))
    blocker_test_accuracy = 100*np.sum(np.argmax(b_test,1)==done_test)/len(done_test)

    b_all = np.array(td3.blocker([state,action]))
    blocker_all_loss = np.mean(np.power(np.argmax(b_all,1) - done, 2))
    blocker_all_accuracy = 100*np.sum(np.argmax(b_all,1)==done)/len(done)
    
    print("step: {}, all_loss: {:.2f}, test_loss: {:.2f}, all_acc: {:.2f}, test_acc: {:.2f}"
      .format(train_step,blocker_all_loss,blocker_test_loss,blocker_all_accuracy,blocker_test_accuracy))
    
    blocker_train_agg.append(blocker_train_loss) 
    blocker_test_agg.append(blocker_test_loss) 
    
    if train_step % 10000 == 0:
        td3.blocker.save_weights(output_dir + "blocker_w" + "{:d}".format(train_step) + ".hdf5")
    train_step = train_step + 1    
    if (blocker_test_accuracy==100) & (blocker_all_accuracy==100):
        n100000=n100000+1

td3.blocker.save_weights(output_dir + "blocker_w" + "{:d}".format(train_step) + ".hdf5")
            