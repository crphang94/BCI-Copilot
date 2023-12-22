# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 13:43:57 2023

@author: Phang
"""

import random
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Input
import scipy.io as sio

output_dir = "/predictions_and_weights/"
cat_pprob = sio.loadmat(output_dir + 'Predicted_BCI_Dataset.mat')['cat_pprob']
cat_pred = sio.loadmat(output_dir + 'Predicted_BCI_Dataset.mat')['cat_pred']

class TD3():
    def __init__(self):
        # Input shape
        self.state_dim = 4
        self.action_size = 4
        self.actor_target = self.build_actor()  
        self.critic_target = self.build_critic()
        self.blocker = self.build_blocker()
        
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
        Q_value = model(model_input)
        print(model.summary())
        return Model([state, action], Q_value)
 
action_size = 4
input_dim = 4
td3 = TD3()
td3.actor_target.load_weights(output_dir + "actor_weights.hdf5")
td3.critic_target.load_weights(output_dir + "critic_weights.hdf5")
td3.blocker.load_weights(output_dir + "blocker_weights.hdf5")

game_space_x = 16
game_space_y = 16
max_action = 1
n_step = 9999
agg_pred = []

for sub in range(12):
    random.seed(20230920)
    total_step = 0
    train_cnt = 0  
    gc = 0
    RA = 0
    HA = 0
    BA = 0
    visible = 0
    invisible = 0
    ppp = 0
    eeg_predcm_1 = cat_pred[sub,0]
    eeg_predcm_2 = cat_pred[sub,1]
    eeg_predcm_3 = cat_pred[sub,2]
    eeg_predcm_4 = cat_pred[sub,3]
    
    eeg_predbp_1 = cat_pred[sub,5]
    eeg_predbp_2 = cat_pred[sub,6]
    eeg_predbp_3 = cat_pred[sub,7]
    eeg_predbp_4 = cat_pred[sub,8]
    
    eeg_pprobcm_1 = cat_pprob[sub,0]
    eeg_pprobcm_2 = cat_pprob[sub,1]
    eeg_pprobcm_3 = cat_pprob[sub,2]
    eeg_pprobcm_4 = cat_pprob[sub,3]
    
    eeg_pprobbp_1 = cat_pprob[sub,5]
    eeg_pprobbp_2 = cat_pprob[sub,6]
    eeg_pprobbp_3 = cat_pprob[sub,7]
    eeg_pprobbp_4 = cat_pprob[sub,8]
    
    env_plot = np.zeros((game_space_x,game_space_y))
    fig = plt.figure()
    plot_img = plt.imshow(env_plot, vmin=-1, vmax=1, cmap='jet')
    plt.axis('off')
    
    while total_step<n_step:
        env_plot = np.zeros((game_space_x,game_space_y))
        
        x1 = round(game_space_x/2)
        y1 = round(game_space_y/2)
        game_close = False
        
        targetx = int(random.choice(np.concatenate((range(0, x1), range(x1+1, game_space_x)))))
        targety = int(random.choice(np.concatenate((range(0, y1), range(y1+1, game_space_y)))))

        env_plot[targetx,targety] = -0.25
        env_plot[x1,y1] = 1
        plot_img.set_data(env_plot)
        plt.pause(0.1)
        plt.show()
        
        old_state = [x1/game_space_x,y1/game_space_y,(x1-targetx)/game_space_x,(y1-targety)/game_space_y]

        while not (game_close):
            
            if (random.random()<0.01) and (ppp == 0):
                ppp = 1
                temp_inv_targetx = np.concatenate((range(0, x1), range(x1+1, game_space_x)))
                temp_inv_targety = np.concatenate((range(0, y1), range(y1+1, game_space_y)))
                if targetx!=x1 and targety!=y1:
                    inv_targetx = int(random.choice(np.delete(temp_inv_targetx,np.where(temp_inv_targetx==targetx)[0][0])))
                    inv_targety = int(random.choice(np.delete(temp_inv_targety,np.where(temp_inv_targety==targety)[0][0])))
                else:
                    inv_targetx = int(random.choice(temp_inv_targetx))
                    inv_targety = int(random.choice(temp_inv_targety))                
                
            blocking = False
            if ppp == 0:
                if ((targety - y1) <= 0) & (abs(targety - y1) >= abs(targetx - x1)):
                    side = 2
                elif ((targety - y1) > 0) & (abs(targety - y1) >= abs(targetx - x1)):
                    side = 3
                elif ((targetx - x1) <= 0) & (abs(targetx - x1) >= abs(targety - y1)):
                    side = 0
                elif ((targetx - x1) > 0) & (abs(targetx - x1) >= abs(targety - y1)):
                    side = 1
            elif ppp == 1:
                if ((inv_targety - y1) <= 0) & (abs(inv_targety - y1) >= abs(inv_targetx - x1)):
                    side = 2
                elif ((inv_targety - y1) > 0) & (abs(inv_targety - y1) >= abs(inv_targetx - x1)):
                    side = 3
                elif ((inv_targetx - x1) <= 0) & (abs(inv_targetx - x1) >= abs(inv_targety - y1)):
                    side = 0
                elif ((inv_targetx - x1) > 0) & (abs(inv_targetx - x1) >= abs(inv_targety - y1)):
                    side = 1
            
            RL_action = np.array(td3.actor_target(np.reshape(old_state,(1,4))))[0]
            if side == 0:
                seleeg = random.randint(0,len(eeg_pprobcm_1)-1)
                eeg_action_cm = eeg_pprobcm_1[seleeg,:]
                eeg_action_bp = eeg_pprobbp_1[seleeg,:]
                
            elif side == 1:
                seleeg = random.randint(0,len(eeg_pprobcm_2)-1)
                eeg_action_cm = eeg_pprobcm_2[seleeg,:]
                eeg_action_bp = eeg_pprobbp_2[seleeg,:]
                
            elif side == 2:
                seleeg = random.randint(0,len(eeg_pprobcm_3)-1)
                eeg_action_cm = eeg_pprobcm_3[seleeg,:]
                eeg_action_bp = eeg_pprobbp_3[seleeg,:]
                
            elif side == 3:
                seleeg = random.randint(0,len(eeg_pprobcm_4)-1)
                eeg_action_cm = eeg_pprobcm_4[seleeg,:]
                eeg_action_bp = eeg_pprobbp_4[seleeg,:]
                   
            exp_reward = 0
            if np.argmax(eeg_action_cm) == np.argmax(eeg_action_bp):
                coaction = eeg_action_cm
                HA = HA + 1
            elif np.argmax(eeg_action_cm) == np.argmax(RL_action):
                coaction = eeg_action_cm
                HA = HA + 1
                RA = RA + 1
            elif np.argmax(eeg_action_bp) == np.argmax(RL_action):
                coaction = eeg_action_bp
                HA = HA + 1
            elif (np.argmax(eeg_action_bp) != np.argmax(RL_action)) & (np.argmax(eeg_action_cm) != np.argmax(RL_action)):
                coaction_pprob = eeg_action_bp + eeg_action_cm
                coaction = coaction_pprob
                HA = HA + 1
                
            exp_reward = np.argmax(td3.blocker([np.reshape(old_state,(1,4)), np.reshape(np.argmax(coaction),(1,1))]))
            if exp_reward == 1: 
                coaction = RL_action
                HA = HA - 1
                RA = RA + 1
                BA = BA + 1
                blocking = True
    
            if ppp == 0:
                coaction = RL_action
            sel_action = np.argmax(coaction)
            if sel_action==0:
                x1 = x1 - 1
                direction = 'UP'
            if sel_action==1:
                x1 = x1 + 1
                direction = 'DOWN'
            if sel_action==2:
                y1 = y1 - 1
                direction = 'LEFT'
            if sel_action==3:
                y1 = y1 + 1
                direction = 'RIGHT'
                
            if x1 >= game_space_x or x1 < 0 or y1 >= game_space_y or y1 < 0:
                game_close = True
                gc = gc + 1
            
            if ppp==1:
                if x1 == inv_targetx and y1 == inv_targety:
                    ppp = 0
                    invisible = invisible + 1

            if x1 == targetx and y1 == targety:
                temp_targetx = np.concatenate((range(0, x1), range(x1+1, game_space_x)))
                temp_targety = np.concatenate((range(0, y1), range(y1+1, game_space_y)))
                targetx = int(random.choice(temp_targetx))
                targety = int(random.choice(temp_targety))
                visible += 1
                if ppp == 1:
                    if inv_targetx!=x1 and inv_targety!=y1:
                        targetx = int(random.choice(np.delete(temp_targetx,np.where(temp_targetx==inv_targetx)[0][0])))
                        targety = int(random.choice(np.delete(temp_targety,np.where(temp_targety==inv_targety)[0][0])))
  
            new_state = [x1/game_space_x,y1/game_space_y,(x1-targetx)/game_space_x,(y1-targety)/game_space_y]     
            del old_state
            old_state = np.copy(new_state)
            
            env_plot = np.zeros((game_space_x,game_space_y))
            if ppp==1: env_plot[inv_targetx,inv_targety] = -1 
            # else: env_plot[inv_targetx,inv_targety]=0
            env_plot[targetx,targety] = -0.25
            if game_close == False:
                env_plot[x1,y1] = 1
            plot_img.set_data(env_plot)
            plt.pause(0.1)
            plt.show()
            
            total_step = total_step + 1
           
            del new_state
            if total_step>n_step:
                break
            
    print("sub.: {}, vis.: {}, invis.: {}, over: {}, HA: {}, RA: {}, BA: {}"
          .format(sub+1, visible, invisible, gc, HA/100,RA/100,BA/100))
    
