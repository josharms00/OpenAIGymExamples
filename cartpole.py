import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
import gym
import random
import numpy as np

env = gym.make('CartPole-v0')

goal_score = 500
req_score = 50
games = 10000

def initial_games():
    # intialize training data
    train_data = []
    scores = [] 

    # iterate through all games
    for _ in range(games):
        score = 0 # score is 0 at beginning of each game
        game_memory = []
        prev_observation = []

        # reset the game before trying to get data from it
        observation = env.reset()

        # iterate as many times as it should ideally last
        for _ in range(goal_score):

            # move in random direction, cartpole can only go two ways
            action = random.randrange(0, 2)

            # sample data based on the action just taken
            observation, reward, done, info = env.step(action)

            # if the list is not empty append the previous observation with the action that caused it
            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action])


            prev_observation = observation

            # add to the score
            score += reward

            if done:
                break

        # game is done
        env.close()

        # if score is good enough append to data
        if score >= req_score:
            scores.append(score)

            # outputs need to be one hot encoded
            for data in game_memory:
                if data[1] == 1:
                    data[1] = [0, 1]
                elif data[1] ==  0:
                    data[1] = [1, 0]

                train_data.append([data[0], data[1]])


    train_data = np.array(train_data)

    np.save('cartpole_training_data.npy', train_data)

    return train_data

def initialize_model():
    # create model
    model = Sequential()

    model.add(Flatten())

    # create feed forward part of neural network
    model.add(Dense(128, input_shape=(2, 1)))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))

    model.add(Dense(256))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))

    model.add(Dense(128))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))

    model.add(Dense(2))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def train(train_data, model):
    x = []
    y = []
    for i in train_data:
        x.append(np.array([i[0]]).reshape(-1, len(train_data[0][0]), 1))
        y.append(np.array([i[1]]))

    x = np.array(x)
    y = np.array(y)

    # train and save model
    model.fit(x, y, epochs=2)

    model.save('CartPole-v0.model')

data = initial_games()
model = initialize_model()
train(data, model)





