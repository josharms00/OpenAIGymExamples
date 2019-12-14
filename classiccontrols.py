import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
import gym
import random
import numpy as np
import argparse

goal_score = 500
req_score = -200
games = 1000

def initial_games(env):
    # intialize training data
    train_data = []
    scores = [] 
    actions = []

    # iterate through all games
    for _ in range(games):
        score = 0 # score is 0 at beginning of each game
        game_memory = []
        prev_observation = []
        

        # reset the game before trying to get data from it
        observation = env.reset()

        # iterate as many times as it should ideally last
        while 1:
            # move in random direction, cartpole can only go two ways
            action = env.action_space.sample()

            if action not in actions:
                actions.append(action)

            # sample data based on the action just taken
            observation, reward, done, info = env.step(action)

            # if the list is not empty append the previous observation with the action that caused it
            if len(prev_observation) > 0:
                s = score + reward
                if s > score:
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
                train_data.append([data[0], data[1]])


    train_data = np.array(train_data)

    np.save('cartpole_training_data.npy', train_data)

    return train_data, actions

def initialize_model(num_actions):
    # create model
    model = Sequential()

    model.add(Flatten())

    # create feed forward part of neural network
    model.add(Dense(128))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))

    model.add(Dense(256))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))

    model.add(Dense(512))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))

    model.add(Dense(256))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))

    model.add(Dense(128))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))

    model.add(Dense(num_actions))
    model.add(Activation('softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def train(train_data, model, envname):
    x = []
    y = []

    # extract labels and data from train_data
    for i in train_data:
        x.append(np.array([i[0]]).reshape(-1, len(train_data[0][0]), 1))
        y.append(np.array([i[1]]))

    # convert lists to numpy arrays
    x = np.array(x)
    y = np.array(y)

    # train and save model
    model.fit(x, y, epochs=3)

    # save model
    model.save(envname + '.h5')

def test(model, games, env):
    prev_observation = []
    scores = []
    
    # play for required amount of games
    for _ in range(games):
        score = 0 
        env.reset()
        while 1:
            env.render()

            # if list is empty there are no actions to predict so movement is random
            if len(prev_observation) == 0:
                action = env.action_space.sample()
            else:
                # predict next action based off previous observation
                action = np.argmax(model.predict(prev_observation.reshape(-1, len(prev_observation), 1))[0])

            # get timestep data based on action
            obver, reward, done, info = env.step(action)
            prev_observation = obver
            score += 1

            if done:
                break

        scores.append(score)

    # print out the highest and average score for the games
    print('Highest score: ', max(scores))
    print('Average score: ', sum(scores)/len(scores))

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-t', '--train',
            action="store_true", dest="train",
            help="train model")

    parser.add_argument('-p', '--test',
            action="store", dest="test",
            help="test model over amount of games")

    parser.add_argument('-e', '--env',
            action="store", dest="env",
            help="choose classic environment to use")

    args = parser.parse_args()

    env = gym.make(args.env)

    if args.train:
        data, actions = initial_games(env)
        model = initialize_model(len(actions))
        train(data, model, args.env)

    if args.test:
        model = tf.keras.models.load_model(args.env + '.h5')
        game = int(args.test)
        test(model, game, env)


if __name__ == '__main__':
    main()