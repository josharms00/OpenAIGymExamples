import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
import gym
import random
import numpy as np
import argparse

goal_score = 500
req_score = 50
games = 100

class Agent(object):

    def __init__(self, env):
        self.model = self.initialize_model(2)
        self.env = gym.make(env)

    def initial_games(self, env):
        # intialize training data
        train_data = []
        scores = [] 
        actions = []
        correct_action = [1, 0]
        games_list = []

        # iterate through all games
        for _ in range(games):
            score = 0 # score is 0 at beginning of each game
            game_memory = []
            prev_observation = []
            Q = np.zeros((100, 2))
            

            # reset the game before trying to get data from it
            observation = self.env.reset()

            # iterate as many times as it should ideally last
            while 1:
                # move in random direction, cartpole can only go two ways
                action = self.env.action_space.sample()

                if action not in actions:
                    actions.append(action)

                # sample data based on the action just taken
                observation, reward, done, info = self.env.step(action)

                # add to the score
                score += reward

                if done:
                    state = self.state_to_int(prev_observation)
                    Q[state][action] = reward
                    break

                # if the list is not empty append the previous observation with the action that caused it
                if len(prev_observation) > 0:
                    game_memory.append([prev_observation, action, observation, reward, done])
                    state = self.state_to_int(prev_observation)
                    next_state = self.state_to_int(observation)
                    Q[state][action] = reward + 1.0*Q[next_state][np.argmax(Q[next_state])]

                prev_observation = observation

            # game is done
            self.env.close()

            # if score is good enough append to data
            if score >= req_score:
                scores.append(score)

                # outputs need to be one hot encoded
                for data in game_memory:
                    train_data.append([data[0], data[1]])
        
       # train_data = self.replay(games_list, actions)

        train_data = np.array(train_data)

        #np.save('cartpole_training_data.npy', train_data)

        return train_data, actions

    def replay(self, games, actions):
        train_data = []

        Q = np.zeros((100, len(actions)))
    
        for game in games:
            for i in range(len(game)):
                action = game[i][1]
                state = self.state_to_int(game[i][0])
                reward = game[i][3]
                next_state = self.state_to_int(game[i][2])
                done = game[i][4]

                if done:
                    Q[i][action] = reward
                    break

                Q[state][action] = reward + 0.9*Q[next_state][np.argmax(Q[next_state])]

                game[i][1] = np.argmax(Q[state])
                train_data.append([game[i][0], game[i][1]])

        return train_data

    def state_to_int(self, state):
        int_states = []
        env_low = self.env.observation_space.low
        env_high = self.env.observation_space.high
        e = (np.round(env_high, 5) - np.round(env_low, 5)) / 100

        for i in range(len(state)):
            int_states.append(int((state[i] - env_low[i]) / e[i]))

        return int(np.sum(int_states) / len(int_states))

    def initialize_model(self, num_actions):
        # create model
        model = Sequential()

        model.add(Flatten())

        # create feed forward part of neural network
        model.add(Dense(128))
        model.add(Dropout(0.3))
        model.add(Activation('relu'))

        model.add(Dense(256))
        model.add(Dropout(0.3))
        model.add(Activation('relu'))

        model.add(Dense(512))
        model.add(Dropout(0.3))
        model.add(Activation('relu'))

        model.add(Dense(256))
        model.add(Dropout(0.3))
        model.add(Activation('relu'))

        model.add(Dense(128))
        model.add(Dropout(0.3))
        model.add(Activation('relu'))

        model.add(Dense(num_actions))
        model.add(Activation('softmax'))

        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

    def train(self, train_data, envname):
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
        self.model.fit(x, y, epochs=2)

        # save model
        self.model.save(envname + '.h5')

    def test(self, model, games, env):
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

    def investigate_env(self, env):
        actions = []
        d = False
        for _ in range(50):
            score = 0 # score is 0 at beginning of each game
            game_memory = []
            prev_observation = []
            
            # reset the game before trying to get data from it
            observation = env.reset()

            # iterate as many times as it should ideally last
            while 1:
                # move in random direction
                action = env.action_space.sample()

                if action not in actions:
                    actions.append(action)

                # sample data based on the action just taken
                observation, reward, done, info = env.step(action)

                if done:
                    break

            # game is done
            env.close()

        print('Number of actions: ', len(actions))

        for action in actions:
            print('Movement for action ', action)
            env.reset()
            for _ in range(1000):
                env.render()

                if not d:
                    # sample data based on the action just taken
                    observation, reward, done, info = env.step(action)

                if done:
                    d = True
        env.close()

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

    parser.add_argument('-i', '--investigate',
            action="store_true", dest="inv",
            help="Investigate how an environment works")

    args = parser.parse_args()

    env = gym.make(args.env)

    agent = Agent(args.env)

    if args.train:
        data, actions = agent.initial_games(env)
        model = agent.initialize_model(len(actions))
        agent.train(data, args.env)

    if args.test:
        model = tf.keras.models.load_model(args.env + '.h5')
        game = int(args.test)
        agent.test(model, game, env)

    if args.inv:
        investigate_env(env)


if __name__ == '__main__':
    main()