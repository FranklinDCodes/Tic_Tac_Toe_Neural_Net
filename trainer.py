import torch as pyt
from torch import nn, optim
from sql_tools import db
from json import loads
from random import choice
from game import Game

# Tic Tac Toe Neural Net object
class T3NN(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = pyt.nn.Sequential(
            nn.Linear(9, 32),
            nn.ReLU(),
            nn.Linear(32, 9)
            )
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return self.net(x)
    
# database
database = db("T3NN_training_data.db")

# 0 == random dataset
# 1 == user dataset
# 2 == defense dataset

# data tables
table0 = database.table["training_data_0"]
table1 = database.table["training_data_user_input_0"]
table2 = database.table["training_data_basics_0"]

# epochs per dataset
epochs_0 = 4
epochs_1 = 500
epochs_2 = 6

# print statement
print_every = 50000

# number of games to test the final net with
trial_count = 10000

# network learning rate
learning_rate = 1e-05

# gather training & test data
training_data_random = [loads(i[0]) for i in table0.getAll()]
training_data_user = [loads(i[0]) for i in table1.getAll()]
training_data_basics = [loads(i[0]) for i in table2.getAll()]
print("Data gathered\n")

# define model
model = T3NN()

# open datasets
datasets = [(epochs_0, training_data_random), (epochs_1, training_data_user), (epochs_2, training_data_basics)]
for epochs, training_data in datasets:

    # open epochs
    for epoch in range(epochs):

        # prepare for model training
        optimizer = optim.Adam(model.parameters(), learning_rate)
        model.train()

        # open data
        for input, label in training_data:

            # forward pass
            output = model(pyt.tensor(input, dtype=pyt.float32))

            # backward pass
            loss = model.loss_fn(output, pyt.tensor(label, dtype=pyt.float32))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
# AI trial
ai_score = {"wins": 0, "wins as X": 0, "wins as O": 0, "losses": 0, "losses as X": 0, "losses as O": 0, 
            "cat's games": 0, "cat's games as X": 0, "cat's games as O": 0}

# start randomized games
for game_num in range(trial_count):

    # init game
    g = Game()

    # if even game number, AI goes first (X)
    if not game_num % 2:

        # while no winner...
        while not g.game_over:

            # get network move
            nn_move_raw = list(model(pyt.tensor(g.get_board_as_input(), dtype=pyt.float32)))

            # re-orient it
            if g.orienting_corner != (0, 0):
                nn_move_raw = g.rotate_board(nn_move_raw, 4 - g.rotation_times)

            # convert it to coordinate tuple
            nn_move_index = nn_move_raw.index(max([i[0] for i in zip(nn_move_raw, range(len(nn_move_raw))) if i[1] not in g.taken_spot_indexes]))
            nn_move = list(g.board.keys())[nn_move_index]

            # take network turn
            g.take_turn(nn_move)

            # check for win
            if g.game_over:
                break
            
            # take random turn for AI opponent
            g.take_turn(choice(g.remaining_moves))

        # adjust ai_score based on game results
        if g.winner == 1:
            ai_score["wins"] += 1
            ai_score["wins as X"] += 1
        elif g.winner == 0:
            ai_score["losses"] += 1
            ai_score["losses as X"] += 1
        else:
            ai_score["cat's games"] += 1
            ai_score["cat's games as X"] += 1
    
    # else AI goes second (O)
    else:

        # while no winner...
        while not g.game_over:
            
            # take random turn for AI opponent
            g.take_turn(choice(g.remaining_moves))

            # check for win
            if g.game_over:
                break
            
            # get network move
            nn_move_raw = list(model(pyt.tensor(g.get_board_as_input(), dtype=pyt.float32)))

            # reorient it
            if g.orienting_corner != (0, 0):
                nn_move_raw = g.rotate_board(nn_move_raw, 4 - g.rotation_times)

            # convert it to coordinate tuple
            nn_move_index = nn_move_raw.index(max([i[0] for i in zip(nn_move_raw, range(len(nn_move_raw))) if i[1] not in g.taken_spot_indexes]))
            nn_move = list(g.board.keys())[nn_move_index]

            # take network turn
            g.take_turn(nn_move)
        
        # update ai_score based on game results
        if g.winner == 0:
            ai_score["wins"] += 1
            ai_score["wins as O"] += 1
        elif g.winner == 1:
            ai_score["losses"] += 1
            ai_score["losses as O"] += 1
        else:
            ai_score["cat's games"] += 1
            ai_score["cat's games as O"] += 1

# print evaluation
print(ai_score)

# save model
pyt.save(model, f"T3NN.pkl")

