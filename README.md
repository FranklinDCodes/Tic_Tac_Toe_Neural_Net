# Tic_Tac_Toe_Neural_Net
A PyTorch neural network trained to play tic-tac-toe, and all the files that helped to create it.

PLAY_ME.py: 
  Run to play the neural network at tic-tac-toe! Type in your moves based on the coordinates of the game board (Column + Row).

T3NN.pkl: 
  Neural network file.

T3NN_training_data.db: 
  The database used to train the network.
  
database_creator.py: 
  The file that generated most of the database by generating random games.

database_user_input.py: 
  Similar to PLAY_ME.py, allows a user to play the network, then saves all the results to the database for training.

game.py: 
  Contains the Game() object used in all of the above code to keep track of turns, decide winners, etc.

trainer.py: 
  The file that trained T3NN.pkl
