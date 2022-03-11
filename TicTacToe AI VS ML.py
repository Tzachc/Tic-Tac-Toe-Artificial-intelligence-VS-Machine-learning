import math
from random import randint
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt
import sys, os

class TicTacToe:

    def __init__(self, size, players):
        self.size = size
        self.players = players
        self.board = [[' ' for j in range(size)] for i in range(size)]
        self.moves = []
        self.winner = None
        self.newData = []

    """
    Before executing move, validMove method will be called from the client side just to check if the move is valid.
    If the move is valid, the move method will be called.
    The sign will be planted in the board (based on given row and col)
    Then the checkWinner method will be called.
    Last thing we need to check is if the match is a draw!

    """

    def move(self, r, c, player):
        if not self.isValidMove(r, c):
            return -1

        sign = self.players[player - 1]
        self.board[r][c] = sign

        # self.generateNewRow()
        self.moves.append([r, c])

        # self.displayBoard()

        if self.checkWinner(r, c, player):
            # print("Player ", player, " wins!")
            return 1

        if self.checkDraw():
            # print("Draw!")
            return 0

        # print("Next move please!")
        return 2

    """
    Check if the game is a draw. Check if the board is full.
    """

    def checkDraw(self):
        status = len(self.moves) == self.size * self.size
        if status:
            self.setWinner(0)

        return status

    """
    Check if the player is the winner after the last move.
    Check-
    * if all the rows contain same sign OR
    * if all the columns contain same sign OR
    * if all diagonal or anti-diagonal boxes contain same sign
    """

    def checkWinner(self, r, c, player):
        status = self.checkRow(r, player) or self.checkCol(c, player) or self.checkDiagonals(player)
        if status:
            self.setWinner(player)
        return status

    """
    Check if all the rows contain same sign

    """

    def checkRow(self, r, player):
        for i in range(self.size):
            if not self.board[r][i] == self.players[player - 1]:
                return False

        # print("Row true")
        return True

    """
    Check if all the columns contain same sign

    """

    def checkCol(self, c, player):
        for i in range(self.size):
            if not self.board[i][c] == self.players[player - 1]:
                return False

        # print("Col true")
        return True

    """
    Check if all diagonal or anti-diagonal boxes contain same sign

    """

    def checkDiagonals(self, player):
        status1 = True
        status2 = True
        for i in range(self.size):
            if not self.board[i][i] == self.players[player - 1]:
                status1 = False  # checking diagonal

        r = 0
        c = self.size - 1
        while r < self.size and c >= 0:
            if not self.board[r][c] == self.players[player - 1]:
                status2 = False  # checking anti-diagonal
            r += 1
            c -= 1

        return status1 or status2

    """
    Set winner once the match is done. 
    winner = 1 for player 1, 2 for player 2, 0 for draw
    """

    def setWinner(self, winner):
        self.winner = winner

    """
    Return the winner.
    winner = 1 for player 1, 2 for player 2, 0 for draw
    """

    def getWinner(self):
        return self.winner

    """
    Check if the move is valid.
    In other word, check if the given position (row, col) in the board is empty

    """

    def isValidMove(self, r, c):
        return not (self.board[r][c] == self.players[0] or self.board[r][c] == self.players[1])

    """
    Undo last move!
    """

    def undo(self):
        if len(self.moves) > 0:
            lastMove = self.moves.pop(-1)
            self.board[lastMove[0]][lastMove[1]] = ' '
            self.setWinner(None)
            # self.displayBoard()

    """
    Display the TicTacToe Board 
    """

    def displayBoard(self):

        seperator = []
        for i in range(self.size - 1):
            seperator.append('-')
            seperator.append('+')
        seperator.append('-')

        print()
        for row, val in enumerate(self.board):
            print(*val, sep=" | ")
            if not row == self.size - 1:
                print(*seperator)
        print()

    """
    Add new row to temporary dataset (newData) after every move. 
    This is using to generate new data so that we can train our machine learning model with new data after each game.

    This function creates two rows for each move considering both of the players as winner 

    """

    def generateNewRow(self):
        newRow = []
        for row in range(self.size):
            for col in range(self.size):
                val = 0
                if (self.board[row][col] == self.players[0]):
                    val = 1
                elif (self.board[row][col] == self.players[1]):
                    val = -1

                newRow.append(val)

        newInvertRow = [v if v == 0 else -1 if v == 1 else 1 for v in newRow]

        self.newData.append(newRow)
        self.newData.append(newInvertRow)

        # print (self.newData)

    """
    getNewData will be called at the end of each match. 
    This function labels all the data and return a new set of dataset so that we can train our ML model with this new set of data

    """

    def getNewData(self, winner):
        if (winner == 1):
            newTrainY = [1 if i % 2 == 0 else 2 for i in range(len(self.newData))]
        elif (winner == 2):
            newTrainY = [2 if i % 2 == 0 else 1 for i in range(len(self.newData))]
        else:
            newTrainY = [0 for i in range(len(self.newData))]

        newTrainX = self.newData
        self.newData = []

        print("Size of newTrainX and newTrainY: ", len(newTrainX), len(newTrainY))
        # print(newTrainX)
        # print(newTrainY)

        return newTrainX, newTrainY

    """
    Reset the variables to make the board ready the next match!
    """

    def clear(self, size, players):
        self.size = size
        self.players = players
        self.board = [[' ' for j in range(size)] for i in range(size)]
        self.moves = []
        self.winner = None


class TicTacToeAI(TicTacToe):

    def __init__(self, size, players):
        super().__init__(size, players)

    def allPossibleNextMoves(self):
        possibleMoves = []

        for row in range(self.size):
            for col in range(self.size):
                if (self.isValidMove(row, col)):
                    possibleMoves.append([row, col])

        return possibleMoves

class AIAlgorithms:
    def __init__(self, game):
        self.game = game
        self.dataset = []


class MiniMaxAlgorithm(AIAlgorithms):

    def __init__(self, game):
        super().__init__(game)

    """
    Thanks to wiki for psudo code.
    """
    """
    Iterate through all the possible moves and call minimax each time to find the best possible move
    """

    def findBestMiniMaxMove(self, player):
        bestScore = -math.inf
        bestMove = None
        counter = [0]

        for possibleMove in self.game.allPossibleNextMoves():
            self.game.move(possibleMove[0], possibleMove[1], player)
            score = self.minimax(False, player, 0, counter)
            self.game.undo()

            if (score > bestScore):
                bestScore = score
                bestMove = possibleMove

        return bestMove

    """
    Return Max Score and Min Score respectively for Maximizing and Minimizing player.
    """

    def minimax(self, isMax, player, depth, counter):

        counter[0] = counter[0] + 1

        winner = self.game.getWinner()
        if (not (winner == None)):
            if (winner == 0):
                return 0
            elif (winner == player):
                return 10 - depth
            else:
                return depth - 10

        maxScore = -math.inf
        minScore = math.inf

        for possibleMove in self.game.allPossibleNextMoves():
            currPlayer = player if isMax else 2 if (player == 1) else 1

            self.game.move(possibleMove[0], possibleMove[1], currPlayer)
            score = self.minimax(not isMax, player, depth + 1, counter)
            self.game.undo()

            if (score > maxScore):
                maxScore = score
            if (score < minScore):
                minScore = score

        return maxScore if isMax else minScore


class LogisticRegressionAlgorithm(AIAlgorithms):

    def __init__(self, game):
        super().__init__(game)
        self.LRModel = None
        self.trainX = []
        self.trainY = []

    """
    Iterate through all the possible moves, generate new test data and call logistic regression algorithm to find the best possible move based on the probability of winning.
    """

    def findBestLogisticMove(self, player):
        testX = []
        positions = []

        for possibleMove in self.game.allPossibleNextMoves():
            self.game.move(possibleMove[0], possibleMove[1], player)
            positions.append(possibleMove)
            testX.append(self.generateTestX())
            self.game.undo()

        index = 1 if player == 1 else 2

        predictions = np.around(self.logisticRegressionTesting(testX), decimals=2)

        maxProb = np.amax(predictions[:, index])
        moveIndex = np.where(predictions[:, index] == maxProb)[0][0]
        """
        print (predictions)
        print (self.LRModel.classes_)
        print (index)
        print (maxProb)
        print (moveIndex)
        """
        return positions[moveIndex]

    """
    Generating testing data based on all possible moves.
    """

    def generateTestX(self):
        newRow = []
        for row in range(self.game.size):
            for col in range(self.game.size):
                val = 0
                if (self.game.board[row][col] == self.game.players[0]):
                    val = 1
                elif (self.game.board[row][col] == self.game.players[1]):
                    val = -1

                newRow.append(val)
        return newRow

    """
    Train your logistic Regression model with the new and old dataset
    """

    def logisticRegressionTraining(self):
        dataset = np.concatenate((np.asarray(self.trainX), np.asarray([self.trainY]).T), axis=1)
        np.random.shuffle(dataset)

        X = dataset[:, :-1]
        y = dataset[:, -1]
        self.LRModel = LogisticRegression(random_state=0).fit(X, y)

    """
    Test your logistic Regression model with all possible moves and find the best move based on the probability of winning.
    """

    def logisticRegressionTesting(self, testX):
        return self.LRModel.predict_proba(np.asarray(testX))

    def learningTime(self):
        print("Logictic Regression: It's learning time!")
        self.logisticRegressionTraining()


    def addNewData(self, newTrainX, newTrainY):

        for i in range(len(newTrainX)):
            self.trainX.append(newTrainX[i])
            self.trainY.append(newTrainY[i])

        print("Training size: ", len(self.trainX))
        # self.learningTime()

#############################################################NEW######################################################

class KNNAlgorithm(AIAlgorithms):

    def __init__(self, game):
        super().__init__(game)
        self.KNNModel = None
        self.trainX = []
        self.trainY = []

    """
    Iterate through all the possible moves, generate new test data and call KNN algorithm to find the best possible move based on the probability of winning.
    """

    def findBestKNNMove(self, player):
        testX = []
        positions = []

        for possibleMove in self.game.allPossibleNextMoves():
            self.game.move(possibleMove[0], possibleMove[1], player)
            positions.append(possibleMove)
            testX.append(self.generateTestX())
            self.game.undo()

        index = 1 if player == 1 else 2

        predictions = np.around(self.KNNTesting(testX), decimals=2)

        maxProb = np.amax(predictions[:, index])
        moveIndex = np.where(predictions[:, index] == maxProb)[0][0]

        return positions[moveIndex]

    """
    Generating testing data based on all possible moves.
    """

    def generateTestX(self):
        newRow = []
        for row in range(self.game.size):
            for col in range(self.game.size):
                val = 0
                if (self.game.board[row][col] == self.game.players[0]):
                    val = 1
                elif (self.game.board[row][col] == self.game.players[1]):
                    val = -1

                newRow.append(val)
        return newRow

    """
    Train your KNN model with the new and old dataset
    """

    def KNNTraining(self):
        dataset = np.concatenate((np.asarray(self.trainX), np.asarray([self.trainY]).T), axis=1)
        np.random.shuffle(dataset)

        X = dataset[:, :-1]
        y = dataset[:, -1]
        self.KNNModel = KNeighborsClassifier().fit(X, y)

    """
    Test your KNN model with all possible moves and find the best move based on the probability of winning.
    """

    def KNNTesting(self, testX):
        return self.KNNModel.predict_proba(np.asarray(testX))

    def learningTime(self):
        print("KNN: Hey! It's learning time. Running towards being a human!")
        self.KNNTraining()
        print("I have learnt lots of new tricks!")

    def addNewData(self, newTrainX, newTrainY):

        for i in range(len(newTrainX)):
            self.trainX.append(newTrainX[i])
            self.trainY.append(newTrainY[i])

        print("Training size: ", len(self.trainX))
        # self.learningTime()

#############################################################NEW######################################################


#############################################################NEW_DecisionTree######################################################

class DecisionTreeAlgorithm(AIAlgorithms):

    def __init__(self, game):
        super().__init__(game)
        self.DecisionTreeModel = None
        self.trainX = []
        self.trainY = []

    """
    Iterate through all the possible moves, generate new test data and call DecisionTree algorithm to find the best possible move based on the probability of winning.
    """

    def findBestDecisionTreeMove(self, player):
        testX = []
        positions = []

        for possibleMove in self.game.allPossibleNextMoves():
            self.game.move(possibleMove[0], possibleMove[1], player)
            positions.append(possibleMove)
            testX.append(self.generateTestX())
            self.game.undo()

        index = 1 if player == 1 else 2

        predictions = np.around(self.DecisionTreeTesting(testX), decimals=2)

        maxProb = np.amax(predictions[:, index])
        moveIndex = np.where(predictions[:, index] == maxProb)[0][0]

        return positions[moveIndex]

    """
    Generating testing data based on all possible moves.
    """

    def generateTestX(self):
        newRow = []
        for row in range(self.game.size):
            for col in range(self.game.size):
                val = 0
                if (self.game.board[row][col] == self.game.players[0]):
                    val = 1
                elif (self.game.board[row][col] == self.game.players[1]):
                    val = -1

                newRow.append(val)
        return newRow

    """
    Train your DecisionTree model with the new and old dataset
    """

    def DecisionTreeTraining(self):
        dataset = np.concatenate((np.asarray(self.trainX), np.asarray([self.trainY]).T), axis=1)
        np.random.shuffle(dataset)

        X = dataset[:, :-1]
        y = dataset[:, -1]
        self.DecisionTreeModel = DecisionTreeClassifier().fit(X, y)

    """
    Test your DecisionTree model with all possible moves and find the best move based on the probability of winning.
    """

    def DecisionTreeTesting(self, testX):
        return self.DecisionTreeModel.predict_proba(np.asarray(testX))

    def learningTime(self):
        print("DecisionTree: Hey! It's learning time.")
        self.DecisionTreeTraining()
        print("I have learnt lots of new tricks!")

    def addNewData(self, newTrainX, newTrainY):

        for i in range(len(newTrainX)):
            self.trainX.append(newTrainX[i])
            self.trainY.append(newTrainY[i])

        print("Training size: ", len(self.trainX))
        # self.learningTime()

#############################################################NEW######################################################


class RandomPlacementAlgorithm(AIAlgorithms):

    def __init__(self, game):
        super().__init__(game)
        self.LRModel = None

    def randomPlacement(self):
        possibleMoves = self.game.allPossibleNextMoves()
        index = randint(0, len(possibleMoves) - 1)
        return [possibleMoves[index][0], possibleMoves[index][1]]


def run_Logistic():
    player = randint(1, 2)
    print("Good Luck!")

    game = TicTacToeAI(3, ['X', 'O'])
    #     game.displayBoard()

    miniMaxAlgorithm = MiniMaxAlgorithm(game)
    logisticRegressionAlgorithm = LogisticRegressionAlgorithm(game)
    randomPlacementAlgorithm = RandomPlacementAlgorithm(game)

    print("Generating data, Please give me a moment :)")
    #
    generateData(player, game, miniMaxAlgorithm, logisticRegressionAlgorithm, randomPlacementAlgorithm)
    #
    print("Data generated!")

    logisticRegressionAlgorithm.learningTime()

    print("======================== IT'S BATTLE TIME =================================")

    mmCount = 0
    lrCount = 0
    drawCount = 0
    #Numbers of rounds
    for i in range(5):
        print("Battle number: ", i + 1)
        print("==============================================================")
        print("Minimax: ", mmCount, "Logistic Regression: ", lrCount, "Draw: ", drawCount)
        print("==============================================================")
        game.clear(3, ['X', 'O'])

        while (True):

            result = None
            #player is minmax so we check his best move and then make the move.
            if player == 1:
                print("It's MiniMax's turn!")
                bestMove = miniMaxAlgorithm.findBestMiniMaxMove(player)
                result = game.move(bestMove[0], bestMove[1], player)

            #player is Logistic so we check his best move and then make the move.
            if player == 2:
                print("It's Logistic Regression's turn!")
                bestMove = logisticRegressionAlgorithm.findBestLogisticMove(player)
                result = game.move(bestMove[0], bestMove[1], player)

            game.generateNewRow()
            game.displayBoard()

            if not result == 2:
                print("It's over!")
                newTrainX, newTrainY = game.getNewData(game.getWinner())

                logisticRegressionAlgorithm.addNewData(newTrainX, newTrainY)
                logisticRegressionAlgorithm.learningTime()

            if game.getWinner() == 0:
                print("Match Draw!")
                drawCount += 1
                break
            elif game.getWinner() == 1:
                print("Minimax wins!")
                mmCount += 1
                break
            elif game.getWinner() == 2:
                print("Logistic Regression wins!")
                lrCount += 1
                break

            player = 2 if (player == 1) else 1
    print("==============================================================")
    print("Minimax: ", mmCount, "Logistic Regression: ", lrCount, "Draw: ", drawCount)
    print("==============================================================")



'''
########## KNN #############
def main():
    player = randint(1, 2)
    print("Let's begin.. KNN vs. MinMax")

    game = TicTacToeAI(3, ['X', 'O'])
    #     game.displayBoard()

    miniMaxAlgorithm = MiniMaxAlgorithm(game)
    KNN = KNNAlgorithm(game)
    randomPlacementAlgorithm = RandomPlacementAlgorithm(game)

    print("======================== Generating data =================================")
    #
    generateData(player, game, miniMaxAlgorithm, KNN, randomPlacementAlgorithm)
    #
    print("======================== Data has been generated =================================")

    KNN.learningTime()

    print("======================== IT'S BATTLE TIME =================================")

    mmCount = 0
    lrCount = 0
    drawCount = 0
    #Numbers of rounds
    for i in range(5):
        print("Battle number: ", i + 1)
        print("==============================================================")
        print("Minimax: ", mmCount, "KNN: ", lrCount, "Draw: ", drawCount)
        print("==============================================================")
        game.clear(3, ['X', 'O'])

        while (True):

            result = None
            #player is minmax so we check his best move and then make the move.
            if player == 1:
                print("It's MiniMax's turn!")
                bestMove = miniMaxAlgorithm.findBestMiniMaxMove(player)
                result = game.move(bestMove[0], bestMove[1], player)

            #player is Logistic so we check his best move and then make the move.
            if player == 2:
                print("It's KNN turn!")
                bestMove = KNN.findBestKNNMove(player)
                result = game.move(bestMove[0], bestMove[1], player)

            game.generateNewRow()
            game.displayBoard()

            if not result == 2:
                print("It's over!")
                newTrainX, newTrainY = game.getNewData(game.getWinner())

                KNN.addNewData(newTrainX, newTrainY)
                KNN.learningTime()

            if game.getWinner() == 0:
                print("Match Draw!")
                drawCount += 1
                break
            elif game.getWinner() == 1:
                print("Minimax wins!")
                mmCount += 1
                break
            elif game.getWinner() == 2:
                print("KNN wins!")
                lrCount += 1
                break

            player = 2 if (player == 1) else 1
    print("==============================================================")
    print("Minimax: ", mmCount, "KNN: ", lrCount, "Draw: ", drawCount)
    print("==============================================================")


################################
'''
'''
########## Decision tree #############
def Run_DecisionTree():
    player = randint(1, 2)
    print("Let's begin.. Decision Tree vs. MinMax")

    game = TicTacToeAI(3, ['X', 'O'])
    #     game.displayBoard()

    miniMaxAlgorithm = MiniMaxAlgorithm(game)
    Decision_Tree = DecisionTreeAlgorithm(game)
    randomPlacementAlgorithm = RandomPlacementAlgorithm(game)

    print("======================== Generating data =================================")
    #
    generateData(player, game, miniMaxAlgorithm, Decision_Tree, randomPlacementAlgorithm)
    #
    print("======================== Data has been generated =================================")

    Decision_Tree.learningTime()

    print("======================== IT'S BATTLE TIME =================================")

    mmCount = 0
    lrCount = 0
    drawCount = 0
    #Numbers of rounds
    for i in range(5):
        print("Battle number: ", i + 1)
        print("==============================================================")
        print("Minimax: ", mmCount, "Decision Tree: ", lrCount, "Draw: ", drawCount)
        print("==============================================================")
        game.clear(3, ['X', 'O'])

        while (True):

            result = None
            #player is minmax so we check his best move and then make the move.
            if player == 1:
                print("It's MiniMax's turn!")
                bestMove = miniMaxAlgorithm.findBestMiniMaxMove(player)
                result = game.move(bestMove[0], bestMove[1], player)

            #player is Logistic so we check his best move and then make the move.
            if player == 2:
                print("It's Decision Tree turn!")
                bestMove = Decision_Tree.findBestDecisionTreeMove(player)
                result = game.move(bestMove[0], bestMove[1], player)

            game.generateNewRow()
            game.displayBoard()

            if not result == 2:
                print("It's over!")
                newTrainX, newTrainY = game.getNewData(game.getWinner())

                Decision_Tree.addNewData(newTrainX, newTrainY)
                Decision_Tree.learningTime()

            if game.getWinner() == 0:
                print("Match Draw!")
                drawCount += 1
                break
            elif game.getWinner() == 1:
                print("Minimax wins!")
                mmCount += 1
                break
            elif game.getWinner() == 2:
                print("Decision Tree wins!")
                lrCount += 1
                break

            player = 2 if (player == 1) else 1
    print("==============================================================")
    print("Minimax: ", mmCount, "Decision Tree: ", lrCount, "Draw: ", drawCount)
    print("==============================================================")


################################
'''

def generateData(player, game, miniMaxAlgorithm, logisticRegressionAlgorithm, randomPlacementAlgorithm):

    # Generating data, the more I generate the better the ML algorithm will preform.
    for i in range(10):
        game.clear(3, ['X', 'O'])

        while (True):

            result = None

            if player == 1:
                #print ("It's MiniMax's turn!")
                bestMove = miniMaxAlgorithm.findBestMiniMaxMove(player)
                result = game.move(bestMove[0], bestMove[1], player)

            if player == 2:
                #print ("It's Random Placement's turn!")
                bestMove = randomPlacementAlgorithm.randomPlacement()
                result = game.move(bestMove[0], bestMove[1], player)

            game.generateNewRow()
            # game.displayBoard()

            if not result == 2:
                # print ("It's over!")
                newTrainX, newTrainY = game.getNewData(game.getWinner())

                logisticRegressionAlgorithm.addNewData(newTrainX, newTrainY)

            if game.getWinner() == 0:
                # print("Match Draw!")
                break
            elif game.getWinner() == 1:
                # print("Minimax wins!")
                break
            elif game.getWinner() == 2:
                # print("Random Placement wins!")
                break

            player = 2 if (player == 1) else 1

    # Generating data for draw matches
    for i in range(5):
        game.clear(3, ['X', 'O'])

        while (True):

            result = None

            #             if player == 1:
            #                 print ("It's MiniMax1's turn!")
            #             if player == 2:
            #                 print ("It's MiniMax2's turn!")

            bestMove = miniMaxAlgorithm.findBestMiniMaxMove(player)
            result = game.move(bestMove[0], bestMove[1], player)

            game.generateNewRow()
            # game.displayBoard()

            if not result == 2:
                # print ("It's over!")
                newTrainX, newTrainY = game.getNewData(game.getWinner())

                logisticRegressionAlgorithm.addNewData(newTrainX, newTrainY)

            if game.getWinner() == 0:
                # print("Match Draw!")
                break
            elif game.getWinner() == 1:
                # print("Minimax1 wins!")
                break
            elif game.getWinner() == 2:
                # print("Minimax2 wins!")
                break

            player = 2 if (player == 1) else 1

    # unblockPrint()



'''
def main():
    player = randint(1, 2)
    print("Let's begin..")

    game = TicTacToeAI(3, ['X', 'O'])
    #     game.displayBoard()

    miniMaxAlgorithm = MiniMaxAlgorithm(game)

    Decision_tree = DecisionTreeAlgorithm(game)
    KNN = KNNAlgorithm(game)

    randomPlacementAlgorithm = RandomPlacementAlgorithm(game)

    print("======================== Generating data =================================")

    generateData(player, game, miniMaxAlgorithm, Decision_tree, KNN,
                 randomPlacementAlgorithm)

    print("======================== Data has been generated =================================")

    Decision_tree.learningTime()
    KNN.learningTime()

    print("======================== IT'S BATTLE TIME =================================")

    cnnCount = 0
    lrCount = 0
    drawCount = 0
    for i in range(5):
        print("Battle number: ", i + 1)
        print("==============================================================")
        print("KNN: ", cnnCount, "Decision Tree: ", lrCount, "Draw: ", drawCount)
        print("==============================================================")
        game.clear(3, ['X', 'O'])

        while (True):

            result = None

            if player == 1:
                print("It's KNN's turn!")
                bestMove = KNN.findBestKNNMove(player)  ###########
                result = game.move(bestMove[0], bestMove[1], player)

            if player == 2:
                print("It's Decision Tree's turn!")
                bestMove = Decision_tree.findBestDecisionTreeMove(player)
                result = game.move(bestMove[0], bestMove[1], player)

            game.generateNewRow()
            game.displayBoard()

            if not result == 2:
                print("It's over!")
                newTrainX, newTrainY = game.getNewData(game.getWinner())

                Decision_tree.addNewData(newTrainX, newTrainY)
                Decision_tree.learningTime()

                KNN.addNewData(newTrainX, newTrainY)
                KNN.learningTime()

            if game.getWinner() == 0:
                print("Match Draw!")
                drawCount += 1
                break
            elif game.getWinner() == 1:
                print("KNN wins!")
                cnnCount += 1
                break
            elif game.getWinner() == 2:
                print("Decision Tree wins!")
                lrCount += 1
                break

            player = 2 if (player == 1) else 1
    print("==============================================================")
    print("KNN: ", cnnCount, "Decision Tree: ", lrCount, "Draw: ", drawCount)
    print("==============================================================")


def generateData(player, game, miniMaxAlgorithm, Decision_tree, KNNAlgorithm,
                 randomPlacementAlgorithm):
    # Generating winning and losing data
    for i in range(10):
        game.clear(3, ['X', 'O'])

        while (True):

            result = None

            if player == 1:
                # print ("It's MiniMax's turn!")
                bestMove = miniMaxAlgorithm.findBestMiniMaxMove(player)
                result = game.move(bestMove[0], bestMove[1], player)

            if player == 2:
                # print ("It's Random Placement's turn!")
                bestMove = randomPlacementAlgorithm.randomPlacement()
                result = game.move(bestMove[0], bestMove[1], player)

            game.generateNewRow()
            # game.displayBoard()

            if not result == 2:
                # print ("It's over!")
                newTrainX, newTrainY = game.getNewData(game.getWinner())

                Decision_tree.addNewData(newTrainX, newTrainY)
                KNNAlgorithm.addNewData(newTrainX, newTrainY)

            if game.getWinner() == 0:
                # print("Match Draw!")
                break
            elif game.getWinner() == 1:
                # print("Minimax wins!")
                break
            elif game.getWinner() == 2:
                # print("Random Placement wins!")
                break

            player = 2 if (player == 1) else 1

    # Generating data for draw matches
    for i in range(5):
        game.clear(3, ['X', 'O'])

        while (True):

            result = None

            #             if player == 1:
            #                 print ("It's MiniMax1's turn!")
            #             if player == 2:
            #                 print ("It's MiniMax2's turn!")

            bestMove = miniMaxAlgorithm.findBestMiniMaxMove(player)
            result = game.move(bestMove[0], bestMove[1], player)

            game.generateNewRow()
            # game.displayBoard()

            if not result == 2:
                # print ("It's over!")
                newTrainX, newTrainY = game.getNewData(game.getWinner())

                Decision_tree.addNewData(newTrainX, newTrainY)
                KNNAlgorithm.addNewData(newTrainX, newTrainY)

            if game.getWinner() == 0:
                # print("Match Draw!")
                break
            elif game.getWinner() == 1:
                # print("Minimax1 wins!")
                break
            elif game.getWinner() == 2:
                # print("Minimax2 wins!")
                break

            player = 2 if (player == 1) else 1
'''

if __name__ == "__main__":
    run_Logistic()




#if __name__ == "__main__":
 #   main()