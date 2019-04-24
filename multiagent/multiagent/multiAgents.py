# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action) #new GameState
        newPos = successorGameState.getPacmanPosition() #position tuple
        newFood = successorGameState.getFood() #Grid, can call asList to get list of position tuples
        newGhostStates = successorGameState.getGhostStates() #list of AgentState
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates] #list of no-harm time left

        "*** YOUR CODE HERE ***"
        #return successorGameState.getScore() #resulting score of the board

        if successorGameState.isWin():
            return float("inf")
        else:
            score = successorGameState.getScore()
            stopped = action == Directions.STOP
            ateFood = successorGameState.getNumFood() < currentGameState.getNumFood()
            minDistToFood = min(manhattanDistance(fp, newPos) for fp in newFood.asList())
            ateCapsule = successorGameState.getPacmanPosition() in currentGameState.getCapsules()
            minDistToGhost = min(manhattanDistance(gp, newPos) for gp in successorGameState.getGhostPositions())
            minScaredTime = min(newScaredTimes)
            return score - stopped + ateFood - minDistToFood + ateCapsule + minDistToGhost + minScaredTime

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        depth = self.depth
        num_agents = gameState.getNumAgents()
        agentindex = -1

        def value(gameState, agentindex, depth):
            agentindex = (agentindex + 1) % num_agents
            if gameState.isLose() or gameState.isWin() or depth == 0:
                return self.evaluationFunction(gameState), None
            elif agentindex == 0:
                return max_value(gameState, agentindex, depth)
            else:
                return min_value(gameState, agentindex, depth)
        
        def max_value(gameState, agentindex, depth):
            pac_actions = gameState.getLegalActions(0)
            list_of_actions = []
            for x in pac_actions:
                val, _ = value(gameState.generateSuccessor(agentindex, x), agentindex, depth - 1)
                list_of_actions.append((val, x))
            return max(list_of_actions, key = lambda y: y[0])
        
        def min_value(gameState, agentindex, depth):
            ghost_actions = gameState.getLegalActions(agentindex)
            list_of_actions = []
            for x in ghost_actions:
                val, _ = value(gameState.generateSuccessor(agentindex, x), agentindex, depth - 1)
                list_of_actions.append((val, x))

            return min(list_of_actions, key = lambda y: y[0])

        _val, action = value(gameState, agentindex, num_agents*depth)
        return action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def value(gameState, depth, a, b, index):
            if depth == 0 or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            if index == 0:
                return maxValue(gameState, depth, a, b, index)
            else:
                return minValue(gameState, depth, a, b, index)

        def maxValue(gameState, depth, a, b, index):
            v = -float("inf")
            pac_actions = gameState.getLegalActions(0)
            for action in pac_actions:
                v = max(v, value(gameState.generateSuccessor(0, action), depth, a, b, index+1))
                if v > b: #prune max
                    return v
                a = max(a, v)
            return v

        def minValue(gameState, depth, a, b, index):
            v = float("inf")
            ghost_actions = gameState.getLegalActions(index)
            for action in ghost_actions:
                if index == gameState.getNumAgents() - 1:
                    d, i = depth-1, 0
                else:
                    d, i = depth, index+1
                v = min(v, value(gameState.generateSuccessor(index, action), d, a, b, i))
                if v < a: #prune min
                    return v
                b = min(b,v)
            return v

        #we do not need to store actions at each level so explicitly do it below
        a, b = -float("inf"), float("inf")
        legalMoves = gameState.getLegalActions()

        scores = []
        for action in legalMoves:
            v = value(gameState.generateSuccessor(0, action), self.depth, a, b, 1)
            a = max(v, a)
            scores.append(v)

        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)
        return legalMoves[chosenIndex]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        depth = self.depth
        num_agents = gameState.getNumAgents()
        agentindex = -1

        def value(gameState, agentindex, depth):
            agentindex = (agentindex + 1) % num_agents
            if gameState.isLose() or gameState.isWin() or depth == 0:
                return self.evaluationFunction(gameState), None
            elif agentindex == 0:
                return max_value(gameState, agentindex, depth)
            else:
                return exp_value(gameState, agentindex, depth), None
        
        def max_value(gameState, agentindex, depth):
            pac_actions = gameState.getLegalActions(0)
            list_of_actions = []
            for x in pac_actions:
                val, _ = value(gameState.generateSuccessor(agentindex, x), agentindex, depth - 1)
                list_of_actions.append((val, x))
            return max(list_of_actions, key = lambda y: y[0])
        
        def exp_value(gameState, agentindex, depth):
            ghost_actions = gameState.getLegalActions(agentindex)
            list_of_actions = []
            for x in ghost_actions:
                val, _ = value(gameState.generateSuccessor(agentindex, x), agentindex, depth - 1)
                list_of_actions.append((val, x))
            return sum(i for i, _ in list_of_actions)/len(ghost_actions)

        _val, action = value(gameState, agentindex, num_agents*depth)
        return action

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: We considered score, minimum distance to ghost, and minimum distance to ghost when on a power capsule as positive factors.
    We want to minimize pacman's distance to food, so we subtract that feature from the heuristic. We incentivized pacman to move closer
    to and eat the white ghosts by dividing the minimum scared time by the minimum distance to a ghost. If pacman did not eat the power
    capsule, the minimum scared time would be 0 and the feature wouldn't be accounted for. We wanted to cap the distance to a ghost because 
    pacman would stop moving when the ghost was far away and only start moving when the ghost was right next to pacman. Putting a cap on 
    the minimum distance to ghost ensures that pacman somtimes moves even when the ghost is far away. We found the hyperparamters through
    manual observation.
    """
    "*** YOUR CODE HERE ***"
    Pos = currentGameState.getPacmanPosition()
    Food = currentGameState.getFood()
    GhostStates = currentGameState.getGhostStates()
    ScaredTimes = [ghostState.scaredTimer for ghostState in GhostStates]
    

    if currentGameState.isWin():
        return float("inf")
    elif currentGameState.isLose():
        return -float("inf")
    else:
        score = currentGameState.getScore()
        minDistToFood = min(manhattanDistance(fp, Pos) for fp in Food.asList())
        minDistToGhost = min(manhattanDistance(gp, Pos) for gp in currentGameState.getGhostPositions())
        minScaredTime = min(ScaredTimes)
        return score - 0.5 * minDistToFood + 0.75 * minScaredTime / minDistToGhost + 0.5 * min(minDistToGhost, 10) 




# Abbreviation
better = betterEvaluationFunction
