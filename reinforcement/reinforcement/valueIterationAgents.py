# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        for i in range(self.iterations):
          ctr = util.Counter()
          for state in self.mdp.getStates():
            max_value = float("-inf")
            for action in self.mdp.getPossibleActions(state):
              q = self.computeQValueFromValues(state, action)
              if q > max_value:
                max_value = q
              ctr[state] = max_value
          self.values = ctr

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        lst = self.mdp.getTransitionStatesAndProbs(state, action)
        rewards = [prob * (self.mdp.getReward(state, action, nex) + self.discount * self.getValue(nex)) for nex, prob in lst]
        return sum(rewards)

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        optimal = None #default
        max_value = float("-inf")
        for action in self.mdp.getPossibleActions(state):
          q = self.computeQValueFromValues(state, action)
          if q > max_value:
            max_value = q
            optimal = action
        return optimal

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
      states = self.mdp.getStates()
      for i in range(self.iterations):
        state = states[i % len(states)]
        if not self.mdp.isTerminal(state):
          values = [self.computeQValueFromValues(state, a) for a in self.mdp.getPossibleActions(state)]
          self.values[state] = max(values) if len(values) > 0 else float("-inf")

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
      predecessors = {}
      for state in self.mdp.getStates():
        if not self.mdp.isTerminal(state):
          for action in self.mdp.getPossibleActions(state):
            for nex, prob in self.mdp.getTransitionStatesAndProbs(state, action):
              if nex not in predecessors:
                predecessors[nex] = {state}
              else:
                predecessors[nex].add(state)

      pq = util.PriorityQueue()
      for state in self.mdp.getStates():
        if not self.mdp.isTerminal(state):
          current = self.values[state]
          values = [self.computeQValueFromValues(state, a) for a in self.mdp.getPossibleActions(state)]
          diff = abs(max(values) - current) if len(values) > 0 else float("inf")
          pq.update(state, -diff)

      for i in range(self.iterations):
        if pq.isEmpty():
          break
        state = pq.pop()
        if not self.mdp.isTerminal(state):
          values = [self.computeQValueFromValues(state, a) for a in self.mdp.getPossibleActions(state)]
          self.values[state] = max(values) if len(values) > 0 else float("-inf")
        for p in predecessors[state]:
          if not self.mdp.isTerminal(p):
            values = [self.computeQValueFromValues(p, a) for a in self.mdp.getPossibleActions(p)]
            diff = abs(max(values) - self.values[p]) if len(values) > 0 else float("inf")
            if diff > self.theta:
              pq.update(p, -diff)
