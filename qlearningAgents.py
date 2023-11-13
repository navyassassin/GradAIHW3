from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *
import random, util, math

class QLearningAgent(ReinforcementAgent):
    """
    Q-Learning Agent

    Functions you should fill in:
      - computeValueFromQValues
      - computeActionFromQValues
      - getQValue
      - getAction
      - update

    Instance variables you have access to
      - self.epsilon (exploration prob)
      - self.alpha (learning rate)
      - self.discount (discount rate)

    Functions you should use
      - self.getLegalActions(state)
        which returns legal actions for a state
    """

    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        "*** YOUR CODE HERE ***"
        self.q_values = util.Counter()

    def getQValue(self, state, action):
        """
        Returns Q(state,action)
        Should return 0.0 if we have never seen a state
        or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        return self.q_values[(state, action)]

    def computeValueFromQValues(self, state):
        """
        Returns max_action Q(state,action)
        where the max is over legal actions. Note that if
        there are no legal actions, which is the case at the
        terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        actions = self.getLegalActions(state)
        if not actions:
            return 0.0
        temp_counter = util.Counter()
        for action in actions:
            temp_counter[action] = self.getQValue(state, action)
        return temp_counter[temp_counter.argMax()]

    def computeActionFromQValues(self, state):
        """
        Compute the best action to take in a state. Note that if there
        are no legal actions, which is the case at the terminal state,
        you should return None.
        """
        "*** YOUR CODE HERE ***"
        actions = self.getLegalActions(state)
        best_action = None
        max_val = float('-inf')
        for action in actions:
            q_value = self.q_values[(state, action)]
            if max_val < q_value:
                max_val = q_value
                best_action = action
        return best_action

    def getAction(self, state):
        """
        Compute the action to take in the current state. With
        probability self.epsilon, we should take a random action and
        take the best policy action otherwise. Note that if there are
        no legal actions, which is the case at the terminal state, you
        should choose None as the action.

        HINT: You might want to use util.flipCoin(prob)
        HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        "*** YOUR CODE HERE ***"
        actions = self.getLegalActions(state)
        explore = util.flipCoin(self.epsilon)
        if explore:
            return random.choice(actions) if actions else None
        else:
            return self.getPolicy(state)

    def update(self, state, action, nextState, reward):
        """
        The parent class calls this to observe a
        state = action => nextState and reward transition.
        You should do your Q-Value update here

        NOTE: You should never call this function,
        it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        old_q_value = self.getQValue(state, action)
        old_part = (1 - self.alpha) * old_q_value
        reward_part = self.alpha * reward
        if not nextState:
            self.q_values[(state, action)] = old_part + reward_part
        else:
            nextState_part = self.alpha * self.discount * self.getValue(nextState)
            self.q_values[(state, action)] = old_part + reward_part + nextState_part

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)
