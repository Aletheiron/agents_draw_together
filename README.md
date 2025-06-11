I think that intelligence is the emergent property of interconnected and interacted agents acting in the system with rewards and external boundaries.
So, can we make a group of entities capable to draw desired picture?
In details. We have 16 agents placed in form of square. Each agent takes care about four elements in matrix two by two. It can assign 0 or 1 in every possible place in matrix. So, each epoch we have some sort of picture from big matrix 8 by 8. 64 pixels in total. 
For make this group of agents paint what we want, I should implement utility function. We have utility function of the whole system. It is simply minus sum of mean square loss. This utility is shared by all agents. 
Plus, each agent adds with some weight utility from accordance with neighbor agents. This accordance measures the deviation from averaged neighbor. 
Optimization process is a simple discrete change in direction of growth of the utility function. Each step agent chooses an action: change the value in the allowed place or not. It checks utility function change, if it grows or doesn’t decline, action is approved. If not – canceled. In case of approval the next period agent repeats this action to the next point. 
And now the most interesting part. I will examine the system utility function with different parameters of neighbor influence. 
