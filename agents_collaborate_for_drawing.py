import numpy as np
from typing import Tuple, List
from PIL import Image
import copy
from copy import deepcopy
import matplotlib.pyplot as plt

#Some external values

#Influence of the neighbors
alpha=1

#Influence of the system
beta=1

#Influence of the internal feeling
gamma=0

#Target picture

#Rather Hard Data
target=np.array(((0,0,0,1,1,0,0,0),
                (0,0,1,0,1,0,0,0),
                (0,1,0,0,1,0,0,0),
                (0,0,1,0,1,0,0,0),
                (0,0,1,0,0,1,0,0),
                (0,1,0,1,1,1,0,0),
                (0,0,1,0,0,1,0,0),
                (0,0,1,0,0,1,0,0)))

#Normal Data
# target=np.array(((0,0,1,1,1,1,0,0),
#                  (0,0,1,1,1,1,0,0),
#                  (0,0,0,0,1,1,0,0),
#                  (0,0,0,0,1,1,0,0),
#                  (0,0,1,1,1,1,1,1),
#                  (0,0,1,1,1,1,1,1),
#                  (0,0,0,0,1,1,0,0),
#                  (0,0,0,0,1,1,0,0)))

#Simple Data
# target=np.array(((1,1,1,1,1,1,0,0),
#                  (1,1,1,1,1,1,0,0),
#                  (1,1,1,1,1,1,0,0),
#                  (1,1,1,1,1,1,0,0),
#                  (1,1,0,0,0,0,0,0),
#                  (1,1,0,0,0,0,0,0),
#                  (1,1,0,0,0,0,0,0),
#                  (1,1,0,0,0,0,0,0)))

#Independent agent class

class Drawing_Agent ():
    
    def __init__(self, coordinates: Tuple[float,float]):
        
        
        # Initial value chooses randomly from the array [0, 1] for each element
        self.value=np.random.choice([0,1],size=(2,2)).astype('float64')
        
        #Memory for changing value
        self.last_value=None
        self.changed_indices=(None,None)
        
        #Memory of actions taken
        self.action=None
        self.action_list=[]
        
        #Coordinates for neighbors searching
        self.coordinates=coordinates
        
        #List of neighbors is initialised empty
        self.neighbors: List['Drawing_Agent'] = [] 
        
        #Utility function value
        self.uf_value=None
        
        #List of utility values
        self.uf_list=[]
        
    #Local Utility function update
    
    def local_uf_update(self):
        
        #Compute average matrix of neighbors
        sum_of_arrays = np.zeros_like(self.value, dtype=float) # Use float for accurate division
        

        for arr in self.neighbors:
            sum_of_arrays += arr.value

        #Average array for the neighbors
        av_neigh = sum_of_arrays / len(self.neighbors)

        #Calculate part of utility function depending on neighbors
        local_uf= mse_joy(y=av_neigh,y_pred=self.value)
        
        #Case of quadratic function favorizing a little divergence from the typical nieghbor
        quadr_local_uf=-1*local_uf*(1+0.5*local_uf)
        
        
        #Internal function depending on internal situation
        #Favorable for a little divergence
        
        if np.sum(self.value) <=3 and np.sum(self.value)>=1:
            
            internal_uf=1
            
        else:
            internal_uf=-0.25
            
        
        
        
        #Whole utility function
        
        #self.uf_value=beta*UF_system+alpha*local_uf
        
        
        #self.uf_value=beta*UF_system+alpha*quadr_local_uf
        
        
        #self.uf_value=beta*UF_system+alpha*quadr_local_uf+gamma*internal_uf
        
        
        
        
        self.uf_value=beta*UF_system+alpha*local_uf+gamma*internal_uf
        
        #Update list of utility values
        self.uf_list.append(self.uf_value)
        
        return self.uf_value
        
        
    #Update internal values
    
    def update_internal_value (self):
        
        # if last update was profitable --> pick action of the last update --> form a list with possibilities of current action --> 
        # choose a random point from this list --> execute given action
        #if the last update was not lucrative --> randomly choose a value from matrix --> change this value --> remember this action
        
        #Conditions for the random picking of the elemnt in agent value matrix: first step or the lowering of the utility function
        if len(self.uf_list)<2 or self.uf_list[-1]<=self.uf_list[-2]:
            
            num_rows = self.value.shape[0] 
            num_cols = self.value.shape[1] 
            
            # Generate a random row index (0 or 1)
            random_row_index = np.random.randint(0, num_rows)

            # Generate a random column index (0 or 1)
            random_col_index = np.random.randint(0, num_cols)
            
            #Memorising indeces
            self.changed_indices=(random_row_index,random_col_index)
            
            # Pick the value using the random indices
            random_value = self.value[random_row_index, random_col_index]
            
            #Memorising previous value
            self.last_value=copy.deepcopy(self.value[random_row_index, random_col_index])
            
            #Update value to inverse
            
            #From 1 to 0
            if random_value==1:
                
                self.value[random_row_index, random_col_index]=0
                self.action=0
            
            #From 0 to 1    
            else:
                
                self.value[random_row_index, random_col_index]=1
                self.action=1
            
            #Append action in the line of actions
            self.action_list.append(self.action)
        
        #If all is going good and previous step was profitable 
        
        else:
            
            #Check if the last action is equal one
            
            if self.action_list[-1]==1:
                
                indices=np.where(self.value==0)
                
                indices=np.array(indices)
                
                #If we have enough variants to continue
                if len(indices[0]) != 0:
                   
                    # We choose the first convinient index
                    self.changed_indices=indices[:,0]
                    
                    #Memorizing value of the changing value in case of restoring needs
                    self.last_value=copy.deepcopy(self.value[self.changed_indices[0], self.changed_indices[1]])
                    
                    #Set new value for a given element
                    self.value[self.changed_indices[0], self.changed_indices[1]]=1
                    #Memorizing action
                    self.action=1
                    
                    #Append action in the line of actions    
                    self.action_list.append(self.action)
            
            #Case when the last action is equal zero    
            else:
                
                indices=np.where(self.value==1)
                
                indices=np.array(indices)
                
                # We choose the first convinient index
                if len(indices[0])!=0:
                    
                    # We choose the first convinient index
                    self.changed_indices=indices[:,0]
                    
                    #Memorizing value of the changing value in case of restoring needs
                    self.last_value=copy.deepcopy(self.value[self.changed_indices[0], self.changed_indices[1]])
                    
                    #Set new value for a given element
                    self.value[self.changed_indices[0], self.changed_indices[1]]=0
                    #Memorizing action
                    self.action=0
            
                    #Append action in the line of actions    
                    self.action_list.append(self.action)
                
    
    def check_and_restore (self):
        #We check the last action if it was profitable. In negative result we restore previous value
        
        #Check the length for possibiliy to comparing
        if len(self.uf_list)>=2:
            
            #For restoring the last utility function should be lower then the previous one
            if self.uf_list[-1]<=self.uf_list[-2]:
                
                #Restoring the value
                self.value[self.changed_indices[0], self.changed_indices[1]]=copy.deepcopy(self.last_value)
    


#Searchig neighbors for agents in case of influence

def assign_neighbors(nodes: List[Drawing_Agent], search_radius: float):
    
    num_nodes = len(nodes)
    # Pre-calculate squared search radius to avoid using math.sqrt in every comparison
    search_radius_sq = search_radius ** 2

    for i in range(num_nodes):
        node1 = nodes[i]
        node1.neighbors = []  # Clear any pre-existing neighbors for this run

        for j in range(num_nodes):
            if i == j:  # A node cannot be its own neighbor
                continue

            node2 = nodes[j]

            # Calculate squared Euclidean distance
            # (x2 - x1)^2 + (y2 - y1)^2
            dx = node2.coordinates[0] - node1.coordinates[0]
            dy = node2.coordinates[1] - node1.coordinates[1]
            dist_sq = dx*dx + dy*dy

            # If squared distance is within squared radius, they are neighbors
            if dist_sq <= search_radius_sq:
                node1.neighbors.append(node2)


#Method for receiving black and white image of the given matrix

def matrix_to_image(matrix):
    
    data = np.array(matrix, dtype=np.uint8)
    scaled_data = (data * 255).astype(np.uint8)
    img_gray = Image.fromarray(scaled_data, mode='L')
    img_bw_converted = img_gray.convert('1')
    img_bw_converted.show()



#System utility function update method
#MSE Function of Joy. Utility function can be anything in general

def mse_joy (y, y_pred):
    
    mse_j = -1*(np.mean((y - y_pred)**2))
    
    return mse_j




#Function of reunion of agents in one picture

def reunion_agents (draw_agent_list: List['Drawing_Agent'], batch_size=4):
    
    #Stack in column 
    sub_view_1=np.vstack(([agent.value for agent in draw_agent_list[0:batch_size]]))
    sub_view_2=np.vstack(([agent.value for agent in draw_agent_list[batch_size:2*batch_size]]))
    sub_view_3=np.vstack(([agent.value for agent in draw_agent_list[2*batch_size:3*batch_size]]))
    sub_view_4=np.vstack(([agent.value for agent in draw_agent_list[3*batch_size:]]))
    
    #Concat received columns
    view=np.concatenate((sub_view_1,sub_view_2,sub_view_3,sub_view_4), axis=1)
    
    return view



#Initializing agents

agent_1=Drawing_Agent(coordinates=(0,0))
agent_2=Drawing_Agent(coordinates=(0,1))
agent_3=Drawing_Agent(coordinates=(0,2))
agent_4=Drawing_Agent(coordinates=(0,3))

agent_5=Drawing_Agent(coordinates=(1,0))
agent_6=Drawing_Agent(coordinates=(1,1))
agent_7=Drawing_Agent(coordinates=(1,2))
agent_8=Drawing_Agent(coordinates=(1,3))

agent_9=Drawing_Agent(coordinates=(2,0))
agent_10=Drawing_Agent(coordinates=(2,1))
agent_11=Drawing_Agent(coordinates=(2,2))
agent_12=Drawing_Agent(coordinates=(2,3))

agent_13=Drawing_Agent(coordinates=(3,0))
agent_14=Drawing_Agent(coordinates=(3,1))
agent_15=Drawing_Agent(coordinates=(3,2))
agent_16=Drawing_Agent(coordinates=(3,3))

#List of drawing agents
draw_agent_list=[agent_1, agent_2, agent_3, agent_4,
                 agent_5, agent_6, agent_7, agent_8,
                 agent_9, agent_10, agent_11, agent_12,
                 agent_13, agent_14, agent_15, agent_16]

#Number of trainig steps
EPOCH=100000

#Establishing utility value list for the entier system
UF_system_list=[]


#Prelimenary steps

#Assign neighbors if we need
assign_neighbors(nodes=draw_agent_list, search_radius=1)

#First view on the given data
first_view=reunion_agents(draw_agent_list=draw_agent_list)
#matrix_to_image(first_view)

#Trainig loop
for epoch in range(EPOCH):
    
    #Constructing the whole picture to compare with
    
    view=reunion_agents(draw_agent_list=draw_agent_list)
    
    #Utility function of the system
    UF_system=mse_joy(y=target, y_pred=view)
    UF_system_list.append(UF_system)
    
    #print(f'Agent_1 values: {agent_1.value}')
    #Checking utility function on the fly
    if epoch%10==0:
        
        print(f'utility function is: {UF_system}')
        #Look at the picture
        #matrix_to_image(matrix=view)
    
    #Agents moves. They are consequetive. Maybe done in parallel
    for agent in draw_agent_list:
        
        agent.local_uf_update()
        agent.check_and_restore()
        agent.update_internal_value()

print(f'maximum of the run: {np.max(UF_system_list)}')
#Final picture
# matrix_to_image(target)
matrix_to_image(view)


#Plot Utility Function
plt.plot(UF_system_list)
plt.show()