
#!/usr/bin/env python

'''
Assignment 01 
- Problem 01: Finding an optimized route between a set of start and end city
- Algorithms implemented: 'A*', 'Uniform Cost', 'BFS', 'DFS'
- Cost Functions used: 'distance', 'time', 'segments', 'longtour'
- Please check the ipython notebook available at :
  http://nbviewer.jupyter.org/gist/Dheeraj2444/9cbf0370e4e7c87a3875dc9de67bf580
  where we did initial data visualization to understand speed and distance distribution
- For observations, where speed was 0 or NaN, we replaced it with median (45). 0 distance was also replaced with median (18).
- There were 19 observations in road-segments.txt with only 4 items; since it was too small 
  in comparision to overall dataset, we removed those observations.
- We have used Dictionary to create bidirectional graph taking one city at a time as a key and rest all as values. 
- For uniform cost and A*, we have used heapq module for priority queue implementation; 
  For longtour, we implemented max heap and for remaining cost functions, we implemented min heap
  
 Problem Abstraction: - Initial State: Start City (user input)
                      - State Space: Set of all possible nodes (or cities) that can be traversed along the graph
                      - Goal: End City (user input)
                      - Successor Function: Returns list of all nodes (or next connected cities) connected to the current node 
		        (or current city); Implemened by creating  a dictionary 
  
  Edge Weights: - In case of BFS and DFS, since they consider uniform cost for each edge, we have nor alloted any weight to edges
                  and they would return the same path for each cost function.
		For uniform cost and A*, edge weights depends upon the cost function:
		- distance: travelling distance as per the input file provided
		- time: travelling distance divided by speed for each set of connected nodes (provided in the input file)
		- segments: one for each edge
		- longtour: travelling distance 
		
 Cost of Node:  Uniform Cost:
                - distance: Sum of total travelling distance from start city to current node
		- time : Sum of total time from start city to current node
		- segments: SUm of total number of edges traversed from start city to current node
		- longtour: Sum of total travelling distance from start city to current node
		A*
		- distance: Sum of travelling distance from start city to current node plus heuristic distance of current node
		- time : Sum of total time from start city to current node plus heuristic time of current node
		- segments: sum of total edges traversed from start city to current node plus 1 (used 1 as heuristic for each
		  node); since we are adding a constant to node cost, it won't have any difference and run similar to uniform 
		  cost
		- longtour: Same as distance cost function
	
Heuristic Explaination: Mentioned in assignment1.pdf Please check the file
Searh Algorithm: Uniform Cost: Depends on cost function
                longtour:
		- Implemented cost-based priority queue using heapq module. Since, we need to find maximum distance; used 
		  max heap. This will remove the element with maximum cost from the fringe
		- Remove eleemnt with maximum cost from the fringe; check if goal, if yes, goal reached; otherwise, get the 
		  successor, check if already visited, if not, check if already in fringe with lesser cost, if yes, update the
		  cost, and if not add to the fringe maitaing the max heap property. Keep on iterating until the fringe 
		  is not empty
		distance, time, segment:
		- Used priority queue and implemented min heap using heapq module removing the element with minimum cost from 
		  the fringe. Similar to above, but with few changes: if the successor is already in the fringe with larger
		  cost, update the cost. And while, removing from fringe, always remove the one with lowest cost
		
		A*: Since, our heuristic was not consistent but admissible, we implemented search algorithm# 2 in which re-visit
		of states are allowed for optimal results
		distance, time, segment:
		- Similar to uniform cost but with no restriciton of revisit of states
		longtour: 
		- Similar to uniform cost; since it was mentioned in the problem to not re-visit any city
		BFS/DFS:
		- traverse along the path without considering all edges with unit cost
		- BFS: traverse along the breadth first (queue implementation)
		- DFS: traverse along the depth first (stack implementation)
		
Assumptions/Design : 
	       - Data: Did few data modifications (mentioned above)
	       - Multiplied the distance heuristic with 1.2 to enhance the performace of A*
                 		
'''

import sys
import heapq
import math
from math import sin, cos, sqrt, atan2, radians

#initializing dictionary to create graph
graph = {}
lines = open('road-segments.txt').readlines()
towns = []

for line in lines:
	towns.append(len(line.split())) 

#indices with missing items
indices = [i for i, x in enumerate(towns) if x!=5]
lines = [i for j, i in enumerate(lines) if j not in indices]
open('road-segment-new.txt', 'w').writelines(lines)

newlines = open('road-segment-new.txt').readlines()

#creating a dictionary
for line in newlines:
	line = line.strip() 	
	city1, city2, dist, speed, highway = line.split()
	graph.setdefault(city1, []).append([city2, int(dist), int(speed), highway])
	graph.setdefault(city2, []).append([city1, int(dist), int(speed), highway])

#replacing 0's and NaN's
for key in graph:
	for j in graph[key]:
		if j[1]==0:
			j[1]=18             #replacing 0 distance with median 
		if j[2]==0 or j[2] not in range(1, 5000):
			j[2]=45             #replacing 0 and NaN speed with median

# Identifying near by cities for Junctions to calculate their heuristic
junctions = []
for key in graph:
	if key[0:3] == 'Jct':
		junctions.append(key)

near_by_city = {}
for i in junctions:
	near_by_city[i] = sorted(graph[i], key=lambda x: int(x[1]))

for key in near_by_city:
	temp = near_by_city[key]
	temp2 = []
	for i in temp:
		if i[0][0:3]!='Jct':
			temp2.append(i)
	near_by_city[key] = temp2

#creating dictionary to store latitude and longitude of cities
coord_lines = open('city-gps.txt').readlines()
coord = {}
for line in coord_lines:
	line = line.strip()
	city, lat, lon = line.split()
	coord[city] = [float(lat), float(lon)]

#gives latitude and latitude in radians for a given city
def GetCoord(state):
	try:
		if state[0:3]!='Jct':
			lat = radians(coord[state][0])
			lon = radians(coord[state][1])
		else:
			lat = radians(coord[near_by_city[state][0][0]][0]) #lat & long of nearby city for junctions
			lon = radians(coord[near_by_city[state][0][0]][1])
		return [lat, lon]
	except (IndexError, KeyError) as e:
		return [float('inf'), float('inf')]                     #if no nearby city, return it as infinity

#calculates heuristic of a current node
def Heuristic(current_state):
	lat_goal = GetCoord(end_city)[0]
	lon_goal = GetCoord(end_city)[1]
	lat_current = GetCoord(current_state)[0]
	lon_current = GetCoord(current_state)[1]

	if lat_goal!=float('inf') and lon_goal!=float('inf') and lat_current!=float('inf') and lon_current!=float('inf'):
		dlon = lon_goal - lon_current
		dlat = lat_goal - lat_current
		a = (sin(dlat/2))**2+cos(lat_current)*cos(lat_goal)*(sin(dlon/2))**2  #Implemented from here: https://stackoverflow.com/questions/19412462/getting-distance-between-two-points-based-on-latitude-longitude
		c = 2*atan2(sqrt(a), sqrt(1 - a))
		if cost_function in ['distance', 'longtour']:
			if start_city.split(',_')[1]==end_city.split(',_')[1]:			
				return (6373.0*c)*0.621371
			else:
				return ((6373.0*c)*0.621371)*1.2  	#if start and end city are in different states; multiply by a factor of 1.2
		elif cost_function=='time':
			return ((6373.0*c)*0.621371)/45      	#dividing by median speed
		elif cost_function=="segments":
			return 1                                #heuristic = 1 for segments cost
	else:
		return 0

#generates all set of successors for a given node
def Successor(current_state):
	#no cost
	if method in ["bfs", "dfs"]:
		return [[i[0], current_state[1]+[i[0]], i[1]+current_state[2], current_state[3]+ float(i[1])/float(i[2])] for i in graph[current_state[0]]]
	# method = uniform; cost = distance or longtour
	if method == "uniform" and cost_function in ["distance", "longtour"]:
		return [[i[1]+current_state[0], i[0], current_state[2]+[i[0]], i[1]+current_state[3], current_state[4]+ float(i[1])/float(i[2])] for i in graph[current_state[1]]]
	# method = uniform; cost = segments 
	if method == "uniform" and cost_function == "segments":
		return [[1+current_state[0], i[0], current_state[2]+[i[0]], i[1]+current_state[3], current_state[4]+ float(i[1])/float(i[2])] for i in graph[current_state[1]]]
	#method = uniform; cost = time
	if method == "uniform" and cost_function == "time":
		return [[float(i[1])/float(i[2])+current_state[0], i[0], current_state[2]+[i[0]], i[1]+current_state[3], current_state[4]+ float(i[1])/float(i[2])] for i in graph[current_state[1]]]
	# method = astar; cost = distance or longtour
	if method == "astar" and cost_function in ["distance", "longtour"]:
		return [[i[1]+current_state[3]+Heuristic(i[0]), i[0], current_state[2]+[i[0]], i[1]+current_state[3], current_state[4]+ float(i[1])/float(i[2])] for i in graph[current_state[1]]]
	# method = astar; cost = time
	if method == "astar" and cost_function == "time":
		return [[float(i[1])/float(i[2])+current_state[4]+Heuristic(i[0]), i[0], current_state[2]+[i[0]], i[1]+current_state[3], current_state[4]+ float(i[1])/float(i[2])] for i in graph[current_state[1]]]
	# method = astar; cost = segments
	if method == "astar" and cost_function == "segments":
		return [[1+current_state[0]+Heuristic(i[0]), i[0], current_state[2]+[i[0]], i[1]+current_state[3], current_state[4]+ float(i[1])/float(i[2])] for i in graph[current_state[1]]]

#checks for a goal state
def Goal(state):
	if method not in ["bfs", "dfs"]:
		if state[1] == end_city:
			return True
	else:
		if state[0] == end_city:
			return True

#searches for an path for a given algorithm and cost function
def Search(initial_state):
	fringe = [initial_state]
	
	if method == 'uniform' and cost_function == 'longtour':         #uniform cost with longtour as cost function
		heapq._heapify_max(fringe)
		visited_city = []
		pop = []
		while len(fringe)>0:
			element_to_visit = heapq.heappop(fringe)
			pop.append(element_to_visit)
			visited_city.append(element_to_visit[1])
			if Goal(element_to_visit):
				return [pop[-1][3], pop[-1][4], pop[-1][2]]						
			for s in Successor(element_to_visit):	
				if s[1] not in visited_city:
					if s[1] in [i[1] for i in fringe]:
						i[0]=max(i[0],s[0])
					else:
						heapq.heappush(fringe, s)
						heapq._heapify_max(fringe)
		return False
	
	elif method == 'uniform':                                  		  #uniform cost
		heapq.heapify(fringe)
		visited_city = []
		pop = []
		while len(fringe)>0:
			element_to_visit = heapq.heappop(fringe)
			pop.append(element_to_visit)
			visited_city.append(element_to_visit[1])
			if Goal(element_to_visit):
				return [pop[-1][3], pop[-1][4], pop[-1][2]]						
			for s in Successor(element_to_visit):		
				if s[1] not in visited_city:
					if s[1] in [i[1] for i in fringe]:
						i[0]=min(i[0],s[0])
					else:
						heapq.heappush(fringe, s)
		return False

	elif method == 'astar' and cost_function == 'longtour':         #A* with longtour as cost function; using search Algo #2 (revisiting allowed)
		heapq._heapify_max(fringe)
		visited_city = []
		pop = []
		while len(fringe)>0:
			element_to_visit = heapq.heappop(fringe)
			pop.append(element_to_visit)
			visited_city.append(element_to_visit[1])
			if Goal(element_to_visit):
				return [pop[-1][3], pop[-1][4], pop[-1][2]]						
			for s in Successor(element_to_visit):	
				if s[1] not in visited_city:
					if s[1] in [i[1] for i in fringe]:
						i[0]=max(i[0],s[0])
					else:
						heapq.heappush(fringe, s)
						heapq._heapify_max(fringe)
		return False
	
	elif method =='astar':                                      	#A*; using search Algo #2 (revisiting allowed)
		heapq.heapify(fringe)
		pop = []
		while len(fringe)>0:			
			element_to_visit = heapq.heappop(fringe)
			pop.append(element_to_visit)
			if Goal(element_to_visit):
				return [pop[-1][3], pop[-1][4], pop[-1][2]]						
			for s in Successor(element_to_visit):		
				if s[1] in [i[1] for i in fringe]:
					i[0]=min(i[0],s[0])
				else:
					heapq.heappush(fringe, s)			
		return False
	
	elif method == 'bfs':                                     		#BFS
		visited = []             
		while len(fringe) > 0:
			for s in Successor(fringe.pop(0)):
				if s[0] not in visited:
					if Goal(s):
						return [s[2], s[3], s[1]]
					visited.append(s[0])
					fringe.append(s)
		return False
	
	elif method == "dfs":                                      		#DFS
		visited = []             
		while len(fringe) > 0:
			for s in Successor(fringe.pop()):
				if s[0] not in visited:
					if Goal(s):
						return [s[2], s[3], s[1]]
					visited.append(s[0])
					fringe.append(s)
		return False

start_city, end_city, method, cost_function = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

def main():
	if len(sys.argv)!=5 or method not in ['astar', 'uniform', 'bfs', 'dfs'] or cost_function not in ['segments', 'distance', 'time', 'longtour']:
		print "Invalid input; 4 arguments required; algorithm accepted: 'astar', 'uniform', 'bfs', 'dfs'; cost accepted: 'segments', 'distance', 'time', 'longtour' " 
	else:
		if method == "astar":
			initial_state = [Heuristic(start_city), start_city, [start_city], 0, 0]
		elif method == "uniform":
			initial_state = [int(0), start_city, [start_city], 0, 0]
		elif method in ["bfs", "dfs"]:
			initial_state = [start_city, [start_city], 0, 0]
		output = Search(initial_state)
		if output:
			print output[0], output[1], ' '.join(output[2])
		else:
			print "Sorry, no route found!"

if __name__ == "__main__":
	main()

