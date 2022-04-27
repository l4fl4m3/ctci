#---------------------------------------------------------------------------------------------------------
# 9.1 Stock Data: Imagine you are building some sort of service that will be called by up to 1,000 client
# applications to get simple end-of-day stock price information (open, close, high, low). You may
# assume that you already have the data, and you can store it in any format you wish. How would
# you design the client-facing service that provides the information to client applications? You are
# responsible for the development, rollout, and ongoing monitoring and maintenance of the feed.
# Describe the different methods you considered and why you would recommend your approach.
# Your service can use any technologies you wish, and can distribute the information to the client
# applications in any mechanism you choose.

# Ans: We could use a a variety of solutions. A fancy way to distribute the data would be building a front end 
# application, in something like React, and therefore controlling the way in which our data is given to the client(s). 
# We would store our data in some back end DB/DataStore and write procedures that are callable by the end users in our 
# front end application. We use MERN or whatever other type of stack to get everything flowing together. This approach seams
# a bit of an overkill, if we are not looking to stricly build a product experience for the user. Alternatively, we could just let
# the end users plug in to or access our DB (SQL,noSQL, etc), and take it themselves from there. Obviously we would limit what they 
# are and are not able to do with the DB, but at minimum we would give them read privilages. This solution allows the client(s) to 
# tailor their own custom data extraction methods, and offers more breadth in what they can't and cant do with the day, of course it 
# is also a further burden on the client(s).


#---------------------------------------------------------------------------------------------------------
# 9.2 Social Network: How would you design the data structures for a very large social network like
# Facebook or Linkedln? Describe how you would design an algorithm to show the shortest path
# between two people (e.g., Me-> Bob-> Susan-> Jason-> You).

# Ans: Now adays probably best to use a graph structure, specfically a Graph Neural Net. Can use an implementation of something 
# like PINSAGE or any other popular GNN used for recommender system. This sort of structure will allow you to implement sophisticated
# deep learning techniques within the entire social network. Of course a much more fundamental approach has to be taken into building
# databases, client side methods, server side code, and just efficient and clean way in which a scalable product would be built upon.
# For shortest path, just use BFS (djikstra's)

# Time Complexity: O(V+E), Space Complexity: O(V) -> maybe O(V^2)
def shortest_path(graph, user1, user2):
	visited={user1}
	queue=[user1]

	while queue:
		path = queue.pop(0)
		vertex = path[-1]
		if  vertex== user2: return path
		for neighbor in graph[vertex]:
			if  neighbor not in visited:
				queue.append(path+[vertex])
				visited.add(neighbor)
	return None
# 0->1->2
# Time Complexity: O(V+E), Space Complexity: O(V)
def shortest_path(graph, user1, user2):
	
	def helper():
		visited={user1}
		queue=[user1]

		while queue:
			vertex = queue.pop(0)
			if  vertex== user2: return True
			for neighbor in graph[vertex]:
				if  neighbor not in visited:
					queue.append(vertex)
					visited.add(neighbor)
					pred[neighbor] = vertex
		return False

	pred = [-1 for _ in range(len(graph))]
	if not helper(): return None
	
	path=[user2]
	idx=user2
	while pred[idx]!=-1:
		path.append(pred[idx])
		idx = pred[idx]
	return reversed(path)

#---------------------------------------------------------------------------------------------------------
# 9.3 Web Crawler: If you were designing a web crawler, how would you avoid getting into infinite loops?


#---------------------------------------------------------------------------------------------------------
# 9.4 Duplicate URLs: You have 10 billion URLs. How do you detect the duplicate documents? In this
# case, assume "duplicate" means that the URLs are identical.

#---------------------------------------------------------------------------------------------------------
# 9.5 Cache: Imagine a web server for a simplified search engine. This system has 100 machines to
# respond to search queries, which may then call out using proc essSearch ( string query) to
# another cluster of machines to actually get the result. The machine which responds to a given query
# is chosen at random, so you cannot guarantee that the same machine will always respond to the
# same request. The method proc essSearch is very expensive. Design a caching mechanism for
# the most recent queries. Be sure to explain how you would update the cache when data changes.

#---------------------------------------------------------------------------------------------------------
# 9.6 Sales Rank: A large eCommerce company wishes to list the best-selling products, overall and by
# category. For example, one product might be the #1056th best-selling product overall but the #13th
# best-selling product under "Sports Equipment" and the #24th best-selling product under "Safety."
# Describe how you would design this system.

#---------------------------------------------------------------------------------------------------------
# 9.7 Personal Financial Manager: Explain how you would design a personal financial manager (like
# Mint.com). This system would connect to your bank accounts, analyze your spending habits, and
# make recommendations.

#---------------------------------------------------------------------------------------------------------
# 9.8 Pastebin: Design a system like Pastebin, where a user can enter a piece of text and get a randomly
# generated URL to access it.