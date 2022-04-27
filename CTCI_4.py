#---------------------------------------------------------------------------------------------------------
# 4.1 Route Between Nodes: Given a directed graph, design an algorithm to find out whether there is a
# route between two nodes.

#Time Complexity: O(V^2) with adj list it would be O(V+E), Space complexity: O(V)
def route_between_nodes(graph,n1,n2):
	
	visited = set()
	def helper(n1,n2):
		if n1 == n2: return True
		if n1 in visited: return False
		visited.add(n1)
		for i in range(len(graph[n1])):
			if graph[n1][i]: 
				return helper(i,n2)
		
		return False
	return helper(n1,n2)

#---------------------------------------------------------------------------------------------------------
# 4.2 Minimal Tree: Given a sorted (increasing order) array with unique integer elements, write an algorithm
# to create a binary search tree with minimal height.

class TreeNode:
	def __init__(self,val):
		self.val = val
		self.left=None
		self.right=None

# Time complexity: O(N), Space Complexity: O(H) since balanced H= logn
def minimal_tree(sorted_arr):
	if not sorted_arr: return None
	mid = len(sorted_arr)//2
	root = TreeNode(sorted_arr[mid])
	root.left = minimal_tree(sorted_arr[:mid])
	root.right = minimal_tree(sorted_arr[mid+1:])

	return root


#---------------------------------------------------------------------------------------------------------
# 4.3 List of Depths: Given a binary tree, design an algorithm which creates a linked list of all the nodes
# at each depth (e.g., if you have a tree with depth D, you'll have D linked lists).

class ListNode:
	def __init__(self,val=0,next=None):
		self.val = val
		self.next = next

class LinkedList:
	def __init__(self,val):
		self.head = ListNode(val)
		self.tail = self.head

	def add(self, item):
		self.tail.next = ListNode(item)
		self.tail = self.tail.next

# Time Complexity: O(N), Space Complexity: O(N) since lists returned take up N space in RAM (uses additional O(H) where H=logN for balanced else H=N space for recursive call stack )
def list_of_depths(root):

	ls = []
	def helper(root,depth):

		if not root: return
		if len(ls)==depth: ls.append(LinkedList(root.val))
		else: ls[depth].add(root.val)
		helper(root.left, depth+1)
		helper(root.right, depth+1)

	
	helper(root,0)
	return ls

#---------------------------------------------------------------------------------------------------------
# 4.4 Check Balanced: Implement a function to check if a binary tree is balanced. For the purposes of
# this question, a balanced tree is defined to be a tree such that the heights of the two subtrees of any
# node never differ by more than one.

#Time Complexity: O(N), Space Complexity: O(N) -> since tree can be completely unbalanced, will use stack of size N
def check_balanced(root):
	def helper(root):
		if not root: return -1
		hl = helper(root.left)
		hr = helper(root.right)
		if abs(hl-hr)>1: return float('-inf')
		else: return max(hl,hr) + 1
	return helper(root)!=float('-inf')

#---------------------------------------------------------------------------------------------------------
# 4.5 Validate BST: Implement a function to check if a binary tree is a binary search tree.

def validate_bst(root):

	def helper(root,min,max):
		if not root: return True
		if root.val<=min or root.val>max: return False
		return helper(root.left,min,root.val) and helper(root.right, root.val, max)

	return helper(root,float('-inf'),float('inf'))

#---------------------------------------------------------------------------------------------------------
# 4.6 Successor: Write an algorithm to find the "next" node (i.e., in-order successor) of a given node in a
# binary search tree. You may assume that each node has a link to its parent.

# to return next smallest element
def successor_small(root):
	if not root: return None
	def helper(root):
		if not root: return root.parent
		if root.left: return root.left
		return helper(root.right)
	return helper(root) 

def successor(root):
	if not root: return None

	# gets left most child of root
	def helper(root):
		if not root: return root.parent
		return helper(root.left)

	# return successor from right subtree if exists
	if root.right: return helper(root.right)

	#keep going up until we are at a left child
	parent = root.parent
	while parent and root != parent.left:
		root = parent
		parent = parent.parent
	return root.parent
	


#---------------------------------------------------------------------------------------------------------
# 4.7 Build Order: You are given a list of projects and a list of dependencies (which is a list of pairs of
# projects, where the second project is dependent on the first project). All of a project's dependencies
# must be built before the project is. Find a build order that will allow the projects to be built. If there
# is no valid build order, return an error.
# EXAMPLE
# Input:
# projects: a, b, c, d, e, f
# dependencies: (a, d), (f, b), (b, d), (f, a), (d, c)
# Output: f, e, a, b, d, c

# Time Complexity: O(V+E), Space Complexity: O(V)
def build_order(projects, dependencies):

	# map project to number, and build graph (adj list)
	idxs = {projects[i]:i for i in range(len(projects))}
	graph= {i: [] for i in range(len(projects))}
	for d in dependencies: graph[idxs[d[0]]].append(idxs[d[1]])
		
	
	def topSortHelper(v,visited,stack):
		visited[v] = True
		# recursuvely call on all adj v's
		for i in graph[v]:
			if not visited[i]: topSortHelper(i,visited,stack)

		# add current v to stack
		stack.append(v)

	def topSort():
		# mark all vertecies as not visited, and create empty stack
		visited = [False for i in projects]
		stack = []
		
		# call recursive helper to store topsort for all v's sequentially
		for i in range(len(projects)):
			if not visited[i]:
				topSortHelper(i,visited,stack)

		# return stack in rev. order, for our case we also need to map back vertex number to project letter
		projs = [projects[stack[i]] for i in range(len(stack)-1,-1,-1) ]
		return projs

	return topSort()

#---------------------------------------------------------------------------------------------------------
# 4.8 First Common Ancestor: Design an algorithm and write code to find the first common ancestor
# of two nodes in a binary tree. Avoid storing additional nodes in a data structure. NOTE: This is not
# necessarily a binary search tree.

#Time Complexity: O(N), Space Complexity: O(1)
def common_ancestor(root,n1,n2):
	# checks if a node exists in the tree
	def existsInTree(root,n):
		if not root: return False
		if root == n: return True
		return existsInTree(root.left,n) or existsInTree(root.right,n)
	
	# if one or both dont exists, no common ancestor
	if not existsInTree(root,n1) or not existsInTree(root,n2): return None
	
	# actual common ancestor logic
	def helper(root,n1,n2):
		if not root: return None
		if root == n1 or root == n2: return root

		l = common_ancestor(root.left,n1,n2)
		r = common_ancestor(root.right,n1,n2)
		if l and r: return root
		return l if l else r
	
	return helper(root,n1,n2)

# cleaner solution, same time compelexity as above but one traversal
def common_ancestor_2(root,n1,n2):
	#check for node in (sub)tree
	def existsInSubTree(root,n):
		if not root: return False
		if root == n: return True
		return existsInSubTree(root.left,n) or existsInSubTree(root.right,n)

	#same algo as above, but save bool flags if either node is made contact with during search
	def helper(root,n1,n2,v):
		if not root: return None
		if root == n1:
			v[0] = root
			return root
		if root == n2:
			v[1] = root
			return root
		l = common_ancestor(root.left,n1,n2,v)
		r = common_ancestor(root.right,n1,n2,v)
		if l and r: return root
		return l if l else r

	v=[False, False]
	res = helper(root,n1,n2,v)
	# if both found during ancestor search, return ancestor, otherwise return ancestor only if other node exists within the subtree of the found node
	if (v[0] and v[1]) or (v[0] and existsInSubTree(res,n2)) or (v[1] and existsInSubTree(res,n1)): return res
	return None

#---------------------------------------------------------------------------------------------------------
# 4.9 BST Sequences: A binary search tree was created by traversing through an array from left to right
# and inserting each element. Given a binary search tree with distinct elements, print all possible
# arrays that could have led to this tree.
# EXAMPLE
# Input:
#					2
#				  /   \
#				 1	   3
#               1        4
# Output: {2, 1, 3}, {2, 3, 1}

# this is a tough question, but it is easy if you think about it in terms of a topological sort problem, basically find all possible ways to peform topsort
# and use a version of kahn's top sort algorithm

# this finds all top sorts, (input is a graph represented as adj list)
def all_topsorts(graph):

	def helper(path, visited):

		for v in range(vertices):
			if not indegree[v] and not visited[v]:
				for u in graph[v]: indegree[u]-=1
				path.append(v)
				visited[v] = True
				
				#recur
				helper(path,visited)

				#backtrack
				for u in graph[v]: indegree[u]+=1
				path.pop()
				visited[v] = False

		# if path contains all vertices add to final list
		if len(path) == vertices: alltops.append(path)

	# get number of vertices, and create vector/list to repr in degree of each vertex (basically how many vertexes point to this vertex)
	vertices = len(graph)
	indegree = [0 for i in vertices]
	for i in graph:
		for j in graph[i]: indegree[j]+=1
	# create visited list and a final list that will be appended to and returned
	visited  = [False for i in vertices]
	alltops = []

	helper([],visited)

# can use a version of the above, except we will have to concoct visited and indegree arrays as we go along
# Time Complexity: O(N^2), Space Complexity: O(N)
def bst_sequence(root):

	def helper(roots):
		all = []
		for root in roots:
			# this is equivalent to a (not) visited array, we include all possible roots except the one we are at, and add the children aswell
			avail = [r for r in roots if r !=root]
			if root.left: avail.append(root.left)
			if root.right: avail.append(root.right)
			# if there nodes availble, we recur on them, after the recurrence finished we prepend root to subsequences and append to our return list
			if len(avail)>0:
				subseq = helper(avail)
				for s in subseq: all.append([root.val]+s)
			# if no choices to be made, append root to return list
			else: all.append([root.val])
		return all

	return helper([root])

#or
def bst_sequence_2(root):
	def helper(choices, toporder):
		# if no avail choices, we have completed a topsort, append to final list
		if not choices: all.append(toporder)
		for c in choices:
			# get all possible choices, minus root, plus children
			n = [i for i in choices if i!=c]
			if c.left: n.append(c.left)
			if c.right: n.append(c.right)
			# recur on new set of choices, and add root to topsort order list
			helper(n,toporder+[c.val])
	
	all = []
	helper([root],[])
	return all
	
#---------------------------------------------------------------------------------------------------------
# 4.10 Check Subtree: T1 and T2 are two very large binary trees, with T1 much bigger than T2. Create an
# algorithm to determine if T2 is a subtree of T1.
# A tree T2 is a subtree of T1 if there exists a node n in T1 such that the subtree of n is identical to T2.
# That is, if you cut off the tree at node n, the two trees would be identical.

def check_subtree(t1,t2):
	# empty t2 is always subtree
	if not t2: return True

	def helper(t1,t2,check):
		if not t1 and not t2: return True
		if not t1 or not t2: return False
		if check:
			if t1.val != t2.val: return False
			return helper(t1.left,t2.left,True) and helper(t1.right,t2.right,True)
		if t1.val == t2.val:
			c = helper(t1.left,t2.left,True) and helper(t1.right,t2.right,True)
			if c: return True
		return helper(t1.left,t2,False) or helper(t1.right,t2,False)
	
	return helper(t1,t2,False)

# cleaner version of the above
def check_subtree_clean(t1,t2):
	# empty t2 is always subtree
	if not t2: return True
	
	#check if two trees are equivalent
	def matchTree(t1,t2):
		if not t1 and not t2: return True
		if not t1 or not t2: return False
		if t1.val != t2.val: return False
		return matchTree(t1.left,t2.left) and matchTree(t1.right,t2.right)
	
	# traverse t1, in hopes of finding match
	def helper(t1,t2):
		if not t1: return False
		if t1.val == t2.val and matchTree(t1,t2): return True
		return helper(t1.left,t2) or helper(t1.right,t2)
	
	return helper(t1,t2)

#---------------------------------------------------------------------------------------------------------
# 4.11 Random Node: You are implementing a binary search tree class from scratch, which, in addition
# to insert, find, and delete, has a method getRandomNode() which returns a random node
# from the tree. All nodes should be equally likely to be chosen. Design and implement an algorithm
# for getRandomNode, and explain how you would implement the rest of the methods.

import random

class TreeNode:
	def __init__(self,val):
		self.val = val
		self.left = None
		self.right = None
		self.size = 1

class BST:
	def __init__(self,val):
		self.root = None
		self.size = 0

	def insert(self,val):
		if not self.size: self.root = TreeNode(val)
		else: self.insertHelper(self.root,val)
		self.size+=1

	def insertHelper(self,root,val):
		if not root: return TreeNode(val)
		if val <= root.val:
			root.left = self.insertHelper(root.left,val)
			root.size+=1 
		else: 
			root.right = self.insertHelper(root.right,val)
			root.size+=1
		return root

	def find(self,val):
		if not val: return None
		return self.findHelper(self.root,val)

	def findHelper(self, root, val):
		if not root: return None
		if root.val == val: return root
		if val <= root.val: self.findHelper(root.left, val)
		self.findHelper(root.right, val)
	
	def delete(self,val):
		if not val: return None
		if self.size == 1:
			self.root = None
			self.size -=1
			return
		self.deleteHelper(self.root,val)
	
	def getMin(self,root):
		if not root.left: return root
		return self.getMin(root.left)

	def deleteHelper(self,root,val):
		if not root: return None

		if val < root.val: root.left = self.deleteHelper(root.left,val)
		elif val> root.val: root.right = self.deleteHelper(root.right,val)
		else:
			if not root.left and not root.right: root = None
			elif not root.right: root = root.left
			else:
				r_min = self.getMin(root.right)
				r_min.left = root.left
				root = None

		return root

	def getIthNode(self,root,i):
		# if left exists get size of left subtree. if i equals left subtree size then our current node is the correct one, since left subtree contains
		# nodes 0:l_size-1. if i is less than l_size, then recur into the left subtreet to find it. Otherwise, it is in the right subtree, so recur into
		# the right subtree, and from i, subtract off l_size nodes in left subtree as well as the 1 parent we were at
		l_size = root.left.size if root.left else 0
		if i == l_size: return root
		if i < l_size: return self.getIthNode(root.left,i)
		return self.getIthNode(root.right,i-l_size-1)

	# Time Complexity: O(D) where D is max depth of the tree, if balanced then D = logN (.....base 2 :) )
	def getRandomNode(self):
		# generate random number in range of size of BST (starting at 0), then get the ith node wrt to inorder traversal
		if not self.size: return None
		r = random.randint(0, self.size-1)
		return self.getIthNode(self.root,r)


#---------------------------------------------------------------------------------------------------------
# 4.12 Paths with Sum: You are given a binary tree in which each node contains an integer value (which
# might be positive or negative). Design an algorithm to count the number of paths that sum to a
# given value. The path does not need to start or end at the root or a leaf, but it must go downwards
# (traveling only from parent nodes to child nodes).

#Time Complexity: O(N^2) (for unbalanced tree) or NlogN for balanced, Space Complexity: O(D) (worst case is N), 
# with lall paths list it becomes O(D^2) = O(N^2), since max length of list can be N and we make N recursive calls
# Remember Master's Theorem: T(n) = aT(n/b)+ f(n), here a = 2, b = 2, f(n) = 1 for traverse. So case 1, O(n^logba-eps=1) = O(1) = f(n) ->T(n) = O(n)
# And helper runs max n times. Therefore O(N^2)
def paths_with_sum(root,sum):

	def traverse(root,sum,path):
		if not root: return None
		if sum - root.val == 0: all_paths.append(path+[root.val])
		traverse(root.left,sum-root.val,path+[root.val])
		traverse(root.right,sum-root.val,path+[root.val])
		return None

	def helper(root,sum):
		if not root: return None
		traverse(root,sum,[])
		helper(root.left,sum)
		helper(root.right,sum)
		return None
	
	all_paths= []
	helper(root,sum)
	return (all_paths)

# Time Complexity: O(N), Space Complexity: O(D) (balanced D = logN)
def paths_with_sum_optimized(root,sum):
	
	def helper(root,rs):
		global count
		if not root: return None
		#update running sum
		rs += root.val
		#difference between running sum and target sum, if present in cache, means we are on a valid path, increment count
		if rs-sum in cache: count+= cache[rs-sum]
		#add running sum to cache, to be used by nodes further down own upcoming traversals
		if rs in cache: cache[rs]+=1
		else: cache[rs]=1
		#recur on left and right children
		helper(root.left,rs)
		helper(root.right,rs)
		#decrement running sum from cache, so it not reused once current recursion is done, if val is zero, pop it out to save space
		cache[rs]-=1
		if not cache[rs]: cache.pop(rs)

	#make cache/hashmap, global count (since not within class, cant use a class var), call helper with init sum 0
	#0:1 in cache bcuz if ever the difference is zero, this means we are at target sum, so it needs to be a part of the cache
	cache = {0:1}
	global count
	count = 0
	helper(root,0)
	return (count)
