#---------------------------------------------------------------------------------------------------------
# 8.1 Triple Step: A child is running up a staircase with n steps and can hop either 1 step, 2 steps, or 3
# steps at a time. Implement a method to count how many possible ways the child can run up the
# stairs.

def triple_step(n):
	if not n: return 0
	dp = [0 for _ in range(n+1)]
	dp[0] = 1
	for i in range(1,n+1): 
		dp[i] = dp[i-1] + dp[i-2] + dp[i-3]
	return dp[n]

#or
def triple_step_mem(n):

	def helper(n):
		if n<0: return 0
		if n==0: return 1
		if m[n]: return m[n]

		m[n] = helper(n-1) + helper(n-2) + helper(n-3)
		return m[n]

	m=[0 for _ in range(n+1)]
	return helper(n)

#print(triple_step(5))

#---------------------------------------------------------------------------------------------------------
# 8.2 Robot in a Grid: Imagine a robot sitting on the upper left corner of grid with r rows and c columns.
# The robot can only move in two directions, right and down, but certain cells are "off limits" such that
# the robot cannot step on them. Design an algorithm to find a path for the robot from the top left to
# the bottom right.

# Time Complexity: O(R*C) therefore N (number of grid spaces), Space Complexity: O(R*C)
def robot_in_grid(grid):

	def helper(i,j,path):
		if i>len(grid)-1 or j>len(grid[0])-1 or grid[i][j]==1: return None
		if i == len(grid)-1 and j == len(grid[0])-1: return path
		if (i,j) in cache: return None
		cache.add((i,j))
		print(cache)
		return helper(i+1,j,path+[(i+1,j)]) or helper(i,j+1,path+[(i,j+1)])
	
	cache = set()
	return helper(0,0,[(0,0)])

# Time Complexity: O(R*C) therefore N (number of grid spaces), Space Complexity: O(R*C)
def robot_in_grid_dp(grid):

	dp = [[0 for _ in range(len(grid[0]))] for _ in range(len(grid))]

'''
g = [[0 for _ in range(5)] for _ in range(5)]

g[1][3] = 1
g[2][3] = 1
g[3][3] = 1
g[4][3] = 1
print(robot_in_grid(g))
'''

#---------------------------------------------------------------------------------------------------------
# 8.3 Magic Index: A magic index in an array A[1 ... n-1] is defined to be an index such that A[i] = i. 
# Given a sorted array of distinct integers, write a method to find a magic index, if one exists, in
# array A.
# FOLLOW UP
# What if the values are not distinct?

# Time Complexity: O(logN), Space Complexity:O(1)
def magic_index(arr):

	l,r = 0, len(arr)-1
	while l <= r:
		mid = (l+r)//2
		if arr[mid] == mid: return mid
		if arr[mid] > mid: r = mid-1
		else: l = mid+1
	return False

# Time Complexity: O(N), Space Complexity: O(logN)
def magic_index_followup(arr):

	def helper(l,r):
		if r<l: return False
		mid = (l+r)//2
		if arr[mid] == mid: return mid
		return helper(l,min(mid-1, arr[mid])) or helper(max(mid+1,arr[mid]),r)

	return helper(0,len(arr)-1)
                    
#print(magic_index_followup([-5,-4,-3,-1,5,6,6]))

#---------------------------------------------------------------------------------------------------------
# 8.4 Power Set: Write a method to return all subsets of a set.

#s =(1,2,3), ps = ((),(a),(b),(c),(a,b),(a,c),(b,c),(a,b,c))

#Time Complexity: O(N*2^N), Space Complexity: O(N*2^N)
# T(n) = 
def power_set(s):

	def helper(myset,subset):
		ps.append(subset)
		if not myset: return
		for i in range(len(myset)): 
			helper(myset[i+1:],subset +[myset[i]])
	
	ps = []
	helper(list(s),[])
	return ps

#s ={1,2,3}
#print(power_set(s))
	
#---------------------------------------------------------------------------------------------------------
# 8.5 Recursive Multiply: Write a recursive function to multiply two positive integers without using
# the * operator (or / operator). You can use addition, subtraction, and bit shifting, but you should
# minimize the number of those operations.

# Time Complexity: O(log(min(x,y))), Space Complexity: O(log(min(x,y))) (recursion depth stack)
def recursive_multiply(x,y):
	# use bit shifting, divide one number by 2(right shift), multiply other (left shift), and watch for the case
	# where the the right shift number is odd, just add it to the return
	def helper(small,big):
		if small&1: return big+ recursive_multiply(small>>1, big<<1)
		if small: return recursive_multiply(small>>1, big<<1)
		return 0
	# find smaller of the two, to reduce number of operations 
	small = x if x<y else y
	big = y if x<y else x
	return helper(small, big)

#print(recursive_multiply(8,6))

#---------------------------------------------------------------------------------------------------------
# 8.6 Towers of Hanoi: In the classic problem of the Towers of Hanoi, you have 3 towers and N disks of
# different sizes which can slide onto any tower. The puzzle starts with disks sorted in ascending order
# of size from top to bottom (i.e., each disk sits on top of an even larger one). You have the following
# constraints:
# (1) Only one disk can be moved at a time.
# (2) A disk is slid off the top of one tower onto another tower.
# (3) A disk cannot be placed on top of a smaller disk.
# Write a program to move the disks from the first tower to the last using Stacks.

# Time Complexity: O(2^N), Space Complexity: O(N) (max depth of stack)
# T(n) = 2T(n-1) + 1 = 2(2T(n-2)+ 1)+1 = (2^2)*T(n-2) + 2^1 + 2^0 = (2^k)*T(n-k) + 2^(k-1) + 2^(k-2) -> base T(1)=1 
# n-k =1, k=n-1 -> T(n) = (2^(n-1))*T(1) + 2^(k-1) + 2^(k-2) = (2^(n-1)) + 2^(k-1) + 2^(k-2) = (2^k) + 2^(k-1) + 2^(k-2) = (2^n)-1

def tower_of_hanoi(n, fromT, toT, tempT):

	if not n: return
	tower_of_hanoi(n-1,fromT,tempT,toT)
	toT.append(fromT.pop())
	tower_of_hanoi(n-1,tempT,toT,fromT)

'''
n = 4
stack1 = [i for i in range(n,0,-1)]
stack2 = []
stack3 = []

print("BEFORE")
print(f'T1: {stack1}')
print(f'T2: {stack2}')
print(f'T3: {stack3}')
tower_of_hanoi(n,stack1,stack3,stack2)
print("AFTER")
print(f'T1: {stack1}')
print(f'T2: {stack2}')
print(f'T3: {stack3}')
'''

#---------------------------------------------------------------------------------------------------------
# 8.7 Permutations without Dups: Write a method to compute all permutations of a string of unique
# characters.

#P(0) = 0, P(1) = 1, P(2) = 2, P(3) = 6, P(4) = 24

# Time Complexity: O(N^2 * N!), Space Complexity: O(N*N!)
def permutations_without_dups(input_string):
	
	def helper(i_s,p):
		if len(p) == len(input_string):
			all.append(p)
			return
		for i in range(len(i_s)): helper(i_s[:i]+i_s[i+1:],p+i_s[i])

	all = []
	helper(input_string,'')
	return all

#print(permutations_without_dups("bert"))

#---------------------------------------------------------------------------------------------------------
# 8.8 Permutations with Duplicates: Write a method to compute all permutations of a string whose
# characters are not necessarily unique. The list of permutations should not have duplicates.

# This way is inefficient, doing alot of unnecessary calcs
def permutations_with_dups_bad(input_string):

	def helper(i_s,p):
		if len(p) == len(input_string):
			if p not in all: all.add(p)
			return
		for i in range(len(i_s)):
			print("a")
			helper(i_s[:i]+i_s[i+1:],p+i_s[i])
			

	all = set()
	helper(input_string,'')
	return all

#print(permutations_with_dups_bad("aaaaaa"))

# Time Complexity: O(N* N!), Space Complexity: O(N*N!), where N=number of unique chars
def permutations_with_dups(input_string):

	def helper(p):
		if len(p) == len(input_string):
			all.append(p)
			return
		for k,v in count.items():
			print("b")
			if v:
				count[k]-=1
				helper(p+k)
				count[k]+=1

	all = []
	count = {}
	for c in input_string:
		if c not in count:count[c]=1
		else: count[c]+=1
	helper('')
	return all

#print(permutations_with_dups("aaaaaa"))

#---------------------------------------------------------------------------------------------------------
# 8.9 Parens: Implement an algorithm to print all valid (i.e., properly opened and closed) combinations
# of n pairs of parentheses.
# EXAMPLE
# Input: 3
# Output: ( ( () ) ) , ( () () ) , ( () ) () , () ( () ) , () () ()

def parens(n):
	
	def helper(p):
		if len(p)==2*n:
			all.append(p)
			return
		
		for k,v in count.items():
			if k=="(" and v:
				count[k]-=1
				helper(p+k)
				count[k]+=1
			if k==")" and v> count['(']:
				count[k]-=1
				helper(p+k)
				count[k]+=1
	
	all = []
	count={'(':n,')':n}
	helper('')
	return all

# Time Complexity: O(N*Cat(N)), Space Complexity: O(N*Cat(N)) where Cat(N) is the nth Catalan number, worst case for all possible
# permutation of parentheses regardless of validity would be O(2^N) where N= 2*number_pairs_parentheses (e.g. for 3-> N=6)
def parens_cleaner(n):

	def helper(c_left,c_right,p):
		if len(p)==2*n:
			all.append(p)
			return
		if c_left: helper(c_left-1,c_right,p+'(')
		if c_right and c_right>c_left: helper(c_left,c_right-1,p+')')
	
	all = []
	helper(n,n,'')
	return all

#print(parens_cleaner(4))

#---------------------------------------------------------------------------------------------------------
# 8.10 Paint Fill: Implement the "paint fill" function that one might see on many image editing programs.
# That is, given a screen (represented by a two-dimensional array of colors), a point, and a new color,
# fill in the surrounding area until the color changes from the original color.

# Time Complexity: O(M*N), Space Complex: O(M*N) (max depth of stack)
def paint_fill(screen,point,new_color):

	def helper(point, old_color, new_color):

		if point[0]==len(screen) or point[1]==len(screen[0]) or \
			screen[point[0]][point[1]] != old_color or new_color==old_color: return
		
		screen[point[0]][point[1]] = new_color

		helper((point[0]+1,point[1]), old_color, new_color)
		helper((point[0],point[1]+1), old_color, new_color)
		helper((point[0]-1,point[1]), old_color, new_color)
		helper((point[0],point[1]-1), old_color, new_color)
	
	helper(point, screen[point[0]][point[1]], new_color)
'''
s = [['W' for _ in range(5)] for _ in range(5)]

s[0][0] = 'G'
s[0][1] = 'G'
s[1][0] = 'G'
s[1][1] = 'G'

print(s)
paint_fill(s,(1,3),'B')
print(s)
'''

#---------------------------------------------------------------------------------------------------------
# 8.11 Coins: Given an infinite number of quarters (25 cents), dimes (1O cents), nickels (5 cents), and
# pennies (1 cent), write code to calculate the number of ways of representing n cents.

# Time Complexity: O(N*k) , Space Complexity: O(N*k) where k in # of coin types
def coins(n):

	def helper(n,coin):

		if not n: return 1
		if n<0 or coin>(len(coins)-1): return 0
		if cache[n][coin]>0: return cache[n][coin]
		count = helper(n-coins[coin],coin) + helper(n,coin+1)
		cache[n][coin] = count
		print("i")
		return count
		
	cache = [[0 for _ in range(4)] for _ in range(n+1)]
	coins = [25,10,5,1]
	c = helper(n,0)
	return c

#print(coins(25))

# can also do DP bottom up
#  1 1 1 1 
#  0 0 0 0  1
#  0 0 0 0  2
#  0 0 0 0  3
#  0 0 0 0  4
#  0 0 0 0  5
# Time Complexity: O(N*k), Space Complexity: O(N*k)
def coins_dp(n):

	coins = [25,10,5,1]	
	dp = [[0 for _ in range(4)] for _ in range(n+1)]
	# ways to repr. sum of zero is always 5, regardless of the coin
	for i in range(4): dp[0][i]=1
	# for every j coins, the count adding to current sum i, will at least be the same as the count 
	# for the previous j-1 coins. If the current sum minus the new coins value, i-coins[j], is greater than or 
	# equal to zero, then add to the count, the count required to make the sum of i-coins[j] with up to j coins
	for i in range(n):
		for j in range(4):
			dp[i+1][j] = dp[i+1][j-1]
			if i+1 - coins[j]>=0: dp[i+1][j]+= dp[i+1 - coins[j]][j]

	return dp[-1][-1]

#print(coins_dp(25))

# Time Complexity: O(N*k), Space Complexity: O(N)
def coins_dp_space_optimized(n):
	""" think of this version as almost a k-hop version of the problem, with a certain coin type we can get to the current 
	 	sum, the same amount times as we could get to sum-coin_value using all the other coins, keeping in mind that we will 
		go thorugh each coin sequetially first, as to not double count anything """

	coins = [25,10,5,1]
	dp = [0 for _ in range(n+1)]
	dp[0] = 1

	for i in range(4):
		for j in range(coins[i]-1,n): dp[j+1] += dp[j+1-coins[i]]
	
	return dp[-1]

#print(coins_dp_space_optimized(25))

#---------------------------------------------------------------------------------------------------------
# 8.12 Eight Queens: Write an algorithm to print all ways of arranging eight queens on an 8x8 chess board
# so that none of them share the same row, column, or diagonal. In this case, "diagonal" means all
# diagonals, not just the two that bisect the board.
#
# X 0 0 0 0 0 0 0
# 0 0 X 0 0 0 0 0
# 0 0 0 0 X 0 0 0   -> not valid
# 0 0 0 0 0 0 X 0
# 0 X 0 0 0 0 0 0
# 0 0 0 X 0 0 0 0
# 0 0 0 0 0 X 0 0
# 0 0 0 0 0 0 0 X

# 0 X 0 0 0 0 0 0
# 0 0 0 X 0 0 0 0
# 0 0 0 0 0 X 0 0   -> not valid
# 0 0 0 0 0 0 0 X
# 0 0 X 0 0 0 0 0
# 0 0 0 0 X 0 0 0
# 0 0 0 0 0 0 X 0
# X 0 0 0 0 0 0 0

# diag_right = row-col, diag_left = row+col, can use this to help us. we will keep track of currently used columns, diag_right, 
# and diag_left.The queens list keep track of column in a row where a queen has been placed, since we can only have one per row, 
# the length of this list can be used as a proxy for the row. 

# Time Complexity: O(N!), Space Complexity: O(result*N^2) ??
def eight_queens(n):

	def helper(queens, diag_left, diag_right):
		row = len(queens)
		if row==n:
			all.append(queens)
			return
		for col in range(n):
			if col not in queens and row-col not in diag_left and row+col not in diag_right:
				helper(queens+[col], diag_left+[row-col], diag_right+[row+col])

	all = []
	helper([],[],[])
	
	for a in all:
		for q in a:
			r = [0 for _ in range(n)]
			r[q] = 1
			print(r)
		print("------")
	
#eight_queens(8)

#---------------------------------------------------------------------------------------------------------
# 8.13 Stack of Boxes: You have a stack of n boxes, with widths w_i , heights h_i, and depths d_i. The boxes
# cannot be rotated and can only be stacked on top of one another if each box in the stack is strictly
# larger than the box above it in width, height, and depth. Implement a method to compute the
# height of the tallest possible stack. The height of a stack is the sum of the heights of each box.

# We can solve this like a knapsack problem, with backtracking
def stack_of_boxes(boxes):
	
	def helper(current):
		if len(current) ==  n: return
		if current: all.append(current)

		for i in range(n):
			if not current: helper(current+[i])
			elif i not in current and boxes[current[-1]][0]>boxes[i][0] \
				and boxes[current[-1]][1]>boxes[i][1] and boxes[current[-1]][2]>boxes[i][2]: 
				helper(current+[i])

	n = len(boxes)
	all = []
	helper([])
	print(all)
	max_height = 0
	for comb in all:
		h = 0
		for box in comb: h += boxes[box][1]
		max_height = max(max_height,h)
	return max_height

# Time Complexity : O(N^2), Space Complexity: O(N)
def stack_of_boxes_optimized(boxes):
	
	def helper(bottom,idx):

		if idx >= len(boxes): return 0
		
		if idx in cache and ((boxes[bottom][0]> boxes[idx][0]) and (boxes[bottom][2]> boxes[idx][2])): return cache[idx]
		
		h1,h2 = helper(bottom, idx+1), 0
		
		if bottom==-1 or ((boxes[bottom][0]> boxes[idx][0]) and (boxes[bottom][2]> boxes[idx][2])): 
			h2 = boxes[idx][1]+ helper(idx, idx+1)

		m = max(h1,h2)
		cache[idx] = m
		
		return cache[idx]

	# sort by height
	boxes = sorted(a, key = lambda x: x[1], reverse=True)
	print(boxes)
	cache={}
	m_h = helper(-1,0)
	return m_h

# Bottom up DP version
# Time Complexity: O(N^2), Space Complexity: O(N)
def stack_of_boxes_optimized(boxes):
	''' We basically turn this into a LIS (longest increasing subsequence) problem. Sort the boxes by height in decreasing order.
		Then make a DP arr and initalize each index to the height of each respective box. Then for each box,
		check if any previous box was bigger by all dimensions and if there is such a box, set the current box's dp val (max height)
		to max(current_box max_height, found box's max_height + current_box height). Finally, since the dp arr represents
		the max height of a stacks with the box at an index being the top box in the given stack, we just find the max value in the 
		dp arr to get the height of the tallest stack possible.
		'''
	boxes = sorted(boxes, key=lambda x: x[1], reverse=True)
	dp = [boxes[i][1] for i in range(len(boxes))]

	for i in range(1,len(boxes)):
		for j in range(i):
			if boxes[j][0]>boxes[i][0] and boxes[j][2]>boxes[i][2] and boxes[j][1]>boxes[i][1]: 
				dp[i] = max(dp[i], dp[j]+boxes[i][1])
	return max(dp)


#a = [(1,4,6),(4,8,2),(4,5,8),(3,2,4),(6,6,4), (2,1,3), (2,3,3), (5,1,3), (4,5,7)]
#b = [(1,4,6),(0,4,5),(4,5,8)]
#print(stack_of_boxes_optimized(b))


#---------------------------------------------------------------------------------------------------------
# 8.14 Boolean Evaluation: Given a boolean expression consisting of the symbols 0 (false), 1 (true), &
# (AND), | (OR), and ^ (XOR), and a desired boolean result value result, implement a function to
# count the number of ways of parenthesizing the expression such that it evaluates to result. The
# expression should be fully parenthesized (e.g., (0)^(1)) but not extraneously (e.g.,(((0))^(1))).
# EXAMPLE
# countEval("1^0|0|1", false) -> 3  , (1) ^ (0|0|1), (1) ^ ((0) | (0|1)), (1) ^ ((0|0) | (1))
# countEval("0&0&0&1^1|0", true)-> 10

#Time Complexity: O(N^4), Space Complexity: O(N)
def boolean_evaluation(b_val, result):

	def helper(b_val, result):
	
		if len(b_val)==0: return 0
		if len(b_val)==1: return 1 if int(b_val) == result else 0 
		if b_val+str(result) in cache: return cache[b_val+str(result)]

		count = 0
		for i in range(1,len(b_val),2):

			c = b_val[i]
			left = b_val[:i]
			right = b_val[i+1:]
			left_true = helper(left, True)
			left_false = helper(left, False)
			right_true = helper(right, True)
			right_false = helper(right, False)

			total = (left_true+left_false) * (right_true+right_false)

			if c == '^':
				all = left_true*right_false + left_false*right_true
			elif c == '&':
				all = left_true*right_true
			elif c == '|':
				all = left_true*right_true + left_true*right_false + left_false*right_true

			# return count depending on whether target is true or false, if true it is equal to calcs above since we did them for 
			# true. If false it is just the total possible count - count for true 
			sub_count = all if result else total-all
			count += sub_count
		
		cache[b_val+str(result)] = count
		return count
	
	cache={}
	c = helper(b_val,result)
	return c

# Bottom up DP approach
# "1^0|0|1"
# T            F
#   1 0 0 1      1 0 0 1 
# 1 1          1 0
# 0   0        0   1
# 0     0      0     1
# 1       1    1       0
#
# Time Complexity: O(N^3), Space Complexity: O(N^2), this approach is very troublesome to wrap your head around             
def boolean_evaluation_dp(b_val, result):

	t = [[0 for _ in range(0,len(b_val),2)] for _ in range(0,len(b_val),2)]
	f = [[0 for _ in range(0,len(b_val),2)] for _ in range(0,len(b_val),2)]

	# initialize diagonals depending on character, these are basically base cases
	for i in range(len(t)):
		t[i][i] = 1 if b_val[i*2] == "1" else 0
		f[i][i] = 0 if b_val[i*2] == "1" else 1
	# iterate thorugh each operator
	for op in range(1,len(t)):

		i=0
		# all chars to the rightnof op
		for j in range(op,len(t)):
			# all chars to the left of op
			for g in range(op):
				
				k = i+g
				tik = t[i][k] + f[i][k]
				tkj = t[k+1][j] + f[k+1][j]

				if b_val[2*k+1] == '&':
					t[i][j] += t[i][k]*t[k+1][j]
					f[i][j] += ((tik*tkj) - t[i][k]*t[k+1][j])

				if b_val[2*k+1] == '|':
					t[i][j] += ((tik*tkj) - f[i][k]*f[k+1][j])
					f[i][j] += f[i][k]*f[k+1][j]

				if b_val[2*k+1] == '^':
					t[i][j] += t[i][k]*f[k+1][j] + f[i][k]*t[k+1][j]
					f[i][j] += t[i][k]*t[k+1][j] + f[i][k]*f[k+1][j]

			i+=1

	return t[0][-1] if result else f[0][-1]

#r = boolean_evaluation_dp("1^0|0|1",True)
#print(r)