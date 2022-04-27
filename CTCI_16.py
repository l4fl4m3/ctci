#---------------------------------------------------------------------------------------------------------
# 16.1 Number Swapper: Write a function to swap a number in place (that is, without temporary variables).

# Time Complexity: O(1), Space Complexity: O(1)
def number_swapper(a,b):

	a = a-b
	b = b+a
	a = b-a

	return a,b

def number_swapper_bit(a,b):   

	a = a^b
	b = b^a
	a = b^a

	return a,b

'''
n1 = 9
n2 = 3
print(n1)
print(n2)
print(number_swapper_bit(n1,n2))
'''

#---------------------------------------------------------------------------------------------------------
# 16.2 Word Frequencies: Design a method to find the frequency of occurrences of any given word in a
# book. What if we were running this algorithm multiple times?

# Time Complexity: O(N), Space Complexity: O(N)
cache={}
def word_frequencies(book,word):

	if word in cache: return cache[word]
	freq=0
	for w in book:
		if w == word:freq+=1
	cache[word]=freq
	return freq

#book=['hello', 'there', 'my' , 'friend', 'oh', 'my', 'friend']
#print(word_frequencies(book,"friend"))

#---------------------------------------------------------------------------------------------------------
# 16.3 Intersection: Given two straight line segments (represented as a start point and an end point),
# compute the point of intersection, if any.

# Time Complexity: O(1), Space Complexity: O(1)
import math
def intersection(line1, line2):

	x1_a, y1_a, x2_a, y2_a = line1[0][0],line1[0][1],line1[1][0],line1[1][1]
	x1_b, y1_b, x2_b, y2_b = line2[0][0],line2[0][1],line2[1][0],line2[1][1]

	# left to right for line1
	if x1_a>x2_a:
		x1_a,x2_a = x2_a,x1_a
		y1_a,y2_a = y2_a,y1_a
	# left to right for line2
	if x1_b>x2_b:
		x1_b,x2_b=x2_b,x1_b
		y1_b,y2_b=y2_b,y1_b
	# line1 always before line2 on x-axis
	if x1_a>x1_b:
		x1_a,x1_b=x1_b,x1_a
		y1_a,y1_b=y1_b,y1_a
		x2_a,x2_b=x2_b,x2_a
		y2_a,y2_b=y2_b,y2_a

	if x2_a-x1_a != 0:
		slope1 = (y2_a-y1_a)/(x2_a-x1_a)
		b1 = y1_a - (slope1*x1_a)
	else:
		slope1 = float('inf')
		b1 = float('inf')

	if x2_b-x1_b != 0:
		slope2 = (y2_b-y1_b)/(x2_b-x1_b)
		b2 = y1_b - (slope2*x1_b)
	else:
		slope2 = float('inf')
		b2 = float('inf')

	diff_slopes = slope1 - slope2
	diff_b = b1-b2

	if slope1 == slope2:
		if math.isnan(diff_b):
			if x1_a!=x1_b: return None
			if y1_a<=y1_b and y1_b<=y2_a: return (x1_a,y1_b)
			if y1_b<=y1_a and y1_a<=y2_b: return (x1_a,y1_a)
			return None

		if diff_b: return None
		if x1_b<=x2_a: return (x1_b,y1_b)
		else: return None
	
	if x2_a-x1_a != 0 and x2_b-x1_b != 0:
		x_int = (-1*diff_b)/diff_slopes
		y_int = slope1*x_int+ b1
	elif not x2_a-x1_a: 
		x_int = x1_a
		y_int = slope2*x_int+ b2
	elif not x2_b-x1_b: 
		x_int = x1_b
		y_int = slope1*x_int+ b1

	if x1_a<=x_int<=x2_a  and x1_b<=x_int<=x2_b  and min(y1_a,y2_a)<=y_int<=max(y1_a,y2_a) and min(y1_b,y2_b)<=y_int<=max(y1_b,y2_b):
		return (x_int,y_int)
	
	return None

'''
print(intersection(((0,0),(0,5)),((0,6),(0,8))))
print(intersection(((0,0),(0,5)),((0,1),(2,3))))
print(intersection(((0,0),(2,2)),((1,1),(4,4))))
print(intersection(((0,5),(2,1)),((0,-2),(3,4))))
'''

#---------------------------------------------------------------------------------------------------------
# 16.4 Tic Tac Win: Design an algorithm to figure out if someone has won a game of tic-tac-toe.

# Time Complexity: O(m*n), Space Complexity: O(1)
# added a hashtable with unique hash if running on reapeateded grids

def tic_tac_win(grid):
	def get_hash():
		hash = 0
		for i in range(len(grid)):
			for j in range(len(grid[0])):
				if grid[i][j] == 'X': hash = hash*3 + 1
				elif grid[i][j] == 'O':hash = hash*3 + 2
				#else: hash = hash + (3**(i+j))*val
		return hash

	def helper(i,j,count_x,count_o):

		if count_x == 3 or count_o == 3: return True
		if i<0 or j<0 or i>len(grid)-1 or j>len(grid)-1: return False
		if (i==1 and j==1) and ((grid[0][0]==grid[1][1] and grid[2][2]==grid[1][1]) or (grid[0][2]==grid[1][1] and grid[2][0]==grid[1][1])): return True
		if grid[i][j] =='X': return helper(i+1,j,count_x+1,0) or helper(i,j+1,count_x+1,0)
		if grid[i][j] =='O': return helper(i+1,j,0,count_o+1) or helper(i,j+1,0,count_o+1)
	
		return helper(i+1,j,0,0) or helper(i,j+1,0,0)

	hash = get_hash()
	if hash in cache: return cache[hash]
	ans = helper(0,0,0,0)
	cache[hash] = ans
	return ans
#cache={}

# Time Complexity: O(m*n), Space Complexity: O(m+n)
def tic_tac_win_clean(grid):
	r = [0 for _ in range(len(grid))]
	c = [0 for _ in range(len(grid))]
	d = [0,0]

	for i in range(len(grid)):
		for j in range(len(grid)):

			if grid[i][j]=="X": val = 1 
			elif grid[i][j]=="O": val = -1
			else: val = 0

			if not i-j: d[0]+=val
			if i+j+1==len(grid): d[1]+=val

			r[i]+=val
			c[j]+=val

	for score in r: 
		if abs(score)==len(grid): return True
	for score in c:
		if abs(score)==len(grid): return True
	for score in d:
		if abs(score)==len(grid): return True 
	return False

'''

a=[["X","O","X"],["O","X","O"],["X","O","O"]]
b=[["X","O","X"],["O","X","O"],["O","O","X"]]
c=[["X","O","O"],["X","O","X"],["X","X","O"]]
d=[["X","O","O"],["X","X","X"],["O","X","O"]]
e=[["X","O","O"],["O","X","O"],["X","X","O"]]
f=[["O","X","O"],["X","O","X"],["O","X","X"]]
g=[["O","X","O"],["X","O","X"],["X","O","X"]]

print(tic_tac_win_clean(a))
print(tic_tac_win_clean(b))
print(tic_tac_win_clean(c))
print(tic_tac_win_clean(d))
print(tic_tac_win_clean(e))
print(tic_tac_win_clean(f))
print(tic_tac_win_clean(g))

'''

#---------------------------------------------------------------------------------------------------------
# 16.5 Factorial zeros: Write an algorithm which computes the number of trailing zeros in n factorial.

# Time Complexity: O(N), Space Complexity: O(N)
def factorial_zeros(n):

	# zero when (2*5), since 2<5, then for a factorial #zeros is n//5, plus x-1 for 5^x <= n (x>0)
	if n<0: return None
	dp = [0 for _ in range(n+1)]
	for i in range(1,n+1):
		dp[i] = i//5 + dp[i//5]
	return dp[-1]

# Time Complexity: O(log5N) -> logbase5(N), Space Complexity: O(1)
def factorial_zeros(n):
	if n<0: return None
	count = 0
	i=5
	while n//i>0:
		count+=n//i
		i = i*5
	return count

#print(factorial_zeros(50))

#---------------------------------------------------------------------------------------------------------
# 16.6 Smallest Difference: Given two arrays of integers, compute the pair of values (one value in each
# array) with the smallest (non-negative) difference. Return the difference.
# EXAMPLE
# Input: {1, 3, 15, 11, 2}, {23, 127, 235, 19, 8}
# Output: 3. That is, the pair (11, 8).

# Time Complexity: O(NlogN) where N is maxlength(arr1,arr2) or just O(AlogA + BlogB), Space Complexity: O(1)
def smallest_difference_naive(arr1, arr2):

	arr1.sort()
	arr2.sort() 
	i,j = 0,0
	min = float('inf')
	while i<len(arr1) and j<len(arr2):
		diff = abs(arr1[i]-arr2[j])
		if diff< min:min = diff 
		if not diff: return diff
		if arr1[i]<arr2[j]: i+=1
		else:j+=1
	return min
	
#a = [1, 3, 15, 11, 2]
#b = [23, 127, 235, 19, 8]

#print(smallest_difference_naive(a,b))

#---------------------------------------------------------------------------------------------------------
# 16.7 Number Max: Write a method that finds the maximum of two numbers. You should not use if-else
# or any other comparison operator.

# Time Complexity: O(1), Space Complexity: O(1)
def number_max(num1, num2):

	# this solution is a bit more complicated of an implementation to handle for overflow (we use 32-bit here)
	diff = num1 - num2
	sign_diff = (((1<<31)-diff) >> 31) &1
	sign_num1 = (((1<<31)-num1) >> 31) &1
	sign_num2 = (((1<<31)-num2) >> 31) &1
	use_sign_num1 = sign_num1 ^ sign_num2
	use_sign_check = ((1<<1)-1)^use_sign_num1 # flip result of (sign_num1 ^ sign_num2), same as 'not (sign_num1 ^ sign_num2)'
	check = use_sign_num1*sign_num1 + use_sign_check*sign_diff
	return (1-check)*num1 + check*num2

#print(2**31 - 1)
#print(number_max(2**31 - 1,-5))


#---------------------------------------------------------------------------------------------------------
# 16.8 English Int: Given any integer, print an English phrase that describes the integer (e.g., "One
# Thousand, Two Hundred Thirty Four").

# we will assume max and min ints are from -2^31 + 1 to 2^31 - 1 (inclusive)

# Time Complexity: O(log10N) -> logbase10(N) where N=integer, Space Complexity: O(1)
def english_int(integer):
	dictionary = {  0:'zero', 1:'one', 2:'two', 3:'three', 4:'four', 5:'five', 6:'six', 7:'seven', 8:'eight', 9:'nine',
					10:'ten', 11:'eleven', 12:'twelve', 13:'thirteen', 14:'fourteen', 15:'fifteen', 16:'sixteen',
					17:'seventeen', 18:'eighteen', 19:'nineteen', 20:'twenty', 30:'thirty', 40:'forty', 50:'fifty', 
					60:'sixty', 70:'seventy', 80:'eighty', 90:'ninety', 100: 'hundred', 1000:'thousand', 
					1_000_000:'million', 1_000_000_000:'billion'
				 }
	if integer == 0: return dictionary[0]

	def helper(num, div):

		if not num: return None
		if num in dictionary:
			all.append(dictionary[num])
			return

		to_check = num - num%div
		if not to_check or not div in dictionary: return helper(num,div/10) 
		
		if to_check>=100:
			helper(to_check/div, div)
			to_check = div

		helper(to_check,div)
		helper(num%div,div/10)
			
	all = []
	if integer<0: 
		all.append("negative")
		integer*=-1
	helper(integer,1_000_000_000)
	return ' '.join(all)

# Time Complexity: O(log10N) -> logbase10(N) where N=integer, Space Complexity: O(1)
def english_int(integer):
	dictionary = {  0:'zero', 1:'one', 2:'two', 3:'three', 4:'four', 5:'five', 6:'six', 7:'seven', 8:'eight', 9:'nine',
					10:'ten', 11:'eleven', 12:'twelve', 13:'thirteen', 14:'fourteen', 15:'fifteen', 16:'sixteen',
					17:'seventeen', 18:'eighteen', 19:'nineteen', 20:'twenty', 30:'thirty', 40:'forty', 50:'fifty', 
					60:'sixty', 70:'seventy', 80:'eighty', 90:'ninety', 100: 'hundred', 1000:'thousand', 
					1_000_000:'million', 1_000_000_000:'billion'
				 }
	if integer == 0: return dictionary[0]

	def helper(num, multiplier):

		if not num: return None

		h = (num%1000)//100
		t_1 = num%100//10
		o = num%10

		if multiplier>1: all.append(dictionary[multiplier])
		if t_1 and o and (t_1*10 + o) in dictionary: all.append(dictionary[(t_1*10 + o)])
		else: 
			if o: all.append(dictionary[o])
			if t_1: all.append(dictionary[t_1*10])

		if h:all.extend([dictionary[100], dictionary[h]])

		helper(num//1000, multiplier*1000)
			
	all = []
	helper(abs(integer), 1)
	if integer<0: all.append("negative")
	return ' '.join(reversed(all))

#a = -34_050_111_139
#print(english_int(a))

#---------------------------------------------------------------------------------------------------------
# 16.9 Operations: Write methods to implement the multiply, subtract, and divide operations for integers.
# The results of all of these are integers. Use only the add operator.

# Lots of rnadom edge cases

def operations(int1, int2, op):

	def negate(integer):
		return ~integer+1 

	def abs_val(integer):
		if integer>=0: return integer
		return negate(integer)

	# Time Complexity: O(min(int1,int2)), Space Complexity: O(1)
	def multiply(int1, int2):
		if not int1 or not int2: return 0
		if int1< int2: int2,int1 = int1, int2
		neg = False
		if (int1<0 and int2>0) or (int1>0 and int2<0): neg = True
		sum = 0
		for _ in range(abs_val(int2)): sum+=abs_val(int1)
		return negate(sum) if neg else sum

	# Time Complexity: O(1), Space Complexity: O(1)
	def subtract(int1, int2):
		if not int1 and not int2: return 0
		if not int1: return multiply(-1, int2)
		if not int2: return int1
		return int1 + negate(int2)

	# Time Complexity: O(N) -> O(int1/int2) = O(int1/1) = O(n) where n = int1 , Space Complexity: O(1)
	def divide(int1, int2):

		if not int2: return None
		if not int1: return 0
		neg = False
		if (int1<0 and int2>0) or (int1>0 and int2<0): neg=True
		count = 0
		check = subtract(abs(int1), abs_val(int2))
		while check>=0:
			check = subtract(check, abs_val(int2))
			count+=1

		if check+abs_val(int2): return negate(count)-1 if neg else count
		return negate(count) if neg else count

	if op == 'm': return multiply(int1,int2)
	if op == 's': return subtract(int1, int2)
	if op == 'd': return divide(int1, int2)

#print(operations(-8,-3,'d'))

#---------------------------------------------------------------------------------------------------------
# 16.10 Living People: Given a list of people with their birth and death years, implement a method to
# compute the year with the most number of people alive. You may assume that all people were born
# between 1900 and 2000 (inclusive). If a person was alive during any portion of that year, they should
# be included in that year's count. For example, Person (birth= 1908, death= 1909) is included in the
# counts for both 1908 and 1909.

# Time Complexity: O(N*R) R=100 (range), Space Complexity: O(R) = O(1)
def living_people_brute(people):
	if not people: return 0
	years = [0 for _ in range(100+1)]
	max_year = 0
	max_count = 0
	for person in people:
		for i in range(person[0]-1900, person[1]-1900+1):
			years[i] +=1

	for i in range(len(years)):
		if years[i]>max_count:
			max_count = years[i]
			max_year = i+1900
	return max_year

#p = [(1900,1957),(1904,1985),(1945,1955),(1980,2000),(1955,1980)]
#print(living_people_brute(p))
	
# Time Complexity: O(N*logN), Space Complexity: O(N)
def living_people(people):
	if not people: return 0
	births = sorted([people[i][0] for i in range(len(people))])
	deaths = sorted([people[i][1] for i in range(len(people))])
	print(births)
	print(deaths)

	max_year = 0
	max_alive = 0
	curr_alive = 0
	j=0
	for i in range(len(people)):
		if births[i]<= deaths[j]: curr_alive +=1
		if curr_alive>max_alive:
			max_alive = curr_alive 
			max_year = i
		while births[i]>= deaths[j]:
			curr_alive -= 1
			j+=1
	
	return births[max_year],max_alive

# Time Complexity: O(N + R) where R= range, Space Complexity: O(R)  # Note this is only more optimal than above depending on the case
def living_people_optimal(people):

	max_alive, curr_alive, max_year = 0,0,0 
	alive = [0 for _ in range(100+2)]
	for i in range(len(people)): alive[people[i][0]-1900]+=1 # increment for each birth year
	for i in range(len(people)): alive[people[i][1]-1900+1] -=1 # decrement for each death year+1 (since death isnt reflected till next year)
	for i in range(len(alive)): #can also go to len(alive)-1 since last element in alive is for overflow to handle end of range death
		curr_alive+=alive[i]
		if curr_alive>max_alive:
			max_alive = curr_alive
			max_year = i

	return max_year+1900


#p = [(1900,1957),(1904,1985),(1945,1955),(1980,2000),(1955,1980)]
#print(living_people_optimal(p))

#---------------------------------------------------------------------------------------------------------
# 16.11 Diving Board: You are building a diving board by placing a bunch of planks of wood end-to-end.
# There are two types of planks, one of length shorter and one of length longer. You must use
# exactly K planks of wood. Write a method to generate all possible lengths for the diving board.

# Time Complexity: O(k^2), Space Complexity: O(k)
def driving_board(k):

	def helper(count,l,s):

		if (count,l,s) in cache: return
		if count == k: 
			all.add(f"{l}L + {s}S")
			return

		helper(count+1,l+1,s)
		helper(count+1,l,s+1)
		cache.add((count,l,s))

	all = set()
	cache = set()
	helper(0,0,0)
	print(all)

#driving_board(5)

# Time Complexity: O(k), Space Complexity: O(k)
def driving_board_optimal(k,l,s):
	if l == s: return l*k

	all = [] # or can use set
	for i in range(k+1):
		lo = k-i
		sum = lo*l + i*s
		all.append(sum)

	return all

#print(driving_board_optimal(5,3,7))

#---------------------------------------------------------------------------------------------------------
# 16.12 XML Encoding: Since XML is very verbose, you are given a way of encoding it where each tag gets
# mapped to a pre-defined integer value. The language/grammar is as follows:
# Element --> Tag Attributes END Children END
# Attribute --> Tag Value
# END --> 0
# Tag --> some predefined mapping to int
# Value --> string value

# Time Complexity: O(a + N*k) ? where a= # parent attributes N= num children k =most attr of any child, Space Complexity: O(a+ N*k) ?
def xml_encoding(element):
	
	def helper(root):

		res.append(root.code)

		for a in root.attributes:
			res.append(a.code)
			res.append(a.val)
		res.append(0)

		if root.val and root.val != "": res.append(root.val)
		else: 
			for c in root.children: helper(c)

		res.append(0)

	res = []
	helper(element)
	return " ".join(res)

#---------------------------------------------------------------------------------------------------------
# 16.13 Bisect Squares: Given two squares on a two-dimensional plane, find a line that would cut these two
# squares in half. Assume that the top and the bottom sides of the square run parallel to the x-axis.

# Time Complexity: O(1), Space Complexity: O(1)
def bisect_squares(s_a,s_b):
	'''We will return tuple of form (slope, y-intercept, x-intercept if vertical line)'''

	mid_a_x = (s_a[1][0] + s_a[0][0])/2
	mid_a_y = (s_a[1][1] + s_a[0][1])/2

	mid_b_x = (s_b[1][0] + s_b[0][0])/2
	mid_b_y = (s_b[1][1] + s_b[0][1])/2

	if (mid_b_x - mid_a_x) == 0: return (0,0,mid_a_x)

	slope = (mid_b_y - mid_a_y)/(mid_b_x - mid_a_x)
	y_intercept = mid_a_y - slope*mid_a_x

	return (slope, y_intercept, 0)

#a = ((0,1),(5,6))
#b = ((3,1),(8,6))
#a = ((0,0),(5,5))
#b = ((0,5),(5,10))
#a = ((0,0),(5,5))
#b = ((1,5),(6,10))
#print(bisect_squares(a,b))

#---------------------------------------------------------------------------------------------------------
# 16.14 Best Line: Given a two-dimensional graph with points on it, find a line which passes the most
# number of points.

# X X X
# X X X
# X X X
# X X X
# X X X

# Time Complexity: O(m*n), Space Complexity: O(m+n)
def best_line(graph):
	''' Returns tuple of form (number of points passed, line index, line type) '''
	rows = [0 for _ in range(len(graph))]
	cols = [0 for _ in range(len(graph[0]))]
	diags_r = [0 for _ in range(len(graph)+len(graph[0])-1)]
	diags_l = [0 for _ in range(len(graph)+len(graph[0])-1)]

	for i in range(len(graph)):
		for j in range(len(graph[0])):
			d_r = i-j
			d_l = i+j
			val = graph[i][j]
			if val and val == "X":
				rows[i]+=1
				cols[j]+=1
				diags_r[d_r]+=1
				diags_l[d_l]+=1

	max_line = (0,None,None)
	if max(rows)> max_line[0]: max_line = (max(rows), rows.index(max(rows)), 'r')
	if max(cols)> max_line[0]: max_line = (max(cols), cols.index(max(cols)), 'c')
	if max(diags_r)> max_line[0]: max_line = (max(diags_r), diags_r.index(max(diags_r)), 'd_r')
	if max(diags_l)> max_line[0]: max_line = (max(diags_l), diags_l.index(max(diags_l)), 'd_l')

	return max_line

#a = [["X",None,None],["X","X",None],["X",None,"X"],["X",None,None],["X",None,None]]
#print(best_line(a))

#---------------------------------------------------------------------------------------------------------
# 16.15 Master Mind: The Game of Master Mind is played as follows:
# The computer has four slots, and each slot will contain a ball that is red (R), yellow (Y), green (G) or
# blue (B). For example, the computer might have RGGB (Slot #1 is red, Slots #2 and #3 are green, Slot
# #4 is blue).
# You, the user, are trying to guess the solution. You might, for example, guess YRGB.
# When you guess the correct color for the correct slot, you get a "hit:' If you guess a color that exists
# but is in the wrong slot, you get a "pseudo-hit:' Note that a slot that is a hit can never count as a
# pseudo-hit.
# For example, if the actual solution is RGBY and you guess GGRR , you have one hit and one pseudohit
# Write a method that, given a guess and a solution, returns the number of hits and pseudo-hits.

# Time Complexity: O(N), Space Complexity: O(C) -> O(1) where N=# number of slots and C=# of colors
def master_mind(guess, solution):
	dict = {}
	hits, pseudo_hits = 0,0
	for i in range(len(solution)):
		if guess[i] == solution[i]: hits+=1
		else: # only add to dict if not a hit, otherwise we would need to decrement hit
			if solution[i] not in dict: dict[solution[i]] =1
			else:dict[solution[i]]+=1
	for i in range(len(guess)): 
		if guess[i] in dict:
			pseudo_hits+=1
			dict[guess[i]]-=1
			if dict[guess[i]]==0: dict.pop(guess[i])

	return (hits,pseudo_hits)
'''		
g = 'GGRR'
s = 'RGRY'
print(master_mind(g,s))
'''

#---------------------------------------------------------------------------------------------------------
# 16.16 Sub Sort: Given an array of integers, write a method to find indices m and n such that if you sorted
# elements m through n , the entire array would be sorted. Minimize n - m (that is, find the smallest
# such sequence).
# EXAMPLE
# Input: 1, 2, 4, 7, 10, 11, 7, 12, 6, 7, 16, 18, 19
# Output: (3, 9)

# Time Complexity: O(N^2), Space Complexity: O(1)
def sub_sort_non_optimal(arr):
	if len(arr)==1: return (0,0)
	m,n = float('inf'),0
	i,j = 0, len(arr)-1
	max = arr[0]
	for i in range(1,len(arr)):
		if arr[i] < arr[i-1] or arr[i] <= max:
			n=i
			for j in range(n):
				if arr[j] > arr[i]:
					m = min(m,j)
					max = arr[j]
					break
	return (m,n)
'''
a = [1, 2, 4, 7, 10, 11, 7, 12, 6, 7, 16, 18, 19]
b = [1, 2, 4, 7, 10, 11, 12, 13, 3, 15, 2, 19, 6]
print(sub_sort_non_optimal(b))'''

# Time Complexity: O(N), Space Complexity: O(1)
def sub_sort(arr):

	if not arr: return None
	if len(arr)==1: return (0,0)

	min_v, max_v = float('inf'), float('-inf')
	i,j=0,len(arr)-1
	while i<len(arr)-1 and arr[i+1]>=arr[i]: i+=1
	while j>0 and arr[j-1]<= arr[j]: j-=1
	for a in range(j): max_v = max(max_v, arr[a])
	for b in range(i+1,len(arr)): min_v = min(min_v, arr[b])

	m,n=0,len(arr)-1
	while arr[m]<= min_v: m+=1
	while arr[n]>= max_v: n-=1

	return (m,n)

'''
a = [1, 2, 4, 7, 10, 11, 7, 12, 6, 7, 16, 15, 19]
b = [1, 2, 4, 7, 10, 11, 12, 13, 3, 15, 2, 19, 21]
b = [21, 2, 4, 7, 10, 11, 12, 13, 3, 15, 2, 19, 23]
print(sub_sort(b))'''

#---------------------------------------------------------------------------------------------------------
# 16.17 Contiguous Sequence: You are given an array of integers (both positive and negative). Find the
# contiguous sequence with the largest sum. Return the sum.
# EXAMPLE
# Input: 2, -8, 3, -2, 4, -10
# Output: 5 ( i. e • , { 3, -2, 4} )

# Time Complexity: O(N), Space Complexity: O(N)
def contiguous_sequence_dp(arr):
	if not arr: return None
	dp = [0 for _ in range(len(arr))]
	dp[0] = arr[0]
	for i in range(1,len(arr)): dp[i] = max(dp[i-1]+arr[i], arr[i])
	return max(dp)
'''
a = [2, -8, 3, -2, 4, -10]
print(contiguous_sequence_dp(a))'''


# this way seems useless, O(N^2)? and O(1)
def contiguous_sequence(arr):

	def helper(val,idx):
		global max_v
		if idx>= len(arr): return None
		if val+arr[idx]>max_v: max_v=val+arr[idx]
		helper(val+arr[idx], idx+1)

	global max_v
	max_v = float('-inf')
	for i in range(len(arr)): helper(0, i)
	return max_v
'''
a = [2, -8, 3, -2, 4, -10]
print(contiguous_sequence(a))'''

# Time Complexity: O(N), Space Complexity: O(1)
def contiguous_sequence_dp_optimized(arr):
	if not arr: return None
	max_v = arr[0]
	for i in range(1,len(arr)):
		arr[i] = max(arr[i], arr[i-1]+arr[i]) 
		max_v = max(arr[i], max_v) 
	return max_v
'''
a = [2, -8, 3, -2, 4, -10]
print(contiguous_sequence_dp_optimized(a))'''

#---------------------------------------------------------------------------------------------------------
# 16.18 Pattern Matching: You are given two strings, pattern and value. The pattern string consists of
# just the letters a and b, describing a pattern within a string. For example, the string catcatgocatgo
# matches the pattern aabab (where cat is a and go is b). It also matches patterns like a, ab, and b.
# Write a method to determine if value matches pattern.

# Time Complexity: O(N^2), Space Complexity: O(1)
def pattern_matching(pattern, value):

	def helper(pattern, value, a, b):
		
		if not pattern and not value : return True
		if not pattern or not value: return False

		for i in range(1,len(value)+1):
			print(f"pattern: {pattern}, value: {value}, a:{a}, b:{b}, i:{i}")
			v = value[0:i]
			p = pattern[0]
			check_1,check_2,check_3,check_4 = False, False, False, False
			if p=="a" and not a: check_1 = helper(pattern[1:], value[i:], a+v, b)
			if p=="a" and v==a: check_2 = helper(pattern[1:], value[i:], a, b)
			if p=="b" and not b: check_3 = helper(pattern[1:], value[i:], a, b+v)
			if p=="b" and v==b: check_4 = helper(pattern[1:], value[i:], a, b)
			if check_1 or check_2 or check_3 or check_4: return True

		return False


	return helper(pattern, value, "", "")

#print(pattern_matching("aabab", "catcatgocatgo"))
#print(pattern_matching("ababbba", "abc"))

#---------------------------------------------------------------------------------------------------------
# 16.19 Pond Sizes: You have an integer matrix representing a plot of land, where the value at that location
# represents the height above sea level. A value of zero indicates water. A pond is a region of water
# connected vertically, horizontally, or diagonally. The size of the pond is the total number of
# connected water cells. Write a method to compute the sizes of all ponds in the matrix.
# EXAMPLE
# Input:
# 0 2 1 0
# 0 1 0 1
# 1 1 0 1
# 0 1 0 1
# Output: 2, 4, 1 (in any order)

# Time Complexity: O(M*N) -> O(N^2), Space Complexity: O(1)
def pond_sizes(land):

	for i in range(len(land)):
		for j in range(len(land[0])):
			if not land[i][j]:
				land[i][j]-=1
				if i-1>=0 and land[i-1][j]<0:
					land[i][j] +=land[i-1][j]
					land[i-1][j]=1
				if j-1>=0 and land[i][j-1]<0:
					land[i][j] += land[i][j-1]
					land[i][j-1]=1
				if i-1>=0 and j-1>0 and land[i-1][j-1]<0:
					land[i][j] += land[i-1][j-1]
					land[i-1][j-1]=1
				if i-1>=0 and j+1<len(land[0]) and land[i-1][j+1]<0:
					land[i][j] += land[i-1][j+1]
					land[i-1][j+1]=1
	
	res = []
	for i in range(len(land)):
		for j in range(len(land[0])):
			if land[i][j]<0: res.append(abs(land[i][j]))
	return res
'''
a = [[0, 2, 1, 0],[0, 1, 0, 1],[1, 1, 0, 1],[0, 1, 0, 1]]
print(pond_sizes(a))'''

# Time Complexity: O(M*N) -> O(N^2), Space Complexity: O(1)
def pond_sizes_alternate_method(land):

	def checkSurrounding(row,col):
		if row<0 or col<0 or row>len(land)-1 or col>len(land[0])-1 or land[row][col] !=0: return 0
		land[row][col] = -1
		pond_size = 1
		for i in range(-1,2):
			for j in range(-1,2): pond_size+= checkSurrounding(row+i,col+j)
		return pond_size

	ponds = []
	for i in range(len(land)):
		for j in range(len(land[0])):
			if not land[i][j]: ponds.append(checkSurrounding(i,j))
	return ponds

'''
a = [[0, 2, 1, 0],[0, 1, 0, 1],[1, 1, 0, 1],[0, 1, 0, 1]]
print(pond_sizes_alternate_method(a))'''

#---------------------------------------------------------------------------------------------------------
# 16.20 T9: On old cell phones, users typed on a numeric keypad and the phone would provide a list of words
# that matched these numbers. Each digit mapped to a set of O - 4 letters. Implement an algorithm
# to return a list of matching words, given a sequence of digits. You are provided a list of valid words
# (provided in whatever data structure you'd like). The mapping is shown in the diagram below:
#
#   1  |  2  |  3
#      | abc | def         
# -----------------
#   4  |  5  |  6
#  ghi | jkl | mno
# -----------------
#   7  |  8  |  9
# pqrs | tuv | xyz
# -----------------
# EXAMPLE
# Input: 8733
# Output: tree, used

# Time Complexity: O(4^N) where N in number of digits in input, Space Complexity: O(D) where D is size of dictionary
def t9_naive(input, dictionary):

	if not input: return None
	num_map = {1:'', 2:'abc', 3:'def', 4:'ghi', 5:'jkl', 6:'mno', 7:'pqrs', 8:'tuv', 9:'xyz'}

	def helper(idx, word):
		if idx == len(input):
			if word in dictionary: res.append(word)
			return
		
		for c in num_map[input[idx]]: helper(idx+1, word+c)

	res = []
	helper(0, '')
	return res
'''
d = {'burt', 'tree', 'used'}
inp = [8,7,3,3]
print(t9_naive(inp,d))'''
	
# Better to use trie, for dictionary
# Time Complexity: unsure O(4*N), Space Complexity: O(D) where where D is size of dictionary
def t9(input, dictionary_trie):

	if not input: return None
	num_map = {1:'', 2:'abc', 3:'def', 4:'ghi', 5:'jkl', 6:'mno', 7:'pqrs', 8:'tuv', 9:'xyz'}

	def helper(idx, dict_trie, word):
		if idx == len(input):
			if -1 in dict_trie: res.append(word)
			return

		for c in num_map[input[idx]]: 
			if c in dict_trie: helper(idx+1, dict_trie[c], word+c)

	res = []
	helper(0, dictionary_trie, '')
	return res
'''
d_trie = {'a':{-1:{}}, 'b':{-1:{}}, 't':{'r':{'e':{'e':{-1:{}}, 'f':{'g':{}}}}}, 'u':{'s':{'e':{'d':{-1:{}}}}}}
inp = [8,7,3,3]
print(t9(inp,d_trie))'''

# Time Complexity: O(D), Space Complexity: O(D), Look portion (without transforming dictionary) is O(1)
def t9_optimal(input, dictionary):

	if not input: return None
	num_map = {1:'', 2:'abc', 3:'def', 4:'ghi', 5:'jkl', 6:'mno', 7:'pqrs', 8:'tuv', 9:'xyz'}
	flipped_map = {a:k for k,v in num_map.items() for a in v}
	dict_to_t9 = {}
	for word in dictionary:
		t9_repr = 0
		for c in word: t9_repr = t9_repr*10 + flipped_map[c]
		if t9_repr not in dict_to_t9: dict_to_t9[t9_repr] = [word]
		else: dict_to_t9[t9_repr].append(word)

	return dict_to_t9[input]

'''
d = ['burt', 'tree', 'used']
inp = 8733
print(t9_optimal(inp, d))'''

#---------------------------------------------------------------------------------------------------------
# 16.21 Sum Swap: Given two arrays of integers, find a pair of values (one value from each array) that you
# can swap to give the two arrays the same sum.
# EXAMPLE
# Input:{4, 1, 2, 1, 1, 2} and {3, 6, 3, 3}
# Output: {1, 3}

# Time Complexity: O(NlogN) where N = max_length(arr1, arr2) or O(AlogA + BlogB), Space ComplexityL O(1)
def sum_swap(arr1, arr2):

	arr1.sort()
	arr2.sort()
	sum_1 = sum(arr1)
	sum_2 = sum(arr2)
	diff = sum_1 - sum_2

	i,j=0,0
	while i < len(arr1) and j < len(arr2):
		if 2*arr1[i] - 2*arr2[j] == diff: return (arr1[i],arr2[j]) # this is a measure of our new difference
		if 2*arr1[i] - 2*arr2[j] < diff: j+=1 # want to converge to diff, increment j (making our new difference larger)
		else: i+=1 # want to converge to diff, increment i (making our new difference smaller)

	return None
'''
a = [4, 1, 2, 1, 1, 2] #11
b = [2, 5, 3, 3] #13
print(sum_swap(a,b))'''

# Time Complexity: O(N) or O(A + B) where N = max(A , B) where A,B = length arr1 and arr2, respectively, Space Complexity: O(A)
def sum_swap_optimal_time(arr1, arr2):

	sum_1 = sum(arr1)
	sum_2 = sum(arr2)
	diff = sum_1 - sum_2
	set_1 = set(arr1)
	for num in arr2:
		to_find = (diff + 2*num)/2
		if to_find in set_1: return (int(to_find), num) # int wrapper only to prettify result
	return None

'''
a = [4, 1, 2, 1, 1, 2] #11
b = [2, 5, 3, 3] #13
print(sum_swap_optimal_time(a,b))'''

#---------------------------------------------------------------------------------------------------------
# 16.22 Langton's Ant: An ant is sitting on an infinite grid of white and black squares. It initially faces right.
# At each step, it does the following:
# (1) At a white square, flip the color of the square, turn 90 degrees right (clockwise), and move forward
# one unit.
# (2) At a black square, flip the color of the square, turn 90 degrees left (counter-clockwise), and move
# forward one unit.
# Write a program to simulate the first K moves that the ant makes and print the final board as a grid.
# Note that you are not provided with the data structure to represent the grid. This is something you
# must design yourself. The only input to your method is K. You should print the final grid and return
# nothing. The method signature might be something like void printKMoves ( int K).

# B B B B B
# B W B B B
# B B W B B
# B B B B W
# B B B B B

class Board:
	def __init__(self, whites, ant_pos, top_left, bottom_right):
		self.top_left= top_left
		self.bottom_right = bottom_right
		self.ant = (ant_pos[0],ant_pos[1],'r')
		self.whites = set()
		for w in whites: self.whites.add(w)
		self.next_o_white= {'r':'d', 'd':'l', 'l':'u', 'u':'r'}
		self.next_o_black= {'r':'u', 'u':'l', 'l':'d', 'd':'r'}

	def sim_step(self):
		ant_pos = (self.ant[0], self.ant[1])
		if  ant_pos in self.whites:
			self.whites.remove(ant_pos)
			self.move_forward(ant_pos, self.next_o_white[self.ant[2]])
		else:
			self.whites.add(ant_pos)
			self.move_forward(ant_pos, self.next_o_black[self.ant[2]])

	def move_forward(self, start, direction):
		row, col = start[0], start[1]
		if direction=='r': col+=1
		elif direction=='l': col-=1
		elif direction=='d': row+=1
		elif direction=='u': row-=1
		row = min(row, self.bottom_right[0])
		row = max(row, self.top_left[0])
		col = min(col, self.bottom_right[1])
		col = max(col, self.top_left[1])
		self.ant = (row,col,direction)

	# Our dict implementation makes growing the grid much simpler, than would an array
	def grow_grid(self, new_whites, new_top_left, new_bottom_right):
		''' Can implement this if needed. We just update our new limit positions, and add new whites to the white dict'''
		for w in new_whites: self.whites.add(w)
		self.top_left= new_top_left
		self.bottom_right = new_bottom_right

	def print_board(self):
		for i in range(self.bottom_right[0]+1):
			for j in range(self.bottom_right[1]+1):
				if (i,j) in self.whites: print (f" W", end="")
				else: print(f" B", end="")
			print("")

# Time Complexity: O(M*N), Space Complexity: O(M*N) --> only if entire board is white, usually much less
def langtons_ant(k):

	board = Board([(1,1), (2,2), (3,4)], (2,2), (0,0), (4,4))
	print("BEFORE: ")
	board.print_board()
	for _ in range(k): board.sim_step()
	print()
	print("AFTER: ")
	board.print_board()


#langtons_ant(6)

#---------------------------------------------------------------------------------------------------------
# 16.23 Rand7 from Rand 5: Implement a method rand7() given rand5(). That is, given a method that
# generates a random number between O and 4 (inclusive), write a method that generates a random
# number between O and 6 (inclusive).

# rand5 -> P = 1/5 , rand5 + rand5 -> P = 1/25 
# a  b  r   a  b  r    p(0) = 1/25  p_mod7(0) = 3/25
# 0  0  0   3  0  3    p(1) = 2/25  p_mod7(1) = 3/25
# 0  1  1   3  1  4    p(2) = 3/25  p_mod7(2) = 3/25
# 0  2  2   3  2  5    p(3) = 4/25  p_mod7(3) = 4/25
# 0  3  3   3  3  6    p(4) = 5/25  p_mod7(4) = 5/25
# 0  4  4   3  4  7    p(5) = 4/25  p_mod7(5) = 4/25
# 1  0  1   4  0  4    p(6) = 3/25  p_mod7(6) = 3/25
# 1  1  2   4  1  5    p(7) = 2/25
# 1  2  3   4  2  6    p(8) = 1/25
# 1  3  4   4  3  7
# 1  4  5   4  4  8
# 2  0  2
# 2  1  3
# 2  2  4
# 2  3  5
# 2  4  6

# rand5 -> P = 1/5 , 5*rand5 + rand5 -> P = 1/25 (uniformly distributed), 4*rand5 + rand5 is not etc. etc.
# a  b  r    a  b  r     p(0) = 1/25  p_mod7(0) = 4/25  p_mod7_<21(0) = 3/21 = 1/7
# 0  0  0    3  0  15    p(1) = 1/25  p_mod7(1) = 4/25  p_mod7_<21(1) = 3/21 = 1/7
# 0  1  1    3  1  16    p(2) = 1/25  p_mod7(2) = 4/25  p_mod7_<21(2) = 3/21 = 1/7
# 0  2  2    3  2  17    p(3) = 1/25  p_mod7(3) = 4/25  p_mod7_<21(3) = 3/21 = 1/7
# 0  3  3    3  3  18    p(4) = 1/25  p_mod7(4) = 3/25  p_mod7_<21(4) = 3/21 = 1/7
# 0  4  4    3  4  19    p(5) = 1/25  p_mod7(5) = 3/25  p_mod7_<21(5) = 3/21 = 1/7
# 1  0  5    4  0  20    p(6) = 1/25  p_mod7(6) = 3/25  p_mod7_<21(6) = 3/21 = 1/7
# 1  1  6    4  1  21    p(7) = 1/25
# 1  2  7    4  2  22    p(x) = 1/25
# 1  3  8    4  3  23
# 1  4  9    4  4  24
# 2  0  10
# 2  1  11
# 2  2  12
# 2  3  13
# 2  4  14

# Non-deterministic, Time Complexity: O(inf) (however tiny prob of happening), Space Complexity: O(1)
def rand7():
	while True:
		rand_num = 5*rand5() + rand5() # 0 ... 24
		if rand_num <21:
			return rand_num % 7
def rand5():
	return random.randint(0,4)

#print(rand7())

#---------------------------------------------------------------------------------------------------------
# 16.24 Pairs with Sum: Design an algorithm to find all pairs of integers within an array which sum to a
# specified value.

# Time Complexity: O(N), Space Complexity: O(N)
def pairs_with_sum(arr, sum_val):
	
	dict_1 = {}
	for a in arr:
		if a not in dict_1: dict_1[a] = 1
		else: dict_1[a]+=1

	pairs = []
	for a in arr:
		if sum_val - a in dict_1 and dict_1[sum_val - a] and dict_1[a]: 
			pairs.append((a, sum_val-a))
			dict_1[sum_val - a]-=1
			dict_1[a]-=1

	return pairs
'''
a = [2,1,9,3,4,5,2,8,19,0,5,5,4,0,1]
print(pairs_with_sum_naive(a, 5))'''

# Time Complexity: O(NlogN), Space Complexity: O(1)
def pairs_with_sum_space_optimized(arr, sum_val):

	arr.sort()

	i,j=0,len(arr)-1
	pairs=[]
	while i<j:
		if (arr[i] + arr[j]) == sum_val:
			pairs.append((arr[i],arr[j]))
			i+=1
			j-=1
		elif (arr[i] + arr[j]) > sum_val: j-=1
		else: i+=1
	return pairs
'''
a = [2,1,9,3,4,5,3,2,2,8,19,0,5,5,4,0,1]
print(pairs_with_sum_space_optimized(a, 6))'''

#---------------------------------------------------------------------------------------------------------
# 16.25 LRU Cache: Design and build a "least recently used" cache, which evicts the least recently used item.
# The cache should map from keys to values (allowing you to insert and retrieve a value associated
# with a particular key) and be initialized with a max size. When it is full, it should evict the least
# recently used item. You can assume the keys are integers and the values are strings.

class LRUCacheNaive:
	def __init__(self, size):
		self.size = size
		self.lru = {}
		self.queue = []

	def add(self, key, item):
		if len(self.lru) != self.size:
			self.queue.append(key)
			self.lru[key] = item
		else:
			self.lru.pop(self.queue[0])
			self.queue.pop(0)
			self.queue.append(key)
			self.lru[key] = item

	def get(self, key):
		if key not in self.lru: return None
		self.queue.pop(key)
		self.queue.append(key)
		return self.lru[key]

class ListNode:
	def __init__(self, key, val):
		self.key = key
		self.val = val
		self.next = None
		self.prev = None

# Time Complexity: O(1) all ops, Space Complexity: O(N)
class LRUCache:
	def __init__(self, size):
		self.size = size
		self.lru = {}
		self.head = None
		self.tail = None

	# Time Complexity: O(1)
	def addToBegList(self, node):
		if not self.head: self.head = self.tail = node
		else:
			self.head.prev = node
			node.next = self.head
			self.head = node
	
	# Time Complexity: O(1)
	def removeFromEndList(self):
		tail = self.tail
		if self.tail == self.head:
			self.tail = self.head = None
			return tail

		self.tail = self.tail.prev
		self.tail.next = None
		tail.prev = None

		return tail
	
	# Time Complexity: O(1)
	def add(self, key, item):
		new_node = ListNode(key,item)
		if len(self.lru) != self.size:
			self.addToBegList(new_node)
			self.lru[key] = new_node
		else:
			least = self.removeFromEndList()
			self.addToBegList(new_node)
			self.lru.pop(least.key)
			self.lru[key] = new_node
		self.size+=1
	
	# Time Complexity: O(1)
	def get(self, key):
		if key not in self.lru: return None
		if self.head.key == key: return self.lru[key].val
		node = self.lru[key]
		prev = node.prev
		next = node.next
		prev.next = next
		if next: next.prev = prev
		node.prev = None
		node.next = None
		self.addToBegList(node)

		return self.lru[key].val

#---------------------------------------------------------------------------------------------------------
# 16.26 Calculator: Given an arithmetic equation consisting of positive integers,+,-,* and/ (no parentheses),
# compute the result.
# EXAMPLE
# Input: 2*3+5/6*3+15
# Output: 23.5

# Time Complexity: O(N), Space Complexity: O(N)
def calculator(input):
	'''METHOD IS WRONG, NEED TO TWEAK, SEE BELOW VERSION WHICH IS CORRECT'''
	def helper(idx):
		if not input: return None
		to_do, num = [],0
		while idx<len(input) and (input[idx].isdigit() or input[idx] in {"*", "/"}): 
			if input[idx].isdigit() : num = num*10 + int(input[idx])
			else: 
				to_do.append(num)
				to_do.append(input[idx])
				num = 0
			idx+=1
		if num: to_do.append(num)
		print(f"todo: {to_do}")
		res = to_do[0]
		for i in range(1,len(to_do)-1,2):
			if to_do[i]=="*": res = res * to_do[i+1]
			if to_do[i]=="/": res = res / to_do[i+1]
		num = res
		if idx<len(input) and input[idx]== "+": return num + helper(idx+1)
		if idx<len(input) and input[idx]== "-": return num - helper(idx+1)
		return num
	
	return helper(0)
'''
a = "2*3+5/6*3+15"
print(calculator(a))'''

# Time Complexity: O(N), Space Complexity: O(N)
def calculator_cleaner(input):
	if not input or len(input)<3: return input

	cleaned_input = []
	num = 0
	for i in range(len(input)):
		if input[i].isdigit(): num = num*10 + int(input[i])
		else: 
			cleaned_input.append(num)
			cleaned_input.append(input[i])
			num = 0
	cleaned_input.append(num)

	total, curr = 0,cleaned_input[0]
	nextOp = None
	idx = 1
	while idx<len(cleaned_input)-1:
		
		if idx+2 < len(cleaned_input): nextOp = cleaned_input[idx+2]
		
		if cleaned_input[idx] == "*": curr = curr * cleaned_input[idx+1]
		elif cleaned_input[idx] == "/": curr = curr / cleaned_input[idx+1]
		elif cleaned_input[idx]=="+": curr = curr + cleaned_input[idx+1]
		elif cleaned_input[idx]=="-": curr = curr - cleaned_input[idx+1]
		
		if not nextOp or nextOp in {"+","-"}: 
			total += curr
			curr = 0
			nextOp= None
		idx += 2

	total += curr

	return total

'''
a = "2*3+5/6*3+15"
b = "2-6-7*8/2+5"
print(calculator_cleaner(b))'''

