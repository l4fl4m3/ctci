#---------------------------------------------------------------------------------------------------------
# 10.1 Sorted Merge: You are given two sorted arrays, A and B, where A has a large enough buffer at the
# end to hold B. Write a method to merge B into A in sorted order.

#Time Complexity: O(max(A,B)) where A and B are # elements in a and b respectively, Space Complexity: O(1)
def sorted_merge(a,b):

	i,idx,j = len(a)-1, len(a)-1, len(b)-1
	while a[i] is None: i-=1
	while i>=0 and j>=0:
		if a[i]>b[j]: 
			a[idx] = a[i]
			i-=1
		else:
			a[idx] = b[j]
			j-=1
		idx-=1

	while i>=0:
		a[idx] = a[i]
		i-=1
		idx-=1
	while j>=0:
		a[idx] = a[j]
		j-=1
		idx-=1

	return a
def sorted_merge_cleaner(a,b):

	idx,j = len(a)-1, len(b)-1
	i = idx-j-1
	while j>=0:
		if a[i]>b[j]: 
			a[idx] = a[i]
			i-=1
		else:
			a[idx] = b[j]
			j-=1
		idx-=1	
	return a
a= [1,2,5,9,23,45,None,None,None,None,None,None]
b= [2,6,18,28,32,56]

#print(sorted_merge_cleaner(a,b))

#---------------------------------------------------------------------------------------------------------
# radix sort implementation
#Time Complexity: O(N) O(d(N+k)), Space Complexity: O(N)
def radix_sort(arr):

	def counting_sort(exp):
		temp = [0 for _ in range(len(arr))]
		count = [0 for _ in range(10)]
		for i in range(len(arr)):
			idx = arr[i]//exp
			count[idx%10] +=1
		for i in range(1,10):
			count[i]+=count[i-1]
		for i in range(len(arr)-1,-1,-1):
			idx = arr[i]//exp
			temp[count[idx%10]-1] = arr[i]
			count[idx%10] -=1
		for i in range(len(arr)):
			arr[i] = temp[i]

	maxN = max(arr)
	exp = 1
	while maxN:
		counting_sort(exp)
		exp = exp*10
		maxN=maxN//10

#arr = [254,33, 656, 33, 2, 590, 99]
#radix_sort(arr)
#print(arr)

#---------------------------------------------------------------------------------------------------------
# 10.2  Group Anagrams: Write a method to sort an array of strings so that all the anagrams are next to
# each other.

# Time Complexity: O(N*klog(k)) where k = max string length, Space Complexity: O(N)
def group_anagrams(strings):

	sord = {}
	for i in range(len(strings)):
		s = ''.join(sorted(strings[i]))
		if s in sord: sord[s].append(strings[i])
		else: sord[s] = [strings[i]]
		strings[i] = s
	i = 0
	for k,v in sord.items():
		for word in v:
			strings[i] = word
			i+=1

#s = ['ayz', 'bkj','tfv', 'zya', 'vft']
#group_anagrams(s)
#print(s)

#---------------------------------------------------------------------------------------------------------
# 10.3 Search in Rotated Array: Given a sorted array of n integers that has been rotated an unknown
# number of times, write code to find an element in the array. You may assume that the array was
# originally sorted in increasing order.
# EXAMPLE
# Input: find 5 in {15, 16, 19, 20, 25, 1, 3, 4, 5, 7, 10, 14}
# Output: 8 (the index of 5 in the array)

# Time Complexity: O(logN), Space Complexity: O(1)
def search_in_rotated_array(arr, element):
	
	i,j = 0, len(arr)-1
	while i<=j:
		mid = (i+j)//2
		if arr[mid] == element: return mid
		# if left side is ordered correct

		if arr[i]<=arr[mid]:
			if arr[i]<=element<=arr[mid]: j = mid-1
			else: i = mid+1
		# if right side is ordered correct
		else:
			if arr[mid]<=element<=arr[j]: i = mid+1
			else: j = mid-1

	
	return None
'''
a=[16, 19, 20, 25, 1, 3, 4, 5, 7, 10, 14, 15]
b=[14, 15, 16, 19, 20, 25, 1, 3, 4, 5, 7, 10]
c=[15, 16, 19, 20, 25, 1, 3, 4, 5, 7, 10, 14]
d=[10, 14, 15, 16, 19, 20, 25, 1, 3, 4, 5, 7]
print(search_in_rotated_array(a,5))
print(search_in_rotated_array(b,16))
print(search_in_rotated_array(c,15))
print(search_in_rotated_array(d,25))
'''

#---------------------------------------------------------------------------------------------------------
# 10.4 Sorted Search, No Size: You are given an array-like data structure Listy which lacks a size
# method. It does, however, have an elementAt (i) method that returns the element at index i in
# 0(1) time. If i is beyond the bounds of the data structure, it returns -1. (For this reason, the data
# structure only supports positive integers.) Given a Listy which contains sorted, positive integers,
# find the index at which an element x occurs. If x occurs multiple times, you may return any index.

# Time Complexity: O(logN), Space Complexity: O(1)
def sorted_search_no_size(arr,x):

	if not arr: return None
	def binary_search_mod(l,h,s):
		if h<l: return None

		mid = (l+h)//2
		if elementAt(mid)==s: return mid
		if elementAt(mid)>s or elementAt(mid)==-1: return binary_search_mod(l,mid-1,s)
		else: return binary_search_mod(mid+1,h,s)

	def elementAt(idx):
		if idx>len(arr)-1 : return -1
		return arr[idx]

	length=1
	while elementAt(length-1) != -1:
		length = length*2

	return binary_search_mod(0,length-1,x)

#a=[1,1,1,1,2]
#a = [1,1,1,1,4,7,9,15,27,33,43,53,63,73,83,101,123,125,127,128,129,130,131,132,133,134,135,136,137,138,139,140,140,140,140,141,141,500]
#print(len(a))
#print(sorted_search_no_size(a,2))

#---------------------------------------------------------------------------------------------------------
# 10.5 Sparse Search: Given a sorted array of strings that is interspersed with empty strings, write a
# method to find the location of a given string.
# EXAMPLE
# Input: ball, {"at", "", "", "", "ball", "", "", "car", "", "", "dad", "", ""}
# Output: 4

#Time Complexity: O(N) impossible for less than O(N) in worst case, Space Complexity: O(1)
def sparse_search(arr,string):

	def binary_search_mod(l,h,s):
		if h<l: return None
		mid=(l+h)//2
		if arr[mid]=="":
			new_mid_l, new_mid_r = mid-1, mid+1
			while True:
				if new_mid_l<l and new_mid_r>h: return None
				if new_mid_l>=l and arr[new_mid_l]!="":
					mid = new_mid_l
					break
				if new_mid_r<=h and arr[new_mid_r]!="":
					mid = new_mid_r
					break
				new_mid_l-=1
				new_mid_r+=1
			'''
			while arr[new_mid_l]=="" and new_mid_l>l: new_mid_l-=1
			while arr[new_mid_r]=="" and new_mid_r<h: new_mid_r+=1
			if arr[new_mid_l]:
				if arr[new_mid_l]==s: 
					return new_mid_l
				if arr[new_mid_l]>s: return binary_search_mod(l,new_mid_l-1,s)
			if arr[new_mid_r]:
				if arr[new_mid_r]==s: return new_mid_r
				if arr[new_mid_r]<s: return binary_search_mod(new_mid_r+1,h,s)
			return None
			'''
		if arr[mid]==s: return mid
		if arr[mid]>s: return binary_search_mod(l,mid-1,s)
		if arr[mid]<s: return binary_search_mod(mid+1,h,s)
	
	if string=="": return None
	return binary_search_mod(0,len(arr)-1,string)
		

#a = ["at", "", "", "", "ball", "", "", "car", "", "", "dad", "", ""]
#print(sparse_search(a,""))

#---------------------------------------------------------------------------------------------------------
# 10.6 Sort Big File: Imagine you have a 20 GB file with one string per line. Explain how you would sort
# the file.

# Ans: Split and parallelize sort process so it can run on available memory across multiple machines, 
# then merge the sorted files, basically divide and conquer, but with parallelization. You can also just split into sizes
# that are able to be held in local ram, sort, and then merge all sorted files, but this will take a long time. Faster to
# do it via parallelization, but this will obviously require a bit more work.

#---------------------------------------------------------------------------------------------------------
# 10.7 Missing Int: Given an input file with four billion non-negative integers, provide an algorithm to
# generate an integer that is not contained in the file. Assume you have 1 GB of memory available for
# this task.
# FOLLOW UP
# What if you have only 1O MB of memory? Assume that all the values are distinct and we now have
# no more than one billion non-negative integers.

# assuming input is int (uses 32 bits), in 32 bits we have 2^32 distinct ints (this is >4 bill) and 2^31 non-negative ints (<4 bill), 
# therefore there are dups. 1Gb memory = 8 billion bits, so enough memory to map.
# So we can use bitmap/bitarray to, and go through all four billion ints from file, and and set bitarray index equivalent to number to
# true for each number, then we we iterate through the bitarray from the beginning, and the first 0(false) value we find is a valid
# integer that is not in the file

# Time Complexity: O(N), Space Complexity: O(N)
def missing_int(input_file):

	num_bits = 4_000_000_000
	bit_vector = 1<< num_bits
	for num in input_file:
		bit_vector  = bit_vector| 1<<num
	
	for i in range(num_bits):
		if not bit_vector & 1<<i: return i

#test_file=[0,1,2,3,4,5,6,7,8,9,10,12]
#print(missing_int(test_file))

#can divide 4bill by into x blocks of 1000, we get 4 mb of space used. Therefore we will have 1000 buckets
def missing_int_followup(input_file):
	
	num_bits = 4_000_000_000
	one_mb_bits = 1_000_000

	i=0
	for num in input_file:
		bit_vector  = bit_vector| 1<<num
	
		for i in range(num_bits):
			if not bit_vector & 1<<i: return i

#test_file=[0,1,2,3,4,5,6,7,8,9,10,12]
#print(missing_int_followup(test_file))

#---------------------------------------------------------------------------------------------------------
# 10.8 Find Duplicates: You have an array with all the numbers from 1 to N, where N is at most 32,000. The
# array may have duplicate entries and you do not know what N is. With only 4 kilobytes of memory
# available, how would you print all duplicate elements in the array?

# Time Complexity: O(N), Space Complexity: O(N)
def find_duplicates(arr):
	four_kb_bits = 8*4*(2**10) # this is greater than 32000
	bit_vector = 1<<four_kb_bits

	for n in arr:
		check = bit_vector & 1<<n
		if check: print(n)
		else: bit_vector = bit_vector | 1<<n

#test_file=[0,1,2,3,4,5,5,6,7,8,9,10,10,11,11,12]
#ind_duplicates(test_file)

#---------------------------------------------------------------------------------------------------------
# 10.9 Sorted Matrix Search: Given an M x N matrix in which each row and each column is sorted in
# ascending order, write a method to find an element.
#
# 1  2  3  4
# 5  6  7  8
# 9  10 11 12
# 13 14 15 16

# Could do binary search on each row -> O(m*logn)

# Time Complexity: O(m+n), Space Complexity: O(1)
def sorted_matrix_search(m,s):

	def helper(r,c,s):

		if c<0 or r>len(m)-1: return None
		if m[r][c]==s: return (r,c)
		elif s < m[r][c]: return helper(r,c-1,s)
		elif s > m[r][c]: return helper(r+1,c,s)

		return None


	return helper(0,len(m[0])-1,s)

#a = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
#print(sorted_matrix_search_naive(a,7))

#---------------------------------------------------------------------------------------------------------
# 10.10 Rank from Stream: Imagine you are reading in a stream of integers. Periodically, you wish
# to be able to look up the rank of a number x (the number of values less than or equal to x).
# Implement the data structures and algorithms to support these operations. That is, implement
# the method track(int x), which is called when each number is generated, and the method
# getRankOfNumber(int x), which returns the number of values less than or equal to x (not
# including x itself).
# EXAMPLE
# Stream(in order of appearance): 5, 1, 4, 4, 5, 9, 7, 13, 3
# getRankOfNumber(1) = 0
# getRankOfNumber(3) = 1
# getRankOfNumber(4) = 3

class TreeNode:
	def __init__(self,val):
		self.val = val
		self.left = None
		self.right = None
		self.size = 1

	def incrementSize(self):
		self.size+=1
	
class BST:
	def __init__(self):
		self.root = None

	def add(self,x):
		if not self.root: self.root=TreeNode(x)
		else: self.addHelper(self.root,x)

	def addHelper(self, root, x):
		if not root: return TreeNode(x)
		if x<=root.val: 
			root.left = self.addHelper(root.left, x)
			root.size+=1
		else:  
			root.right = self.addHelper(root.right, x)
			root.size+=1
		return root

	def find(self,val):
		if not val: return None
		return self.findHelper(self.root,val)
	
	def findHelper(self, root, val):
		if not root: return None
		if val == root.val: return root
		if val < root.val: return self.findHelper(root.left, val)
		else: return self.findHelper(root.right, val)

	def getRank(self,val):
		return self.rankHelper(self.root,val)

	def rankHelper(self, root, val):
		if not root: return -1
		if val == root.val:
			if not root.left: return 0
			return root.left.size
		if val < root.val:
			left = self.rankHelper(root.left,val)
			return left if left>=0 else -1
		else:
			if not root.right: return -1
			right = self.rankHelper(root.right,val)
			if right ==-1: return -1
			if not root.left: return 1 + right
			return 1 + root.left.size + right

def rank_of_stream(stream):
	bst = BST()

	#Time Complexity: O(N)
	def track(x):
		bst.add(x)
	# Time Complexity: O(N)
	def getRankOfNumber(x):
		return bst.getRank(x)

	for i in range(len(stream)):
		track(stream[i])

	for i in range(len(stream)):
		r = getRankOfNumber(stream[i])
		print(f"Rank of {stream[i]}: {r}")

	r = getRankOfNumber(21)
	print(f"Rank of {2}: {r}")
		
#stream = [5, 1, 4, 4, 5, 9, 7, 13, 3]
#rank_of_stream(stream)

#---------------------------------------------------------------------------------------------------------
# 10.11 Peaks and Valleys: In an array of integers, a "peak" is an element which is greater than or equal
# to the adjacent integers and a "valley" is an element which is less than or equal to the adjacent
# integers. For example, in the array {5, 8, 6, 2, 3, 4, 6}, {8, 6} are peaks and {5, 2} are valleys. Given an
# array of integers, sort the array into an alternating sequence of peaks and valleys.
# EXAMPLE
# Input: {5, 3, 1, 2, 3}
# Output: {5, 1, 3, 2, 3}


# 6,6,5,4,3,2,1 -> 6,1,6,2,5,3,4
# 6,6,5,4,3,3,2,1 -> 6,1,6,2,5,3,4,3
# 5,5,5,3,3,3,3,2,1 -> 5,1,5,3,5,3,2,3,3

# Time Complexity: O(NLogN), Space Complexity: O(N)
def peaks_and_valleys_naive(arr):
	
	temp = sorted(arr, reverse=True)
	i,j=0,((len(arr)-1)//2)+1
	while j<len(arr):
		arr[i*2] = temp[i]
		arr[(i*2)+1] = temp[j]
		i+=1
		j+=1
	if len(arr)%2:
		arr[2*i] = temp[i]

# Time Complexity: O(NLogN), Space Complexity: O(1), since peak/element based on <= or >= duplciates dont affect
def peaks_and_valleys_naive_2(arr):
	arr.sort()
	i=1
	while i<len(arr):
		arr[i],arr[i-1] = arr[i-1],arr[i]
		i+=2

# Time Complexity: O(N), Space Complexity: O(1)
def peaks_and_valleys(arr):

	i=0
	while i<len(arr):
		if i>0 and arr[i]<arr[i-1]: arr[i],arr[i-1]=arr[i-1],arr[i]
		if i<len(arr)-1 and arr[i]<arr[i+1]: arr[i],arr[i+1]=arr[i+1],arr[i]
		i+=2

'''
a = [8, 5, 4, 3 , 3, 1, 6, 5, 7]
peaks_and_valleys(a)
print(a)'''
