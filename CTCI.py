#------------------------------------------------------------------------------
# 1.1 Implement an alforithm to determine if a string has all unique characters. What if you cannot use additional data structures

# Naive solution
from inspect import stack
from re import I
from tkinter import N
from numpy import char


def question_one_naive(string):
	chars = {}
	for c in string:
		if c in chars: return False
		chars[c] = 1
	return True

# or
def question_one_naive_2(string):
	chars = set()
	for c in string:
		if c in chars: return False
		chars.add(c)
	return True

# Time Complexity: O(N), Space Complexity: O(1)
def question_one(string):

	# This assumes that string is ASCII (128 characters)
	if len(string)>128: return False

	# Create a constant size array of all possible ascii values with an initial count of zero. For each character in the string, increment 
	# the count at the respective index, if count is greater than zero, return False
	character_set = [0 for I in range(128)]
	for c in string:
		val = ord(c)
		if character_set[val] !=0 : return False
		character_set[val]+=1
	
	return True

# If we cant use additional data structures
# Time Complexity: O(N^2), Space Complexity: O(1)
def question_one_2(string):

	# Loop through each char and compare it with every other char in the string 
	for i in range(len(string)):
		for j in range(len(string)):
			if i!=j and string[i]==string[j]: return False
	return True

# If we can use addition data structures
# Time Complexity: O(nlogn), Space Complexity: O(n)?
def question_one_2_2(string):

	# Sort string, and at every index of the string compare the current char with the next char, make sure to only
	# traverse to the second last index
	sorted_string=sorted(string)
	for i in range(len(sorted_string)-1):
		if sorted_string[i]==sorted_string[i+1]: return False
	
	return True

#------------------------------------------------------------------------------
# 1.2 Given two strings, write a method to decide if one is a permutation of the other

# Time Complexity: O(N), Space Complexity: O(N)
def question_two(string1,string2):

	# If lengths dont match, impossible, return False
	if len(string1)!=len(string2): return False

	# Create hashmap of chars in string1, increment for each occurence of a char. Then, loop through each char in string2,
	# if it doesnt exist return False, if the count for the current char is zero return False, otherwise decrement the count.
	# If every char in the second string has been passed thorugh, all hashmap count will be zero, return True
	char_dict = {}
	for c in string1:
		if c in char_dict: char_dict[c] +=1
		else: char_dict[c] =1
	
	for c in string2:
		if c not in char_dict: return False
		if char_dict[c] == 0: return False
		else: char_dict[c] -= 1
	
	return True

# Time Complexity: O(N), Space Complexity: O(1)
def question_two_best(string1,string2):

	# If lengths dont match, impossible, return False
	if len(string1)!=len(string2): return False

	# Create a constant size zero val array of size 128 (ASCII chars), loop through string1 and increment count for respective ASCII char.
	# Loop thorugh string2, if count at char index is zero, return False, otherwise decrement count. If all satisfied, return True
	character_set = [0 for i in range(128)]
	for c in string1:
		character_set[ord(c)] +=1
	for c in string2:
		if character_set[ord(c)] == 0: return False
		character_set[ord(c)] -=1
	return True

def question_two_naive(string1, string2):

	if len(string1)!=len(string2): return False
	
	char_dict={}
	for i in range(len(string1)):
		if string1[i] in char_dict: char_dict[string1[i]] +=1
		else: char_dict[string1[i]] = 1
		if string2[i] in char_dict: char_dict[string1[i]] -=1
		else: char_dict[string2[i]] = -1
	
	for c,v in char_dict.items():
		if v!=0: return False
	return True


#------------------------------------------------------------------------------
# 1.3 Write a method to replace all spaces in a string with '%20'. You may assume that the string has sufficient space at the end
#     to hold the additional characters, and that you are given the "true" length of the string. (Note: if implementing in java
#     please use a character array so that you can perform this operation in place.)


def question_three_naive(string, length):

	for i in range(len(string)):
		if string[i]==' ': 
			string[i+3:] = string[i+1:]
			string[i] = '%'
			string[i+1] = 2
			string[i+2] = 0

	return string

# Time Complexity: O(N) Space Complexity: O(1)
def question_three(string,length):

	# Create index idx for total length of string (array). Traverse the string backwards, from its "true" end, and if we encounter a space
	# char replace the 3 chars at position idx-3:idx with '%20', and decrease the idx count by three. Otherwise, replace the current char
	# with the char at position idx-1 (since idx represent length, we must use length-1 for relative index)

	idx = len(string)
	for i in reversed(range(length)):
		if string[i]==' ': 
			string[idx-3:idx]="%20"
			idx-=3
		else: 
			string[idx-1] = string[i]
			idx-=1


#------------------------------------------------------------------------------
# 1.4 Given a string, write a function to check if it is a permutation of a palindrome.

# Time Complexity: O(N), Space Complexity: O(1)
def question_four(string):

	character_set = [0 for i in range(128)]
	length = len(string)
	odd_count = 0
	for c in string:
		character_set[ord(c)]+=1
		if character_set[ord(c)]%2 ==0: odd_count -= 1
		else: odd_count += 1

	if odd_count > 1: return False
	return True

#------------------------------------------------------------------------------
# 1.5 There are three types of edits that can be performed on strings: insert, remove, or replace. Given two strings write
#     a function to check if they are one edit (or zero edits) away.

# Time Complexity: O(N), Space Complexity: O(1)
def question_five(string1, string2):

	# Check if length is same of difference of 1
	length1 = len(string1)
	length2 = len(string2)
	if abs(length1-length2) >1: return False
	
	#Retrieve longer string
	if length1 > length2: greater = string1
	elif length2 > length1: greater = string2
	else: greater = 0

	# Loop through, and update hashmap count, if even count decrement difference, otherwise increment difference
	# If difference count + length difference is greater than 1 return False, otherwise return True.
	character_set = [0 for i in range(128)]
	diff_count = 0
	for i in range(min(length1,length2)):

		character_set[ord(string1[i])] += 1
		character_set[ord(string2[i])] += 1
		if string1[i] != string2[i]:
			if character_set[ord(string1[i])]%2 ==0: diff_count -=1
			else: diff_count +=1
			if character_set[ord(string2[i])]%2 ==0: diff_count -=1
			else: diff_count +=1
	
	# If length difference, calculate difference count for it
	if greater:
		if character_set[ord(greater[-1])]%2 ==0: diff_count +=1
		else: diff_count -=1
	
	# Check if there is an edit difference of 1 or more
	if diff_count>1: return False
	return True

#------------------------------------------------------------------------------
# 1.6 Implement a method to perform basic string compression using the counts of repeated characters. For example, 
# the string aabcccccaaa would become a2b1c5a3. If the "compressed" string would not become smaller than the original 
# string, your method should return the original string. You can assume the string has only uppercase and lowercase letters (a - z).

# Time Complexity: O(N), Space Complexity: O(N), remember concatenate using '+' is O(N^2) so use '.join()' instead with list
def question_six(string):
	
	#Check if exists or length of 1
	if not string or len(string)==1: return string

	# Loop through, if prev char doesnt match append prev char to list with its count, otherwise increment count and at the end
	# append the last char with its respective count
	count = 1
	new_string = []
	for i in range(1, len(string)):
		if string[i-1]!= string[i]: 
			new_string.extend([string[i-1],str(count)])
			count=1
		else: count +=1
	new_string.extend([string[-1],str(count)])

	# check length
	return ''.join(new_string) if len(new_string)< len(string) else string


#------------------------------------------------------------------------------
# 1.7 Given an image represented by an NxN matrix, where each pixel in the image is 4
#     bytes, write a method to rotate the image by 90 degrees. Can you do this in place?

# Time Compelxity: O(N^2), Space Complexity: O(1)
def question_seven(matrix):
	n = len(matrix)
	def transpose():
		for i in range(n):
			for j in range(i+1,n):
				matrix[i][j],matrix[j][i] = matrix[j][i],matrix[i][j]
		

	def mirror():
		for i in range(n):
			for j in range(n//2):
				matrix[i][j],matrix[i][n-j-1] = matrix[i][n-j-1],matrix[i][j]
	transpose()
	mirror()
	return matrix
	
# Harder to understand version
def question_seven_2(matrix):
	n = len(matrix)
	# once for each layer
	for i in range(n//2):
		# once for each portion of layer
		for j in range(i, n-1-i):
			offset = j-i
			#save top val
			top = matrix[i][j]
			#left to top
			matrix[i][j]=matrix[n-1-i-offset][i]
			#bottom to left
			matrix[n-1-i-offset][i] = matrix[n-1-i][n-1-i-offset]
			#right to bottom
			matrix[n-1-i][n-1-i-offset]= matrix[j][n-1-i]
			#top to right
			matrix[j][n-1-i] = top

	return matrix

def print_matrix(matrix):
	for row in matrix:
		print(' '.join(map(str,row)))

#------------------------------------------------------------------------------
# 1.8 Write an algorithm such that if an element in an MxN matrix is 0, its entire row and column are set to 0.

# Time Complexity: O(N^2), Space Complexity: O(M+N)
def question_eight(matrix):

	m = len(matrix)
	n = len(matrix[0])

	rows = set()
	cols = set()

	for i in range(m):
		for j in range(n):
			if matrix[i][j] == 0:
				rows.add(i)
				cols.add(j)

	for i in range(m):
		for j in range(n):
			if i in rows or j in cols: matrix[i][j]=0
	
	return matrix
# REEEEDDDDOOO
# Space efficient version, Time Complexity: O(N^2), Space Complexity: O(1)
def question_eight_2(matrix):
	
	m = len(matrix)
	n = len(matrix[0])

	firstrow_zero = False
	firstcol_zero = False

	for i in range(n): 
		if matrix[0][i]==0: firstrow_zero=True
	for i in range(m): 
		if matrix[i][0]==0: firstcol_zero=True

	for i in range(1,m):
		for j in range(1,n):
			if matrix[i][j] == 0:
				matrix[i][0]=0
				matrix[0][j]=0

	def zero_out_row(row):
		for i in range(n): matrix[row][i]=0
	def zero_out_col(col):
		for i in range(m): matrix[i][col]=0			

	for i in range(1,m): 
		if matrix[i][0] == 0: zero_out_row(i)
	for i in range(1,n):
		if matrix[0][i] == 0: zero_out_col(i)
		
	if firstrow_zero:
		for i in range(n): matrix[0][i]=0
	if firstcol_zero:
		for i in range(m): matrix[i][0]=0
	
	return matrix

#---------------------------------------------------------------------------------------------------------
# 1.9 Assume you have a method isSubstringwhich checks if one word is a substring
#     of another. Given two strings, s1 and s2, write code to check if s2 is a rotation of s1 using only one
#     call to isSubstring (e.g., "waterbottle" is a rotation of"erbottlewat").

# Time Complexity: O(N), Space Complexity: O(N), assuming isSubstring() runs in O(M+N)
def question_nine(s1,s2):

	if len(s1) == len(s2) != 0:
		concat_s2 = ''.join((s2,s2))
		if isSubstring(concat_s2,s1): return True
	return False


#---------------------------------------------------------------------------------------------------------
# 2.1 Remove Dups: Write code to remove duplicates from an unsorted linked list.
#     FOLLOW UP
#     How would you solve this problem if a temporary buffer is not allowed?

# Time Complexity: O(N), Space Complexity: O(N)
def question_2_1(head):
	if not head: return head

	head_orig = head
	elements = set()
	while head and head.next:
		elements.add(head.val)
		while head.next.val in elements: head.next = head.next.next
		head = head.next
		
	return head_orig

# Time Complexity: O(N^2), Space Complexity: O(1)
def question_2_1_followup(head):
	if not head: return None
	head_orig = head
	while head:
		through = head
		while through.next:
			if through.next.val == head.val: through.next= through.next.next
			else: through = through.next
		head = head.next

#---------------------------------------------------------------------------------------------------------
# 2.2 Return Kth to Last: Implement an algorithm to find the kth to last element of a singly linked list

# Time Complexity: O(N), Space Complexity: O(1), takes two passes
def question_2_2(head,k):
	if not head: return None
	head_orig = head
	length = 0
	while head:
		length+=1
		head = head.next
	for i in range(length-k):head_orig = head_orig.next
	return head_orig.val
                            
# Time Complexity: O(N), Space Complexity: O(1), takes only one pass
def question_2_2_2(head,k):
	
	slow,fast = head,head

	for _ in range(k): fast = fast.next
	while fast:
		slow=slow.next
		fast=fast.next
	return slow.val

#---------------------------------------------------------------------------------------------------------
# 2.3 Delete Middle Node: Implement an algorithm to delete a node in the middle (i.e., any node but
# the first and last node, not necessarily the exact middle) of a singly linked list, given only access to that node.
# EXAMPLE
# Input:the node c from the linked lista->b->c->d->e->f
# Result: nothing is returned, but the new linked list looks like a->b->d->e- >f

# Time Complexity: O(N), Space Complexity: O(1)
def question_2_3(node):
	if not node or not node.next: return node
	node.val = node.next.val
	node.next = node.next.next


#---------------------------------------------------------------------------------------------------------
# 2.4 Partition: Write code to partition a linked list around a value x, such that all nodes less than x come
# before all nodes greater than or equal to x. If x is contained within the list, the values of x only need
# to be after the elements less than x (see below). The partition element x can appear anywhere in the
# "right partition"; it does not need to appear between the left and right partitions.
# EXAMPLE
# Input:
# Output:
# 3 -> 5 -> 8 -> 5 -> 10 -> 2 -> 1 [partition= 5]
# 3 -> 1 -> 2 -> 10 -> 5 -> 5 -> 8

# Time Compelxity: O(N), Space Complexity: O(1)
def question_2_4(head,partition):

	# Initialize two empty nodes
	less = less_o = ListNode(None)
	head_o = prev = ListNode(None)
	prev.next = head

	while head:
		if head.val < partition:
			less.next = head
			less = less.next
			prev.next = head.next
			head= head.next
		else:
			prev = head
			head= head.next

	less.next = head_o.next
	return less_o.next

#---------------------------------------------------------------------------------------------------------
# 2.5 Sum Lists: You have two numbers represented by a linked list, where each node contains a single
# digit. The digits are stored in reverse order, such that the 1 's digit is at the head of the list. Write a
# function that adds the two numbers and returns the sum as a linked list.
# EXAMPLE
# Input: (7-> 1 -> 6) + (5 -> 9 -> 2).That is,617 + 295.
# Output: 2 -> 1 -> 9. That is, 912.
# FOLLOW UP
# Suppose the digits are stored in forward order. Repeat the above problem.
# Input: (6 -> 1 -> 7) + (2 -> 9 -> 5).That is,617 + 295.
# Output: 9 -> 1 -> 2. That is, 912.

class ListNode:
	def __init__(self,val=0, next=None):
		self.val=val
		self.next=next

def print_list(head):
	while head:
		print("->"+str(head.val), end="")
		head = head.next
	print()

# Time Complexity: O(N), Space Complexity: O(1)
def question_2_5(head1, head2):
	#store original head1 pointer, and create prev variable, aswell as a carry
	head_o = prev =  head1
	carry = 0
	
	#loop thorugh h1 and h2 simultaneously
	while head1:
		sum = head1.val+carry
		#add h2 value to sum if it exists, and update head2 node
		if head2: 
			sum+=head2.val
			head2 = head2.next

		#calculate carry, set digit, update prev and head1 nodes
		carry = (sum - sum%10)//10
		head1.val = sum%10
		prev = head1
		head1 = head1.next

	#loop thorugh remaining h2 nodes, and do similar as above
	prev.next=head2
	while head2:
		sum = head2.val+carry
		carry = sum//10
		head2.val = sum%10
		prev = head2
		head2 = head2.next

	#if carry still exists, create a new node, set value, and point to it
	if carry: prev.next = ListNode(val=carry)

	return head_o

# Time Complexity: O(N), Space Complexity: O(N)
def question_2_5_followup(head1, head2):

	# get length of lists
	head1_o,head2_o = head1,head2
	l1,l2 = 0,0
	while head1:
		l1+=1
		head1=head1.next
	while head2:
		l2+=1
		head2=head2.next

	# zero pad any list if shorter than the other
	new_head1 = nh1o = ListNode()
	new_head2 = nh2o = ListNode()

	if l1<l2:
		for _ in range(l2-l1): 
			new_head1.next = ListNode()
			new_head1 = new_head1.next
	else:
		for _ in range(l1-l2): 
			new_head2.next = ListNode()
			new_head2 = new_head2.next

	new_head1.next, new_head2.next = head1_o, head2_o

	# recursively build up a new list
	def helper(h1,h2):

		if not h1 and not h2: return None,0
		res  = ListNode()
		res.next, carry = helper(h1.next,h2.next)
		sum = carry
		if h1: sum+=h1.val
		if h2: sum+=h2.val
		res.val = sum%10
		carry = sum//10

		return res,carry

	# run recursively starting from beginning of zero padded lists
	l,c = helper(nh1o.next, nh2o.next)

	# make head node with carry if exists
	if c:
		return ListNode(c,l)
	return l
	

#---------------------------------------------------------------------------------------------------------
# 2.6 Palindrome: Implement a function to check if a linked list is a palindrome.

# Time Complexity: O(N), Space Complexity: O(N)
def question_2_6(head):

	# reverse while also making completely new list
	def reverse(node):
		prev=None
		while node:
			new = ListNode(node.val)
			new.next = prev
			prev = new
			node = node.next
		return prev

	def reverse2(node):
		if not node or not node.next: return node
		rem = reverse(head.next)
		node.next.next = node
		node.next = None
		return rem

	# Create a reversed copy and compare
	reversed = reverse(head)
	while head:
		if head.val!=reversed.val: return False
		head = head.next
		reversed = reversed.next
	return True

# Time Compelxity: O(N), Space Complexity O(N)
def question_2_6_2(head):
	slow = fast = head
	stack = []
	while fast and fast.next:
		stack.append(slow)
		slow = slow.next
		fast = fast.next.next

	# if list is odd
	if fast: slow=slow.next

	while slow:
		cur = stack.pop()
		if cur.val!=slow.val: return False
		slow=slow.next
	return True

# Time Complexity: O(N), Space Complexity: O(1)
def question_2_6_3(head):

	# find middle node, if odd, skip a node ahead to get mid+1 node
	def get_middle(node):
		slow=fast=node
		while fast and fast.next:
			slow=slow.next
			fast=fast.next.next
		if fast: slow = slow.next
		return slow

	# reverse list
	def reverse(node):
		prev = None
		while node:
			next = node.next
			node.next = prev
			prev = node
			node = next
		return prev
	# get mid node and reverse list from mid node onwards, then sequentially compare first half to reversed second half
	mid =  get_middle(head)
	rev_mid = reverse(mid)
	while rev_mid:
		if rev_mid.val!=head.val: return False
		rev_mid=rev_mid.next
		head=head.next

	return True

#---------------------------------------------------------------------------------------------------------
# 2.7 Intersection: Given two (singly) linked lists, determine if the two lists intersect. Return the intersecting
# node. Note that the intersection is defined based on reference, not value. That is, if the kth
# node of the first linked list is the exact same node (by reference) as the jth node of the second
# linked list, then they are intersecting.

# Time Complexity: O(N), Space Complexity: O(1)
def question_2_7(head1,head2):

	if not head1 or not head2: return None

	# run through both lists, if end nodes dont match, then there is no intersection
	r1 = head1
	r2 = head2
	l1,l2 = 1,1
	while r1.next: 
		l1+=1
		r1 = r1.next
	while r2.next: 
		l2+=1
		r2 = r2.next
	if r1 != r2: return None

	# if one list is greater than the other, remove excess initial elements from the longer list

	if l1>l2: 
		for _ in range(l1-l2): head1 = head1.next
	else: 
		for _ in range(l2-l1): head2 = head2.next
	
	# traverse through lists of equal size and return the common/intersecting element
	while head1!=head2:
		head1 = head1.next
		head2 = head2.next
	return head1


#---------------------------------------------------------------------------------------------------------
# 2.8 Loop Detection: Given a circular linked list, implement an algorithm that returns the node at the
# beginning of the loop.
# DEFINITION
# Circular linked list: A (corrupt) linked list in which a node's next pointer points to an earlier node, so
# as to make a loop in the linked list.
# EXAMPLE
# Input: A -> B -> C - > D -> E -> C [the same C as earlier]
# Output: C

def question_2_8(head):
	if not head.next: return None

	fast = slow = head
	while fast and fast.next:
		slow = slow.next
		fast = fast.next.next
		if fast == slow: 
			break
	if not fast or not fast.next: return None

	slow = head
	while slow!=fast:
		slow = slow.next
		fast = fast.next
	print(fast.val)
	return fast

'''
n1,n2,n3,n4,n5 = ListNode(5),ListNode(2),ListNode(3),ListNode(2),ListNode(1)
n11, n12, n13= ListNode(7),n3, ListNode(6)
n1.next = n2
n2.next = n3
n3.next = n4
n4.next = n1
#n5.next = n6
#n11.next = n12
#n12.next = n13


#print_list(n1)
#print_list(n11)
print(question_2_8(n1))
#print_list(n4)
#print_list(question_2_5_followup(n1,n4))
'''

#---------------------------------------------------------------------------------------------------------
# 3.1 Three in One: Describe how you could use a single array to implement three stacks.

class FixedMultiStack:
	def __init__(self, stacksize):
		self.numstacks = 3
		self.array = [0]*(stacksize*self.numstacks)
		self.sizes = [0]*self.numstacks
		self.stacksize = stacksize

	def isEmpty(self,stacknum):
		if self.sizes[stacknum]  == 0: return True
		return False
	
	def isFull(self,stacknum):
		if self.sizes[stacknum] == self.stacksize: return True
		return False

	def push(self,stacknum,item):
		if self.isFull(stacknum): raise Exception("Stack Full!")
		self.array[stacknum*self.stacksize + self.sizes[stacknum]] = item
		self.sizes[stacknum]+=1

	def pop(self,stacknum):
		if self.isEmpty(stacknum): raise Exception("Stack Empty!")
		self.sizes[stacknum] -= 1
		item = self.array[stacknum*self.stacksize + self.sizes[stacknum]]
		self.array[stacknum*self.stacksize + self.sizes[stacknum]] = 0
		return item

	def peek(self,stacknum):
		if self.isEmpty(stacknum): raise Exception("Stack Empty!")
		return self.array[stacknum*self.stacksize + self.sizes[stacknum]-1]

#---------------------------------------------------------------------------------------------------------
# 3.2 Stack Min: How would you design a stack which, in addition to push and pop, has a function min
# which returns the minimum element? Push, pop and min should all operate in 0(1) time.
 
# Time Complexity: O(1), Space Complexity: O(N)
class MinStack:
	def __init__(self):
		self.stack = []
		self.size = 0
		self.min = []
	
	def isEmpty(self):
		return self.size == 0

	def push(self,item):
		self.stack.append(item)
		if item < self.getMin(): self.min.append(item)
		self.size += 1
	
	def pop(self):
		if self.isEmpty(): return None
		item = self.stack.pop()
		if item == self.getMin(): self.min.pop()
		self.size-=1
		return item

	def peek(self):
		if self.isEmpty(): return None
		return self.stack[-1]

	def getMin(self):
		if self.isEmpty(): return float('inf')
		return self.min[-1]

# More refined way, without using python list, will use linkedlist instead
class MinStackBare:
	def __init__(self):
		self.top = ListNode()
		self.size = 0
		self.min = ListNode()

	def isEmpty(self):
		return self.size == 0

	def push(self,item):
		new_node = ListNode(item,self.top)
		self.top = new_node
		if item < self.getMin(): self.addMin(item)
		self.size+=1

	def pop(self):
		if self.isEmpty(): return None
		item = self.top.val
		self.top = self.top.next
		if item == self.getMin(): self.min = self.min.next
		self.size -=1
		return item

	def peek(self):
		if self.isEmpty(): return None
		return self.top.val
	
	def getMin(self):
		if self.isEmpty(): return float('inf')
		return self.min.val

	def addMin(self,item):
		new_node = ListNode(item,self.min)
		self.min = new_node

#---------------------------------------------------------------------------------------------------------
# 3.3 Stack of Plates: Imagine a (literal) stack of plates. If the stack gets too high, it might topple.
# Therefore, in real life, we would likely start a new stack when the previous stack exceeds some
# threshold. Implement a data structure SetOfStacks that mimics this. SetOfStacks should be
# composed of several stacks and should create a new stack once the previous one exceeds capacity.
# SetOfStacks.push() and SetOfStacks.pop() should behave identically to a single stack
# (that is, pop () should return the same values as it would if there were just a single stack).
# FOLLOW UP
# Implement a function popAt ( int index) which performs a pop operation on a specific sub-stack.

class SetOfStacksO:

	def __init__(self, threshold):
		self.threshold = threshold
		self.stacks = []
		self.stacksfilled = 0
		self.current_size = 0
		self.temp = []

	def push(self,item):
		if self.current_size == self.threshold:
			self.stacks.append([item])
			self.stacksfilled+=1
			self.current_size = 1
		else:
			if self.isEmpty(): self.stacks.append([item])
			else: self.stacks[-1].append(item)
			self.current_size+=1

	def pop(self):
		if self.isEmpty(): return None
		item = self.stacks[self.stacksfilled].pop()
		self.current_size-=1
		if not self.current_size: self.stacks.pop()
		if self.stacksfilled and not self.current_size: 
			self.current_size = self.threshold
			self.stacksfilled-=1
		return item

	def isEmpty(self):
		return self.stacksfilled==0 and self.current_size==0

#FOLLOWUP
class Node:
	def __init__(self,val):
		self.val = val
		self.above=None
		self.below=None

class Stack:
	def __init__(self,threshold):
		self.threshold = threshold
		self.size = 0
		self.bottom = None
		self.top=None
	
	def isEmpty(self):
		return self.size==0

	def isFull(self):
		return self.size==self.threshold

	def push(self,item):
		if self.size>=self.threshold: return False
		self.size+=1
		new_node = Node(item)
		if self.size ==1: self.bottom=new_node
		new_node.below=self.top
		if self.top: self.top.above = new_node
		self.top = new_node
		return True
	
	def pop(self):
		if self.isEmpty(): return False
		item = self.top
		self.top=self.top.below
		self.size-=1
		return item.val

	def removeBottom(self):
		bot = self.bottom
		self.bottom = self.bottom.above
		if self.bottom: self.bottom.below = None
		self.size-=1
		return bot.val

class SetOfStacks:
	def __init__(self,threshold):
		self.threshold = threshold
		self.stacks=[]

	def lastStack(self):
		if self.stacks: return self.stacks[-1]
		return None

	def isEmpty(self):
		last_stack = self.lastStack()
		return last_stack.isEmpty()

	def push(self,item):
		last_stack = self.lastStack()
		if last_stack and not last_stack.isFull: last_stack.push(item)
		else:
			new_stack = Stack(self.threshold)
			new_stack.push(item)
			self.stacks.append(new_stack)
		return True

	def pop(self):
		last_stack = self.lastStack()
		if not last_stack: return None
		item = last_stack.pop()
		if last_stack.size==0: self.stacks.pop()
		return item

	def popAt(self,index):
		return self.leftShift(index,True)

	def leftShift(self,index,removeTop):
		stack = self.stacks[index]
		item = stack.pop() if removeTop else stack.removeBottom()
		if stack.isEmpty(): del self.stacks[index]
		elif len(self.stacks)>index+1:
			val = self.leftShift(index+1,False)
			stack.push(val)
		return item


#---------------------------------------------------------------------------------------------------------
# 3.4 Queue via Stacks: Implement a MyQueue class which implements a queue using two stacks.

class MyQueue:

	def __init__(self):
		self.stacknew = []
		self.stackold = []
		self.size = 0

	def isEmpty(self):
		return self.size == 0

	# O(1)
	def push(self,item):
		self.stacknew.append(item)
		self.size+=1

	def shift(self):
		if not self.stackold:
			for _ in range(self.size): self.stackold.append(self.stacknew.pop())

	# worst case O(N)
	def pop(self):
		if self.isEmpty(): return None
		self.shift()
		item = self.stackold.pop()
		self.size-=1
		return item

	# worst case O(N)
	def peek(self):
		if self.isEmpty(): return None
		self.shift()
		return self.stackold[-1]


#---------------------------------------------------------------------------------------------------------
# 3.5 Sort Stack: Write a program to sort a stack such that the smallest items are on the top. You can use
# an additional temporary stack, but you may not copy the elements into any other data structure
# (such as an array). The stack supports the following operations: push, pop, peek, and is Empty.

# Time Complexity: O(N^2), Space Complexity: O()
def sort_stack(stack):

	temp_stack = []

	while stack:
		item =  stack.pop()
		while temp_stack and temp_stack[-1] > item:
			stack.append(temp_stack.pop())
		temp_stack.append(item)
	while temp_stack: stack.append(temp_stack.pop())
	return stack




#---------------------------------------------------------------------------------------------------------
# RANDOM SORTING

def partition(arr,l,r):

	mid = (l+r)//2
	piv = arr[mid]
	i,j =  l, r
	while i<=j:
		while arr[i]<piv:i+=1
		while arr[j]>piv:j-=1
		if i<=j:
			arr[i],arr[j]=arr[j],arr[i]
			i+=1
			j-=1

	return i

# worst case time: O(N^2), space: O(1)
def quick_sort(arr,l,mid,r):
	p = partition(arr,l,r)
	if l<p-1: quick_sort(arr,l,p-1)
	if p<r: quick_sort(arr,p,r)


def merge(l,r):
	
	merged=[]
	i,j= 0,0

	while i<len(l) and j<len(r):
		if l[i]<r[j]:
			merged.append(l[i])
			i+=1
		else:
			merged.append(r[j])
			j+=1
	while i<len(l):
		merged.append(l[i])
		i+=1
	while j<len(r):
		merged.append(r[j])
		j+=1
	return merged

# worst cast time: O(NlogN), space: O(N)
def merge_sort(arr):
	if len(arr)==1: return arr
	mid = len(arr)//2
	left = merge_sort(arr[:mid])
	right = merge_sort(arr[mid:])
	return merge(left,right)


#---------------------------------------------------------------------------------------------------------
# 3.6 Animal Shelter: An animal shelter, which holds only dogs and cats, operates on a strictly "first in, first
# out" basis. People must adopt either the "oldest" (based on arrival time) of all animals at the shelter,
# or they can select whether they would prefer a dog or a cat (and will receive the oldest animal of
# that type). They cannot select which specific animal they would like. Create the data structures to
# maintain this system and implement operations such as enqueue, dequeueAny, dequeueDog,
# and dequeueCat. You may use the built-in Linked list data structure.

# Nodes for Queue
class Node:
	def __init__(self,val):
		self.val = val
		self.next = None
		self.prev = None

class AnimalNode(Node):
	def __init__(self,val, order):
		super().__init__(val)
		self.order = order

# Queue for animal shelter, O(1) enqueue, O(1) dequeue
class MyQueue2:
	def __init__(self):
		self.head = None
		self.tail = None
		self.size = 0

	def isEmpty(self):
		return self.size == 0

	def enqueue(self,item, order):
		new_node = AnimalNode(item, order)
		if not self.head: self.head = new_node
		if self.tail: self.tail.next = new_node
		new_node.prev = self.tail
		self.tail = new_node
		self.size+=1

	def dequeue(self):
		if self.isEmpty(): return None
		item = self.head
		self.head = item.next
		if self.size==1: self.tail = None
		if self.head: self.head.prev = None
		item.next = None
		self.size-=1
		return item.val

class AnimalShelter:
	def __init__(self):
		self.dogs = MyQueue2()
		self.cats = MyQueue2()
		self.order = 0

	def isEmpty(self):
		return self.dogs.isEmpty() and self.cats.isEmpty()

	def enqueue(self,animal_type, name):
		# enqueue to correct queue and increment order/timestamp 
		if animal_type == 'dog': self.dogs.enqueue(name, self.order)
		elif animal_type == 'cat': self.cats.enqueue(name, self.order)
		self.order += 1

	def dequeueAny(self):
		# pop last animal type from stack and dequeue accordingly
		if self.dogs.isEmpty(): return self.dequeueCat()
		if self.cats.isEmpty(): return self.dequeueDog()
		return self.dequeueDog() if self.dogs.head.order < self.cats.head.order else self.dequeueCat()

	# if we are not incorparting order of animal within animal/Node class itself
	'''
	def correctQueue(self, animal_type):
		# dequeue and into temp queue until correct animal type found, then dequeue remaining into temp, replace class queue (oldest) with temp
		temp = MyQueue2()
		while self.oldest.head.val != animal_type: temp.enqueue(self.oldest.dequeue())
		self.oldest.dequeue()
		while self.oldest.size: temp.enqueue(self.oldest.dequeue())
		self.oldest = temp
	'''
	def dequeueCat(self):
		if self.cats.isEmpty(): return None
		return self.cats.dequeue()
	def dequeueDog(self):
		if self.dogs.isEmpty(): return None
		return self.dogs.dequeue()


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

	def find(self,val):
		if not val: return None
		return self.findHelper(self.root,val)

	def findHelper(self, root, val):
		if not root: return None
		if root.val == val: return root
		if val <= root.val: self.findHelper(root.left)
		self.findHelper(root.right)
	
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

#---------------------------------------------------------------------------------------------------------
# 5.1 Insertion: You are given two 32-bit numbers, N and M, and two bit positions, i and
# j. Write a method to insert M into N such that M starts at bit j and ends at bit i. You
# can assume that the bits j through i have enough space to fit all of M. That is, if
# M = 10011, you can assume that there are at least 5 bits between j and i. You would not, for
# example, have j = 3 and i = 2, because M could not fully fit between bit 3 and bit 2.
# EXAMPLE
# Input: N 10000000000, M = 10011, i = 2, j = 6
# Output: N = 10001001100

def insertion(n,m,i,j):
	clear_n = n & ~((1<<(j+1)) - (1<<i))
	shift_m = m<<i
	return clear_n|shift_m

#---------------------------------------------------------------------------------------------------------
# 5.2 Binary to String: Given a real number between O and 1 (e.g., 0.72) that is passed in as a double, print
# the binary representation. If the number cannot be represented accurately in binary with at most 32
# characters, print "ERROR:'

# Time Complexity: O(len(num)-1), Space Complexity(len(num)) where len(num) is digits in num
def binary_to_string(num):
	if num == 1 or num == 0: return bin(num)
	if num<0 or num>1: return "ERROR"

	binary = ["0."]
	while num:
		if len(binary)>=33: return "ERROR"
		num= num*2
		if num>=1: 
			binary.append("1")
			num-=1
		else: binary.append("0")
		print(num)

	return ''.join(binary)


#---------------------------------------------------------------------------------------------------------
# 5.3 Flip Bit to Win: You have an integer and you can flip exactly one bit from a 0 to a 1. Write code to
# find the length of the longest sequence of 1s you could create.
# EXAMPLE
# Input: 1775  (or: 11011101111)
# Output: 8

# Time Complexity: O(b) where b is number of bits in int, Space Complexity: O(1)
def flip_bit_to_win(integer):
	# if all 1sequence return len of sequence, since python, bitwise not returns signed int (2s compliment), which is just ~n = -n-1 (for 7 -> -8, 6 -> -7, etc.)
	if ~integer == -integer-1: return len(bin(integer))-2
	prev = 0
	cur = 0
	m = 0
	while integer:
		b = integer&1
		if b: cur+=1
		else:
			prev = cur
			cur = 0
		integer = integer>>1
		m = max(m, prev+cur) 
	return m+1

#---------------------------------------------------------------------------------------------------------
# 5.4 Next Number: Given a positive integer, print the next smallest and the next largest number that
# have the same number of 1 bits in their binary representation.

# Time Complexity: O(b), Space Complexity: O(1)
def next_number(integer):

	def largest(integer):
		copy = integer
		count_0s = 0
		count_1s = 0
		# get number trailing zeroes
		while not copy&1 and copy:
			count_0s += 1
			#right shift
			copy = copy >>1
		# get number 1s following trailing zeroes
		while copy&1:
			count_1s += 1
			copy = copy >>1
		#position where to flip 0 to 1, right most non trailing zero
		position = count_0s+count_1s
		# flip right most non trailing zero to 1
		integer = integer | (1<<position)
		#clear all bits to the right (of bit we just flipped)
		integer = integer& ~((1<<position)-1)
		# insert count_1s-1 ones to the right (starting from right most index)
		integer = integer| ((1<<(count_1s-1))-1)
		return integer

	def smallest(integer):
		copy = integer
		count_0s = 0
		count_1s = 0
		# get number trailing ones
		while copy&1:
			count_1s += 1
			#right shift
			copy = copy >>1
		# get number 0s following trailing ones
		while not copy&1 and copy:
			count_0s += 1
			copy = copy >>1
		#position where to flip 1 to 0, left most trailing 1
		position = count_0s+count_1s
		#clear all bits to the right of and including the position (left most trailing 1)
		integer = integer& ~((1<<(position+1))-1)
		# insert count_1s+1 ones to the right (starting from left most index (position+1)), this mean we need to left shift the sequence of ones
		# which we are adding by count_0s-1 since one position is taken by the one we are adding
		t = ((1<<(count_1s+1))-1)
		integer = integer| (t<<(count_0s-1))
		return integer
		
	return smallest(integer), largest(integer)

#---------------------------------------------------------------------------------------------------------
# 5.5 Debugger: Explain what the following code does: ((n & (n-1 )) == 0).

# Ans: n = 10001110100, n-1= 10001110011  !=0
#      n = 1000, n-1= 0111  ==0
#  Therefore it checks whether the current number is the highest possible number able to be represented by the givens set of bits,
#  and therefore checks if the current number is a power of two (or is 0).
#

#---------------------------------------------------------------------------------------------------------
# 5.6 Conversion: Write a function to determine the number of bits you would need to flip to convert
# integer A to integer B.
# EXAMPLE
# Input: 29 (or: 11101), 15 (or: 01111)
# Output: 2

# Time Complexity: O(b), Space Complexity: O(1)
def conversion(a,b):
	xor= a^b
	c=0
	while xor:
		if xor&1:c+=1
		xor=xor>>1
	return c

def conversion_optimized(a,b):
	xor= a^b
	c=0
	# we keep removing least significant bits until we hit zero, the amount of times we do this is the amount of ones present
	# in the xor operation result
	while xor:
		# removes least significant bits
		xor = xor&(xor-1)
		c+=1
	return c

#---------------------------------------------------------------------------------------------------------
# 5.7 Pairwise Swap: Write a program to swap odd and even bits in an integer with as few instructions as
# possible (e.g., bit O and bit 1 are swapped, bit 2 and bit 3 are swapped, and so on).

# 011101010100010101   101 ->r 0010
# 101110101000101010   101 ->l 1010

def pairwise_swap(integer):
	# create an even an odd mask, and extract respective bits. right shift even bits, left shift odd bits
	even_mask = 0xAAAA
	odd_mask = 0x5555
	r_s = (even_mask&integer)>>1
	l_s = (odd_mask&integer)<<1
	return r_s|l_s
	
#---------------------------------------------------------------------------------------------------------
# 5.8  Draw Line: A monochrome screen is stored as a single array of bytes, allowing eight consecutive
# pixels to be stored in one byte. The screen has width w, where w is divisible by 8 (that is, no byte will
# be split across rows). The height of the screen, of course, can be derived from the length of the array
# and the width. Implement a function that draws a horizontal line from (xl, y) to (x2, y).
# The method signature should look something like:
# drawLine(byte[] screen, int width, int xl, int x2, int y)

def draw_line():
	


print(bin(0x5))
print(bin(0xA))
print(bin(191018))
a = pairwise_swap(191018)
print(bin(a))
'''
a = 1024
b = 19
print(bin(a))
print(bin(b))
c=insertion(a,b,2,6)
print(bin(c))
'''

a = TreeNode(2)
a.left = TreeNode(1)
a.right = TreeNode(3)
a.left.left = TreeNode(-1)
a.left.right = TreeNode(1)
#a.left.left.left = TreeNode(9)

b = TreeNode(1)
b.left = TreeNode(9)
b.right = TreeNode(8)

#print(check_subtree(a,b))


p = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
d = [('f','c'),('f','b'),('f','a'),('b','h'),('a','e'),('d','g')]
