#------------------------------------------------------------------------------
# 1.1 Implement an alforithm to determine if a string has all unique characters. What if you cannot use additional data structures

# Naive solution
from inspect import _ParameterKind, stack
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
# and the width. Implement a function that draws a horizontal line from (x1, y) to (x2, y).
# The method signature should look something like:
# drawLine(byte[] screen, int width, int x1, int x2, int y)

def draw_line(screen, w, x1, x2, y):

	height = len(screen)/(w/8)

	if x1<x2: start,end = x1,x2
	else: start,end = x2,x1
	if start<0 or end>w or y>height: return None

	start_offset = start%8
	first_fullbyte = start//8
	# first full byte will be in the next byte within the array, we will have to deal with this one itself
	if start_offset: first_fullbyte+=1
	end_offset = end%8
	last_fullbyte = end//8
	# last full byte will have been the previous one, we will have to deal with this one itself
	if end_offset!=7: last_fullbyte-=1

	# place/set all full bytes (8 bits set to 1 is just 0xFF)
	for i in range(first_fullbyte,last_fullbyte+1): screen[height*(w/8) +i] = 0xFF

	#set first and last bytes, there is a special case for when both are the same
	# if x1 and x2 are the 1 bits in the following: 00100010, our final byte should be: 00111110
	# if we 0xFF>>start_offset we get: 00111111, if we 0xFF^(0xFF>>(end_offset+1)) we get: 11111110
	# if we AND these two we get: 00111110, which is what we want!
	first_byte = start//8
	last_byte = end//8
	if first_byte == last_byte: 0xFF>>start_offset & (0xFF^(0xFF>>(end_offset+1)))
	else:
		if start_offset: screen[height*(w/8)+first_byte] = 0xFF>>start_offset
		if end_offset!=7: screen[height*(w/8)+last_byte] = 0xFF^(0xFF>>(end_offset+1))
	
	# only if required
	# set all bytes before first and after last to 0
	for i in range(0,height*(w/8)+first_byte): screen[i] = 0
	for i in range(height*(w/8)+last_byte+1,len(screen)): screen[i] = 0
	
#---------------------------------------------------------------------------------------------------------
# 6.1 The Heavy Pill: You have 20 bottles of pills. 19 bottles have 1.0 gram pills, but one has pills of weight
# 1.1 grams. Given a scale that provides an exact measurement, how would you find the heavy bottle?
# You can only use the scale once.

#Ans: Take 1 pill from bottle 1,  2 pills from bottle 2,..., n pills from bottle n, ..., 20 pills from bottle 20
# So we have weight = (1*w)+ (2*w)+ ... + (20*w), if all w same then weight = w * arthmetic_seq_sum = w * (n(n+1)/2) =w*210=210
# however one bottle has an additional 0.1 per pill, call this a1 if from bottle 1, a2 if from bottle 2, etc. Therefore, if a1
# w += 0.1, if a2 w += 0.2, ..., if a20 w+= 2. Therfore all we need to do is measure our collection of pills, and subtract from it
# 210 grams. The remainder tells us which bottle is heavier according to the aforementioned statements. Or simply we could do
# remainder / 0.1 = bottle number.

#---------------------------------------------------------------------------------------------------------
# 6.2 Basketball: You have a basketball hoop and someone says that you can play one of two games.
# Game 1: You get one shot to make the hoop.
# Game 2: You get three shots and you have to make two of three shots.
# If p is the probability of making a particular shot, for which values of p should you pick one game
# or the other?

#Ans: P1(win) = P, P2(win)= P^3 + 3(p*p*(1-p)) = P^3 + 3P^2 - 3P^3 = 3P^2 - 2P^3
# P1 > P2 -> P > 3P^2 - 2P^3 -> 2P^2 - 3P + 1 > 0 -> (2P - 1)(P - 1) > 0
# both terms need to be postive or negative (to hold inequality), but P<1 (axiom), therefore second term is negative, then
# first term needs to be as well, therefore flip inequality -> P < 1/2 or P>1 (cant happen) (...just sketch parabola, easier to see )
# Therefore when P< 1/2 choose Game1, when P>1/2 choose Game2, if P= 0 or 1 or 1/2, then does not matter

#---------------------------------------------------------------------------------------------------------
# 6.3 Dominos: There is an 8x8 chessboard in which two diagonally opposite corners have been cut off.
# You are given 31 dominos, and a single domino can cover exactly two squares. Can you use the 31
# dominos to cover the entire board? Prove your answer (by providing an example or showing why
# it's impossible).

#Ans: X X X 
#     X X X X
#     X X X X
#       X X X
# No, its impossible. Will always need to use next row or column when laying down current row or column. Alternatively, we
# started out with 32 white and 32 black squares. No we have 32 white and 30 black, and no matter what, a dominoe always covers
# one black and one white square, with 31 dominoes we need 31 white and 31 black squares, but we now only have 20 black squares.
# Therefore it is impossible to lay down all 31 dominoes.

#---------------------------------------------------------------------------------------------------------
# 6.4 Ants on a Triangle: There are three ants on different vertices of a triangle. What is the probability of
# collision (between any two or all of them) if they start walking on the sides of the triangle? Assume
# that each ant randomly picks a direction, with either direction being equally likely to be chosen, and
# that they walk at the same speed.
# Similarly, find the probability of collision with n ants on an n-vertex polygon.

#Ans: P(collide) = 1 - P(not collide), P(not collide) = P(all right) + P(all left) = (1/2)^3 + (1/2)^3 = 1/4
# So P(collide) = 1 - (1/4) = 3/4
# In general P(collide)_n = 1 - ((1/2)^n + (1/2)^n) = 1 - 2(1/2)^n = 1 - (1/2)^(n-1)

#---------------------------------------------------------------------------------------------------------
# 6.5 Jugs of Water: You have a five-quart jug, a three-quart jug, and an unlimited supply of water (but
# no measuring cups). How would you come up with exactly four quarts of water? Note that the jugs
# are oddly shaped, such that filling up exactly "half" of the jug would be impossible.

#Ans: Fill up the 5, pour from 5 into 3 until 3 is full, dump 3, now we have 2 in 5, pour this into 3. Now 5 is empty, and
# and 3 is 2/3 full with 1 remaning space of 1. Now fill 5 all the way up, and pour into 3 until full, since there is exactly
# room for 1 left in 3, 5 will now have 4 left. And so here we are, there are 4 quartz of water in the 5 quart jug.
# Note if the two jug sizes are relatively prime, able to measure any value between one and sum of jug sizes!


#---------------------------------------------------------------------------------------------------------
# 6.6 Blue-Eyed Island: A bunch of people are living on an island, when a visitor comes with a strange
# order: all blue-eyed people must leave the island as soon as possible. There will be a flight out at
# 8:00 pm every evening. Each person can see everyone else's eye color, but they do not know their
# own (nor is anyone allowed to tell them). Additionally, they do not know how many people have
# blue eyes, although they do know that at least one person does. How many days will it take the
# blue-eyed people to leave?

#Ans: If only one person has blue eyes, and since they know at least one person does, he can look at everyone and realize that
# he is the one with the blue eyes, and he will leave the first day. Therefore it will take one day in this case. In the case of two, 
# on the second day they would see each other, and since yesterday they assumed the other would have left because of the reasoning 
# for the base case they would instantly realize that both of them have blue eyes and would both leave on the second day. This same 
# logic applies to when n number of people with blue eyes and would take n days for them to leave. In general, when a person sees n
# people with blue eyes (n>=1), he knows that there are either n or n+1 people with blue eyes. The n he sees or the n he sees
# and himself aswell. To know what the truth is, all he has to do is wait n days, if after n days the people with blue eyes are
# still there, he can deduce that there are n+1 people with blue eyes and he is one of them. This logic applies to all the other n
# people, and they together at once will leave on that nth day.

#---------------------------------------------------------------------------------------------------------
# 6.7 The Apocalypse: In the new post-apocalyptic world, the world queen is desperately concerned
# about the birth rate. Therefore, she decrees that all families should ensure that they have one girl or
# else they face massive fines. If all families abide by this policy-that is, they have continue to have
# children until they have one girl, at which point they immediately stop-what will the gender ratio
# of the new generation be? (Assume that the odds of someone having a boy or a girl on any given
# pregnancy is equal.) Solve this out logically and then write a computer simulation of it.

#Ans: G, BG, BBG, BBBG, BBBBG, BBBBBG, ...
# P(G) = 1/2, P(BG) = (1/2)^2, P(BBG) = (1/2)^3 , therefore every family has exactly one girl, for boys use E(V)
# E(B) = 0*(1/2) + 1*(1/2)^2 + 2*(1/2)^3 + ... + to infinity = Sum of i/(2^(i+1)) for i=0 to inf. 
# Sum of i/(2^i) for i=0 to inf = 1/4 + 2/8 + 3/16 + 4/32 + 5/64 + 6/128 = (32+32+24+16+10+6)/128 = 120/128 +eps ~= 128/128
# can tell by rough approximation that it is converging towards 1. Therefore on average every family has one boy. Therefore,
# the gender ratio is 1:1, it is even.

def apocalypse():

	
	def makeFamily():
		global count_girls, count_boys
		g_or_b = random.randint(0,1)
		if g_or_b: 
			count_girls+=1
			return
		count_boys+=1
		makeFamily()
	global count_girls, count_boys
	count_boys = 0
	count_girls = 0
	for i in range(1000): makeFamily()
	print(f"GENDER RATIO (B:G): {count_boys}:{count_girls}")

def apocalypse_():
	
	def makeFamily():
		num_girls, num_boys = 0,0
		while not num_girls:
			g_or_b = random.randint(0,1)
			if g_or_b: num_girls+=1
			else:num_boys+=1
		return num_girls, num_boys

	count_boys = 0
	count_girls = 0
	for i in range(100000): 
		g,b = makeFamily()
		count_girls+=g
		count_boys+=b
	print(f"GENDER RATIO (B:G): {count_boys}:{count_girls}")


#---------------------------------------------------------------------------------------------------------
# 6.8 The Egg Drop Problem: There is a building of 100 floors. If an egg drops from the Nth floor or
# above, it will break. If it's dropped from any floor below, it will not break. You're given two eggs. Find
# N, while minimizing the number of drops for the worst case.

# Ans: Cant do binary search because only two eggs.
# Egg 1 will need to test a larger gap of floors, once it breaks egg 2 will need to be used to linearly search which floor N is.
# To minimize worst case, its best to use the load balancing approach. Basically when we choose our initial floor for egg1,
# call it x, if egg1 breaks,egg two will need to do a maximum of x-1 drops. However, if egg1 doesnt break then we will try at
# floor 2x, and again if it breaks, egg2 will need to linearly scan from x+1 to 2x-1. So at worst, egg2 is always 
# doing x-1 drops, while at worst egg1 could do 100/x drops. Instead we can keep the total number of drops constant if we
# decrement by one x (the next floor egg1 will test). This means that as the count increased, due to us testing on a new floor,
# which would be floor x + (x-1), the worst case for egg2 will have reduced by one, it will now only need to test x+1 to 2x-2 floor.
# So we need to choose x + (x-1) + (x-2) + ... + 1 = x(x+1)/2 =100 - > x = 13.651. If we choose 13, then worst is 
# 13+12+11+10+9+8+7+6+5+4+3+2+1 = 91, so 13+9 = 22 drops.
# If we choose 14, then worst is 14, e.g. 14*1=14, so 1+13, which is 14 drops.

#---------------------------------------------------------------------------------------------------------
# 6.9 100 Lockers: There are 100 closed lockers in a hallway. A man begins by opening all 100 lockers.
# Next, he closes every second locker. Then, on his third pass, he toggles every third locker (closes it if
# it is open or opens it if it is closed). This process continues for 100 passes, such that on each pass i,
# the man toggles every ith locker. After his 100th pass in the hallway, in which he toggles only locker
# #100, how many lockers are open?

# Ans: Can think of this in terms of numbers/number theory. Every locker(number) from 1 through 100 has an even number of 
# divisors, except for numbers that have a square root. Numbers that have an integer square root, have an odd number of factors.
# E.g 10's factors are 1,10,2,5, this is because every factor has a compliment which is also a factor, in the case of
# a perfect square, one of its factors is the same as its compliment factor. For 16: 1,16,2,8,4, and hence it has an odd number of 
# divisors. So it is only the square locker numbers that will be toggled by the man an odd number of times. Since they are initially
# closed if they are toggled an odd number of times they will be open. 
# So the open lockers will be 1, 4, 9, 16, 25, 36, 49, 64, 81, 100. Therefore # of lockers open = 10.

#---------------------------------------------------------------------------------------------------------
# 6.10 Poison: You have 1000 bottles of soda, and exactly one is poisoned. You have 10 test strips which
# can be used to detect poison. A single drop of poison will turn the test strip positive permanently.
# You can put any number of drops on a test strip at once and you can reuse a test strip as many times
# as you'd like (as long as the results are negative). However, you can only run tests once per day and
# it takes seven days to return a result. How would you figure out the poisoned bottle in as few days
# as possible?
# Follow up: Write code to simulate your approach.

# Ans: Group into 100bottles, test on 10 strips. Get group of poisoned, now we have 100b and 9s. Group into 10b test 9 groups with
# the 9s. If poisoned, we have 10b 8s. Test 8b with the 8s. If not poisoned, we ahve 2b 7s, test the 2b with 2s. Get poisoned. 
# This takes a max of 28 days. All other possible combinations with this method take less. Therefore fewest days is 28.
#
# Better approach: 10 test strips can act as 10 binary bits. To represent 1000 we need 10 bits, therefore all 10 bottles can be 
# represented by some combiantion of test strips. This means we will provide a drop of soda on the 1 bit of the binary 
# representation of the bottle number, e.g 1000 -> 1111101000, so teststrips 9-5 and 3 will have a drop from bottle 1000. We do this
# for all 1000s bottles. Since exactly one bottle is poisoned after 7 days, we align the test strips in same order as before and
# the positive test strips will act as a filter and match the binary representation of the bottle which is poisoned.

def poison(bottles,strips):
	# Loop through bottles and and add drop to corresponding strip, 
	# we only simulate adding a poisoned drop, all other drops are effectively 0. This does drop adding and testing at the same time
	for i in range(len(bottles)):
		for j in range(len(strips)):
			if i & (1<<j) and bottles[i]:
				strips[j]=1
	
	print("7 Days elapsed: Results returned")
	
	#Check positive strips and get poisoned bottle from the binary representation
	result = 0 
	for i in range(len(strips)): 
		if strips[i]:result = result | (1<<i)
	
	return result
'''
poisonb = 597
bottles = [i==poisonb for i in range(1000)]
strips = [0 for i in range(10)]
print(f"Poisoned Bottle: {poisonb}")
detected = (poison(bottles,strips))
print(f"Detected Bottle: {detected}")
'''

#---------------------------------------------------------------------------------------------------------
# 7.1 Deck of Cards: Design the data structures for a generic deck of cards. Explain how you would
# subclass the data structures to implement blackjack.

class Card:
	# ranks and suits to be used
	RANKS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)
	SUITS = ('C', 'D', 'H', 'S')

	def __init__(self,rank,suit):
		self.rank = rank
		self.suit = suit

	# when str called on Card, we print face cards accordingly
	def __str__ (self):
		if (self.rank == 1):
			rank = 'A'
		elif (self.rank == 13):
			rank = 'K'
		elif (self.rank == 12):
			rank = 'Q'
		elif (self.rank == 11):
			rank = 'J'
		else:
			rank = str (self.rank)
		return rank + self.suit

class CardDeck:
	def __init__(self):
		self.cards = []
		self.makeDeck()

	#Make New Deck
	def makeDeck(self):
		for rank in Card.RANKS:
			for suit in Card.SUITS:
				self.cards.append(Card(rank,suit))
	#Shuffle
	def shuffle(self):
		for i in range(len(self.cards)):
			s = random.randint(0,i)
			self.cards[s],self.cards[i] = self.cards[i],self.cards[s]

		# or could juse use this
		#random.shuffle(self.cards)
	
	#Get Top Card
	def get_card(self):
		return self.cards.pop()

class Player:
	'''Player class'''
	def __init__(self,cards):
		self.cards = cards
		self.bust = False

	# get score of player
	def get_score(self):
		score = 0
		for c in self.cards:
			if c.rank >9: score+=10
			elif c.rank==1: score+=11
			else: score+=c.rank
		for c in self.cards:
			if score<=21: return score
			if c.rank==1: score-=10

		return score
	
	# return comma seperated string of cards the player holds
	def get_cards(self):
		cards = []
		for c in self.cards:
			cards.append(str(c))
		return ','.join(cards)

	# add to the players cards/hand
	def hit(self,card):
		self.cards.append(card)
	
	# set bust flag
	def make_bust(self):
		self.bust = True

class Dealer(Player):
	''' Same as Player, can add other functions as needed'''
	def __init__(self,cards):
		super().__init__(cards)

class BlackJack(CardDeck):
	def __init__(self,n_players):
		# create a deck, shuffle it
		self.deck = CardDeck()
		self.deck.shuffle()
		self.players = []
		self.cards_in_hand = 2

		#Create Players with their initial hands
		for i in range(n_players):
			hand = []
			for j in range(self.cards_in_hand): hand.append(self.deck.get_card())
			self.players.append(Player(hand))
		
		#Create Dealers with initial hand
		dealer_hand=[]
		for i in range(self.cards_in_hand): dealer_hand.append(self.deck.get_card())
		self.dealer = Dealer(dealer_hand)

	def play(self):
		# print player hands
		for i in range(len(self.players)):
			print(f"Player: {i}, Cards: {self.players[i].get_cards()}, Score: {self.players[i].get_score()}")
		# print dealer first card
		print(f"Dealer, First Card: {str(self.dealer.cards[0])}")

		# Let each player play
		for i in range(len(self.players)):
			player = self.players[i]
			hit = True
			while hit and player.get_score()<21:
				resp = input(f"Player {i}: Want to hit? (y/n")
				hit = resp=='y'
				if hit: 
					player.hit(self.deck.get_card())
					print(f"Player: {i}, Cards: {self.players[i].get_cards()}, Score: {self.players[i].get_score()}")
				
			if player.get_score()>21: player.make_bust()

		# Play Dealer
		while self.dealer.get_score()<17:
			self.dealer.hit(self.deck.get_card())

		# check if dealer is busted
		if self.dealer.get_score()>21: self.dealer.make_bust()

		# print dealer score
		print(f"Dealer, Cards: {self.dealer.get_cards()}, Score: {self.dealer.get_score()}")
		

		# if dealer bust, check who, if any, players win
		if self.dealer.bust:
			for i in range(len(self.players)):
				if not self.players[i].bust: print(f"Player {i} wins!")
				else: print(f"Player {i} loses.")

		# otherwise check who has higher score than dealer
		else: 
			for i in range(len(self.players)):
				if not self.players[i].bust:
					if self.players[i].get_score()>self.dealer.get_score(): print(f"Player {i} wins!")
					elif self.players[i].get_score()==self.dealer.get_score(): print(f"Player {i} ties.")
					else: print(f"Player {i} loses.")

				else: print(f"Player {i} loses.")


#b = BlackJack(2)
#b.play()


#---------------------------------------------------------------------------------------------------------
# 7.2 Call Center: Imagine you have a call center with three levels of employees: respondent, manager,
# and director. An incoming telephone call must be first allocated to a respondent who is free. If the
# respondent can't handle the call, he or she must escalate the call to a manager. If the manager is not
# free or not able to handle it, then the call should be escalated to a director. Design the classes and
# data structures for this problem. Implement a method dispatchCall() which assigns a call to
# the first available employee.

class Employee:
	def __init__(self,rank=None):
		self.rank = rank
		self.call = None

	def take_call(self,call):
		self.call = call

	def escalate(self,call):
		if self.rank == 'R': self.call.rank = "M"
		if self.rank == 'M': self.call.rank = "D"
		call.dispatchCall(call)
	
	def getRank(self):
		return self.rank

	def isFree(self):
		return self.call


class Respondent(Employee):
	def __init__(self,emp_number):
		super().__init__(emp_number, "R")

class Manager(Employee):
	def __init__(self,emp_number):
		super().__init__(emp_number, "M")

class Director(Employee):
	def __init__(self,emp_number):
		super().__init__(emp_number, "D")

class Call:
	def __init__(self,caller):
		self.caller = caller
		self.rank = "R"
		self.handler = None
	
	def setHandler(self,employee):
		self.handler = employee

	def setRank(self,rank):
		self.rank = rank

	def getRank(self):
		return self.rank
	

class CallCenter:

	RANKS = ('R','M','D')

	def __init__(self,num_r=10,num_m=2,num_d=2):
		self.respondents = []
		self.managers = []
		self.directors = []
		self.employees = 0
		self.makeEmployees(num_r,'R')
		self.makeEmployees(num_m,'M')
		self.makeEmployees(num_d,'D')
		self.queue = []

	def makeEmployees(self,num,rank):
		if not rank: return None
		emp = []
		for i in range(num): 
			if rank == 'R':
				emp.append(Respondent(self.employees))
			elif rank == 'M':
				emp.append(Manager(self.employees))
			elif rank == 'D':
				emp.append(Manager(self.employees))
			self.employees+=1
		
		if rank == 'R': self.respondents.append(emp)
		elif rank == 'M': self.managers.append(emp)
		elif rank == 'D': self.directors.append(emp)
	
	def callHandler(self,call):
		if call.rank == 'R': handler = self.respondents
		if call.rank == 'M': handler = self.managers
		if call.rank == 'D': handler = self.directors
		for i in range(len(handler)):
			if handler[i].isFree(): return handler[i]

		# for manager ranked calls, try to see if director is free
		if call.rank == 'M':
			handler = self.directors
			for i in range(len(handler)):
				if handler[i].isFree():
					call.rank = 'D' 
					return handler[i]
		
		return None

	def dispatchCall(self,call):
		c = call
		if self.queue:
			self.queue.append(c)
			c = self.queue.pop(0)
		handler = self.callHandler(c)
		if not handler:
			print("Call in Queue")
			self.queue.append(c)
			return

		handler.take_call(c)
		call.setHandler = handler


#---------------------------------------------------------------------------------------------------------
# 7.3 Jukebox: Design a musical jukebox using object-oriented principles.

class JukeBox:
	def __init__(self):
		self.cdplayer = None
		self.cds = []
	def play(self):
		self.cdplayer.playSong()
	def selectCD(self,cd):
		self.cdplayer.selectCD(cd)
	def selectSong(self,song):
		self.cdplayer.playSong(song)
	def addCDs(self,cds):
		for c in cds: self.cds.append(c)
	def removeCDs(self,cds):
		for c in cds: self.cds.remove(c)

class CDPlayer:
	def __init__(self,cd,playlist):
		self.song = cd
		self.cd = playlist
		self.playlist = None

	def playSong(self,song):
		#send to music/audio buffer, and play
		pass
	def setPlaylist(self, playlist):
		self.playlist = playlist
	def getPlaylist(self, playlist):
		return self.playlist
	def setCD(self,cd):
		self.cd = cd
	def getCD(self):
		return self.cd
	def setPlaylist(self, playlist):
		self.playlist = playlist
	
class CD:
	def __init__(self,name,songs):
		self.name = name
		self.songs = songs
	
class Song:
	def __init__(self,name,artist):
		self.name = name
		self.artist = artist

class Playlist:
	def __init__(self,name):
		self.name = name
		self.songs = []

	def addSong(self,song):
		self.songs.append(song)

	def removeSong(self,song):
		self.songs.remove(song)

	def shuffle(self):
		self.songs.random.shuffle()

	def getNextSong(self):
		return self.songs[0]

# Cleaner way if no other info is givens
class JukeBox:
	def __init__(self,songs):
		self.songs={}
		for s in songs:
			self.songs[s] = s.name
		self.current_song = None
	
	def playSong(self, name):
		if self.current_song: self.stopSong()
		self.current_song = name
		self.current_song.play() 

	def stopSong(self):
		if self.current_song: self.current_song.stop()

class Song:
	def __init__(self,name):
		self.name = name
		self.playing = False

	def play(self):
		self.playing = True

	def stop(self):
		self.playing = False


#---------------------------------------------------------------------------------------------------------
# 7.4 Parking Lot: Design a parking lot using object-oriented principles.

# Assume lot has multiple levels. Can park motorcycles, cars and buses. Has motorcycle, and regular. A motorcycle
# can park in any spot. A car can park in a regular spot. Bus can park in five regular spots that are consecutive and within
# the same row.

# The answer in the book is what this one follows. However the answer gets convoluted. There are many random things/assumptions
# that are introduced. I think its best to jsut create a Vehicle base class, then a few types of vehicles. Then create a Parking Lot    
# wrapper class, and a Level and Parking Spot class, that should suffice. All the other stuff becomes too much needlessly.

VEHICLE_TYPES = ('M', 'C', 'B')
class Vehicle:
	def __init__(self, vehicle_type):
		self.vehicle_type = vehicle_type
		#self.spots_required = spots_required
		#self.parking_spots = set()
	'''
	def parkInSpot(self, spot):
		self.parking_spots.add(spot)

	def unParkFromSpot(self, spot):
		self.parking_spots.remove(spot)
	'''

class Motorcycle(Vehicle):
	def __init__(self):
		super().__init__('M')

class Car(Vehicle):
	def __init__(self):
		super().__init__('C')

class Bus(Vehicle):
	def __init__(self):
		super().__init__('B')

class ParkingLot:

	def __init__(self, num_levels=3):
		self.levels = []
		self.num_levels = num_levels
		self.makeLevels()

	def makeLevels(self):
		for i in range(self.num_levels):
			self.levels.append(Level(i,100,10))
	
	def parkVehicle(self,vehicle):
		for level in self.levels:
			spot = level.findAvailableSpot(vehicle)
			if spot:
				level.parkAtSpot(spot,vehicle)
				return True
		return False

class Level:
	def __init__(self,floor,num_spots=100,spots_per_row=10):
		self.floor = floor
		self.parking_spots = []
		self.available_spots = 0
		self.spots_per_row = spots_per_row
		self.makeSpots(num_spots)

	def makeSpots(self, num_spots):
		num_rows = num_spots//self.spots_per_row
		for i in range(num_rows):
			for _ in range(self.spots_per_row):
				self.parking_spots.append(ParkingSpot(self.floor, i, self.available_spots,'C'))
				self.available_spots+=1

	def parkAtSpot(self,spot,vehicle):
		spot.parkInSpot(vehicle)

	def findAvailableSpot(self, vehicle):
		if not self.available_spots: return None
		for spot in self.parking_spots: 
			if spot.vehicle_type == vehicle.vehicle_type: return spot
		return None

	def spotFreed(self):
		self.available_spots+=1

class ParkingSpot:
	def __init__(self, level, row, spot_number, vehicle_type):
		self.level = level
		self.row = row
		self.spot_number = spot_number
		self.vehicle_type = vehicle_type
		self.vehicle = None

	def isAvailable(self):
		return self.vehicle

	def canFitVehicle(self, vehicle):
		return self.vehicle_type==vehicle.vehicle_type

	def parkInSpot(self, vehicle):
		self.vehicle = vehicle
		#self.vehicle.parkInSpot(self)

	def unParkFromSpot(self):
		if not self.vehicle: return None
		#self.vehicle.unParkFromSpot(self)
		self.vehicle = None
		self.level.spotFreed()

	def getRow(self):
		return self.row
	
	def getSpotNumber(self):
		return self.spotNumber

#---------------------------------------------------------------------------------------------------------
# 7.5 Online Book Reader: Design the data structures for an online book reader system.

class OnlineBookReader:
	def __init__(self):
		self.userManager = UserManager()
		self.library = Library()
		self.display = Display()
		self.current_book = None
		self.current_user = None

	def getLibrary(self):
		return self.library
	
	def getDisplay(self):
		return self.display
	
	def getCurrentBook(self):
		return self.current_book
	
	def getCurrentUser(self):
		return self.current_user

	def setCurrentBook(self,book):
		self.current_book = book
		self.display.displayBook(book)

	def setCurrentUser(self,user):
		self.current_user = user
		self.display.displayUser(user)
	


class UserManager:
	def __init__(self):
		self.users = {}

	def addUser(self,id, info):
		if id in self.users: return None
		new_user = User(id, info)
		self.users[id] = new_user
		return new_user

	def deleteUser(self,user):
		if user.id not in self.users: return False
		self.users.pop(user.id)
		return True
	
	def getUser(self,id):
		if id not in self.users: return None
		return self.users[id]
	
class User:
	def __init__(self,id,info):
		self.id = id
		self.info = info
	def getUserId(self):
		return self.id
	def setUserId(self,id):
		self.id = id
	def getInfo(self):
		return self.info
	def setInfo(self,info):
		self.info = info

class Display:
	def __init__(self):
		self.current_user= None
		self.current_book= None
		self.page_num = 0
	
	def displayUser(self, user):
		self.current_user = user.id
		self.reRender()
	
	def displayBook(self,book):
		self.current_book = book
		self.reRender()

	def nextPage(self):
		if self.page_num < self.current_book.getNumPages(): self.page_num+=1
		else: return None
		self.reRender()

	def previousPage(self):
		if self.page_num>0: self.page_num-=1
		else: return None
		self.reRender()

	def reRender(self):
		#display(self.current_book, self.current_user, self.page_num)
		# peripheral / buffers / drivers that handle 
		pass

class Library:
	def __init__(self):
		self.books = {}

	def addBook(self,id,info):
		if id in self.books: return None
		new_book = Book(id, info)
		self.books[id] = new_book
		return new_book

	def deleteBook(self,book):
		if book.id not in self.books: return False
		self.books.pop(book.id)
		return True
	
	def getBook(self,id):
		if id not in self.books: return None
		return self.books[id]

class Book:
	def __init__(self, id, info):
		self.id = id
		self.info = info

	def getBookId(self):
		return self.id
	def setUserId(self,id):
		self.id = id
	def getInfo(self):
		return self.info
	def setInfo(self,info):
		self.info = info
	def getNumPages(self):
		return self.info.getNumPages


#---------------------------------------------------------------------------------------------------------
# 7.6 Jigsaw: Implement an NxN jigsaw puzzle. Design the data structures and explain an algorithm to
# solve the puzzle. You can assume that you have a fitsWith method which, when passed two
# puzzle edges, returns true if the two edges belong together.

# This is not as convoluted as the official answer. Basically we create a puzzle, and for each piece, have a set 
# of the (max 4) edges/surrounding pieces that it fits with and is currently connected to. The piece class also 
# has a method that allows you to connect another piece to it, this just adds the other piece to the connectedset 
# within the current piece and vice versa. To solve the puzzle we iterate through each piece in the puzzle, and if 
# it is in the other fits set, we connect the two.

class JigsawPuzzle:
	def __init__(self, size):
		self.pieces = []
		self.size = size
		self.makePuzzle()

	def makePuzzle(self):
		pieces = [[Piece() for _ in range(self.size)] for _ in range(self.size)]
		for i in range(self.size):
			for j in range(self.size):
				if i: pieces[i][j].fitsWith(pieces[i-1][j])
				if j: pieces[i][j].fitsWith(pieces[i][j-1])
				self.pieces.append(pieces[i][j])
	
	def isSolved(self):
		for piece in self.pieces:
			if piece.connected != piece.fits: return False
		return True

	def solvePuzzle(self):
		for a in self.pieces:
			for b in self.pieces:
				if b in a.fits: a.connect(b)

class Piece:
	def __init__(self):
		self.fits = set()
		self.connected = set()

	def fitsWith(self,piece):
		self.fits.add(piece)
		piece.fits.add(self)

	def connect(self,piece):
		self.connected.add(piece)
		piece.connected.add(self)

#---------------------------------------------------------------------------------------------------------
# 7.7 Chat Server: Explain how you would design a chat server. In particular, provide details about the
# various backend components, classes, and methods. What would be the hardest problems to solve?

# Could be done cleaner with sets, but oh well

class ChatServer:

	def __init__(self):
		self.chats = {}
		self.chat_count = 0

	def createChat(self, info):
		chat = Chat(self.chat_count, info)
		self.chats[chat.id] = chat
		self.chat_count+=1
		return chat

	def deleteChat(self,chatId):
		if not chatId in self.chats: return False
		for user in self.chats[chatId].users: user.leaveChat(chatId)
		self.chats.pop(chatId)
		return True

class Chat:

	def __init__(self, id, info):
		self.id = id
		self.info = info
		self.users = {}
		self.messages = []

	def addUser(self,user):
		self.users[user.id] = user

	def removeUser(self,userId):
		if not userId in self.users: return False
		self.users.pop(userId.id)
		return True
	
	def addMessage(self,userId, message):
		self.messages.append((userId,message))

class User:

	def __init__(self,id,info):
		self.id = id
		self.info = info
		self.chats = {}
		self.sent_messages = []
		self.current_chat = None
	
	def getId(self):
		return self.id
	def setID(self,id):
		self.id = id
	def getChats(self):
		return self.chats

	def createChat(self,info):
		chat = ChatServer.createChat(info)
		chat.addUser(self)
		self.chats[chat.id] = chat
		self.current_chat = chat

	def leaveChat(self,chatId):
		self.chats[chatId].removeUser(self.id)
		self.chats.pop(chatId)

	def setCurrentChat(self, chat):
		self.current_chat = chat

	def sendMessage(self, message):
		self.current_chat.addMessage((self.id,message))
		self.sent_messages = ((self.current_chat.id,message))
	

#---------------------------------------------------------------------------------------------------------
# 7.8 Othello: Othello is played as follows: Each Othello piece is white on one side and black on the other.
# When a piece is surrounded by its opponents on both the left and right sides, or both the top and
# bottom, it is said to be captured and its color is flipped. On your turn, you must capture at least one
# of your opponent's pieces. The game ends when either user has no more valid moves. The win is
# assigned to the person with the most pieces. Implement the object-oriented design for Othello.

PIECE_COLOR = ('B', 'W')

class Othello:

	def __init__(self,rows,cols):
		self.rows = rows
		self.cols = cols
		self.board = Board(rows,cols)
		self.players = [Player(1,'B',self), Player(2,'W',self)]

	def getBoard(self):
		return self.board

	def play(self):
		"""Logic for game play in here"""
		pass


class Board:

	def __init__(self,rows,cols):
		self.board = [[None for _ in range(cols)] for _ in range(rows)]
		self.score_white = 0
		self.score_black = 0

	def setPiece(self,row,col,piece):
		self.board[row][col] = piece
		self.flipPieces(row,col,piece.color,direction)

	def flipPieces(self, row, color, direction):
		# add logic for flipping consecutive pieces and get score_delta
		self.updateScore(color, score_delta)

	def getScore(self,color):
		if color == 'B': return self.score_black
		return self.score_white
	
	def updateScore(self,color, score_delta):
		if color == 'B': 
			self.score_black += score_delta
			self.score_white -= score_delta
		else: 
			self.score_black -= score_delta
			self.score_white += score_delta

class Piece:

	def __init__(self,color):
		self.color = color
	def getColor(self):
		return self.color
	def setColor(self, color):
		self.color = color
	def toggleColor(self):
		self.setColor('B') if self.color == 'W' else self.setColor('W')

class Player:

	def __init__(self,id,color):
		self.id = id
		self.color = color

	def getId(self):
		return self.id
	def getColor(self):
		return self.color

#---------------------------------------------------------------------------------------------------------
# 7.9 Circular Array: Implement a CircularArray class that supports an array-like data structure which
# can be efficiently rotated. If possible, the class should use a generic type (also called a template), and
# should support iteration via the standard for (Obj o : circularArray) notation.


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

# Time Complexity: O(R*C) therefore N (number of grid spaces), Space Complexity: O(R+C)
def robot_in_grid(grid):

	def helper(i,j,path):
		if i>len(grid)-1 or j>len(grid[0])-1 or grid[i][j]==1: return None
		if i == len(grid)-1 and j == len(grid[0])-1: return path
		if (i,j) in cache: return None
		cache.add((i,j))
		return helper(i+1,j,path+[(i+1,j)]) or helper(i,j+1,path+[(i,j+1)])
	
	cache = set()
	return helper(0,0,[(0,0)])

# Time Complexity: O(R*C) therefore N (number of grid spaces), Space Complexity: O(N)
def robot_in_grid_dp(grid):

	dp = [[0 for _ in range(len(grid[0]))] for _ in range(len(grid))]


g = [[0 for _ in range(5)] for _ in range(5)]
g[4][0] = 1
g[4][3] = 1
print(robot_in_grid(g))