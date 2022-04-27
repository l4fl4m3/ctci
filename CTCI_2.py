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
