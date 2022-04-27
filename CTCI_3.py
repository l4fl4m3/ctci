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
