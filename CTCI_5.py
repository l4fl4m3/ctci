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
	