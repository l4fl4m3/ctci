#---------------------------------------------------------------------------------------------------------
# 17.1 Add Without Plus: Write a function that adds two numbers. You should not use + or any arithmetic
# operators.

'''
4  -> 0100    29 -> 011101    4   -> 00100 
9  -> 1001    11 -> 001011    13  -> 01101

13 -> 1101    40 -> 101000    17  -> 10001
'''

# Assuming only positive integers, if psoitive and negative, will need to implement subtraction function
# Time Complexity: O(logb) where b = second number, Space Complexity: O(1)
def add_without_plus(a,b):
	""" Think of a as current sum, b as to add. Get bits that will carry over with &, get bits that will be 
	set (to 1) with ^ -> set this to a (current sum). Shift left all carry bits by 1 -> set this to b (to add to current sum).
	Repeat this until there is no more to add (no more b) """

	def add(a,b):
		while b:
			carry = a&b
			a = a ^ b
			b = carry<<1
		return a
	
	def subtract(a,b):
		# TODO
		pass


	return add(a,b)


#print(add_without_plus(8,8))

#---------------------------------------------------------------------------------------------------------
# 17.2 Shuffle: Write a method to shuffle a deck of cards. It must be a perfect shuffle-in other words, each
# of the 52! permutations of the deck has to be equally likely. Assume that you are given a random
# number generator which is perfect.

# random.randint(0,51) -> P(x) = 1/52, random.randint(0,50) -> P(x) = 1/51, so everytime we generate a random number within the range
# of current available cards and select that number to be our card (this number will map to some card). We do this 52 times, and each 
# time our set of cards shrinks. Therefore P(any permutation) = (1/52)*(1/51)...(1/2)*(1/1), we just need to make sure that we are 
# appropriately shrinking our card space each time, and only leave non-selected cards, this makes sure that the probably for a unique
# permutation stays uniform.

# Time Complexity: O(N^2) where N = number of cards in deck, Space Complexity: O(N)
from itertools import count
import random
def shuffle_naive():
	deck = []
	avail = [i for i in range(52)]
	a,b=0,51
	while b>=0:
		random_card = random.randint(a,b)
		deck.append(avail[random_card])
		avail.pop(random_card)
		b-=1

	return deck

#print(shuffle())

# This is similar to the explanation above, but built from the bottom up, can also do random.randint(i,51) and and swapping
# Time Complexity: O(N), Space Complexity: O(1)
def shuffle(deck):
	
	for i in range(len(deck)):
		rand_card = random.randint(0,i)
		deck[rand_card], deck[i] = deck[i], deck[rand_card]

'''
d = [i for i in range(1,53)]
shuffle(d)
print(d)'''

#---------------------------------------------------------------------------------------------------------
# 17.3 Random Set: Write a method to randomly generate a set of m integers from an array of size n. Each
# element must have equal probability of being chosen.

# [5,2,8,9,4,2] , m = 4 , n = 6 -> P(x) = (1/6) + (5/6)*((1/5) + (4/5)*((1/4) + (3/4)*(1/3))) = 2/3 (which equals 4/6)
# Time Complexity: O(N^2), Space Complexity: O(M) -> if we are allowed to (structurally) change the input array, otherwise O(N+M) -> O(N)
def random_set_naive(m,n,arr):
	r_set = []
	for i in range(m):
		rand = random.randint(0,n-1-m)
		r_set.append(arr[rand])
		arr.pop(rand)
	return r_set

''''
a = [5,2,8,9,4,2]
print(random_set_naive(4,len(a),a))
print(a)'''

# Time Complexity: O(N), Space Complexity: O(M) -> O(1) if we can just return continuous range in arr where set is stored
def random_set(m,n,arr):

	r_set = []
	for i in range(m):
		rand = random.randint(i,n-1)
		r_set.append(arr[rand])
		arr[i], arr[rand] = arr[rand], arr[i]

	return r_set
'''
a = [5,2,8,9,4,2]
print(random_set(4,len(a),a))'''

#---------------------------------------------------------------------------------------------------------
# 17.4 Missing Number: An array A contains all the integers from O to n, except for one number which
# is missing. In this problem, we cannot access an entire integer in A with a single operation. The
# elements of A are represented in binary, and the only operation we can use to access them is "fetch
# the jth bit of A[i];' which takes constant time. Write code to find the missing integer. Can you do it
# in O(n) time?
#
# [0,1,2,3,4,6] -> [000, 001, 010, 011, 100, 110], missing: 101 (5)
#  
#  			  XOR:       001  011  000  100  001  111 -> res
# if contains all: [000, 001, 010, 011, 100, 101, 110] -> res would be ~miss# if number from range missing
#
# [0,1,2,3,4,5,6,7,9] -> [0000, 0001, 0010, 0011, 0100, 0101, 0110, 0111, 1001], missing: 1000 (8)
#  
#  			  XOR:        0001  0011  0000  0100  0001  0111  0000  1000  0001
# if contains all: [0000, 0001, 0010, 0011, 0100, 0101, 0110, 0111, 1000, 1001] -> res would be NOT be ~miss#, need to xor all remaning comb of bits aswell
# 
import math
# Time Complexity: O(NlogN), Space Complexity: O(1)
def missing_number_nonoptimal(arrA):

    n = len(arrA)
    num_bits = math.ceil(math.log2(n+1))
    ac_sum,arr_sum = 0,0
    for i in range(n+1): ac_sum += i
    for num in arrA:
        ac_num = 0
        for i in range(num_bits): ac_num = ac_num | ((1&int(num[-1-i]))<<i)
        arr_sum+=ac_num
    return ac_sum - arr_sum

    '''
    n = len(arrA)
    num_bits = math.ceil(math.log2(n+1))
    res = 0
    count = 0
    for num in arrA:
        ac_num = 0
        for i in range(num_bits):
            ac_num = ac_num | ((1&int(num[-1-i]))<<i)
        res = res^ac_num
        count+=1

    for n in range(n+1, 2**num_bits): 
        print(f"n: {n:08b}")
        res = res ^ n
        count+=1
    print(count)
    return res
    '''

'''
#a = ['000', '001', '010', '011', '100', '110', '101']
a = ['0000', '0001', '0010', '0011', '0100', '0101', '0110', '0111', '1001']
r = missing_number_nonoptimal(a)
print(f"NUM: {r} -> {r:08b}")'''

# Bit representation of 0 to n, least significant bit always -> 0's >= 1's (> by 1 if n is even, == if n is odd).
# Therefore, if one number is removed, #0s will be <= #1s if LSB was 0, #0s will be > #1s id LSB was 1, this holds for each bit ->
# #0's >= #1's
 
# Time Complexity: O(N + N/2 + ...) -> O(N), Space Complexity: O(N)
def missing_number(arrA):
    
    def helper(nums_to_check, lsb):
        print(nums_to_check)
        if not nums_to_check: return 0
        zeros,ones = [],[]
        for n in nums_to_check:
            if arrA[n][-1-lsb] == '0': zeros.append(n)
            if arrA[n][-1-lsb] == '1': ones.append(n)
            
        if len(zeros)<=len(ones): return 0 | (helper(zeros, lsb+1)<<1)
        else: return 1 | (helper(ones, lsb+1)<<1)

    return helper([i for i in range(len(arrA))], 0)

'''
a = ['0000', '0001', '0010', '0011', '0100', '0101', '0110', '0111', '1001']
r = missing_number(a)
print(f"NUM: {r} -> {r:08b}")'''

#---------------------------------------------------------------------------------------------------------
# 17.5 Letters and Numbers: Given an array filled with letters and numbers, find the longest subarray with
# an equal number of letters and numbers.

# [1,a,5,7,9,b,4,f]
# Time Complexity: O(N^2), Space Complexity: O(N)
def letters_and_numbers_naive(arr):
    if not arr or len(arr) <2: return None

    def helper(nums, letters, curr, arr):
        if nums == letters and curr: res.append(curr)
        if not arr: return

        if type(arr[0]) == int: helper(nums+1, letters, curr+[arr[0]], arr[1:])
        else: helper(nums, letters+1, curr+[arr[0]], arr[1:])

    res = []
    for i in range(len(arr)): helper(0,0,[],arr[i:])
    max_s = 0
    for i in range(len(res)): 
        if len(res[i])>len(res[max_s]): max_s = i

    return res[max_s]
'''
a = [1,'a',5,7,9,'b',4,'f']
print(letters_and_numbers_naive(a))'''

# [1,'a',5,7,9,'b',4,'f']
#
# dp = [1,1,2,3,4,4,5,5]
#      [0,1,1,1,1,2,2,3]
def letters_and_numbers_dp(arr):
    if not arr or len(arr) <2: return None

    count_n = [0 for i in range(len(arr))]
    count_l = [0 for i in range(len(arr))]
    for i in range(len(arr)):
        count_n[i] = count_n[i-1]
        count_l[i] = count_l[i-1]
        if type(arr[i]) == int: count_n[i]+=1
        else: count_l[i]+=1
    print(count_n)
    print(count_l)


letters_and_numbers_dp([1,'a',5,7,9,'b',4,'f'])