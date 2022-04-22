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
#diff:0 1 0 1 2 3 2 3 2

# Time Complexity: O(N), Space Complexity: O(N)
def letters_and_numbers(arr):
    if not arr or len(arr) <2: return None

    count_n = [0 for i in range(len(arr))]
    count_l = [0 for i in range(len(arr))]
    diffs = [0 for i in range(len(arr)+1)]
    first_occurence = {0:0}
    cur_max = 0
    max_idxs = (0,0)

    for i in range(len(arr)):
        count_n[i] = count_n[i-1]
        count_l[i] = count_l[i-1]
        if type(arr[i]) == int: count_n[i]+=1
        else: count_l[i]+=1
        diffs[i+1] = count_n[i] - count_l[i]
        if diffs[i+1] not in first_occurence: first_occurence[diffs[i+1]] = i
        else: 
            if (i - first_occurence[diffs[i+1]]) > cur_max:
                cur_max = i - first_occurence[diffs[i+1]]
                max_idxs = (first_occurence[diffs[i+1]]+1,i)
    
    return arr[max_idxs[0]:max_idxs[1]+1]

#print(letters_and_numbers([1,'a',5,7,9,'b',4,'f','g',5]))

#---------------------------------------------------------------------------------------------------------
# 17.6 Count of 2s: Write a method to count the number of 2s between O and n.

# 0-15 -> [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15] -> 2 (0010), 12 (1100) -> 2 
# 0-20 -> [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20] -> 2 (00010), 12 (01100), 20 (10100) -> 3
# 0-22 -> [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22] -> 2 (00010), 12 (01100), 20 (10100), 21 (10101), 22 (10110) -> 6
# 22 %10 = 2, 22 - 22%10 = 20 

# 61523
# 0 digit: 30//10, 1 digit:  , 2 digit: 2000//10, 3 digit: 60000//10, 4 digit: 100000/10

# Time Complexity: O(logN), Space Complexity: O(1)
def count_of_2s(n):

    def twos_at_digit(num, digit):

        digit_val = (num // (10**(digit))) % 10
        next_digit_val = (num // (10**(digit+1))) % 10
        rounded_up = (next_digit_val+1)*(10**(digit+1))
        rounded_down = (next_digit_val)*(10**(digit+1))
        count_right = num % (10**digit)
        if digit_val<2: return rounded_down//10
        if digit_val>2: return rounded_up//10
        else: return rounded_down//10 + count_right +1

    count = 0
    num_digits = int(math.log10(n))+1
    for i in range(num_digits): count+=twos_at_digit(n, i)
    return count

#print(count_of_2s(29))

#---------------------------------------------------------------------------------------------------------
# 17.7 Baby Names: Each year, the government releases a list of the 10,000 most common baby names
# and their frequencies (the number of babies with that name). The only problem with this is that
# some names have multiple spellings. For example, "John" and "Jon" are essentially the same name
# but would be listed separately in the list. Given two lists, one of names/frequencies and the other
# of pairs of equivalent names, write an algorithm to print a new list of the true frequency of each
# name. Note that if John and Jon are synonyms, and Jon and Johnny are synonyms, then John and
# Johnny are synonyms. (It is both transitive and symmetric.) In the final list, any name can be used
# as the "real" name.
# EXAMPLE
# Input:
# Names: John (15), Jon (12), Chris (13), Kris (4), Christopher (19)
# Synonyms: (Jon, John), (John, Johnny), (Chris, Kris), (Chris, Christopher)
# Output: John (27), Kris (36)

# Time Complexity: O(N+S), Space Complexity: O(N) , where N=# baby names, S= pair of synonyms
def baby_names(names, equivalents):

    names_map = {k:v for k,v in names}
    def make_graph():
        adj_list = {}
        for eqv in equivalents:
            if eqv[0] in adj_list: adj_list[eqv[0]].append(eqv[1]) 
            else: adj_list[eqv[0]] = [eqv[1]]

            if eqv[1] in adj_list: adj_list[eqv[1]].append(eqv[0]) 
            else: adj_list[eqv[1]] = [eqv[0]] 
        # add any remaining names, as solo nodes 
        for name in names: 
            if name[0] not in adj_list: adj_list[name[0]] = [name[0]] 

        return adj_list
    
    def dfs_helper(vertex):
        if vertex in visited: return 0
        visited.add(vertex)
        c = names_map[vertex] if vertex in names_map else 0
        for adj in graph[vertex]: c+= dfs_helper(adj)
        return c

    def dfs():
        for v in graph:
            if v not in visited:
                count = dfs_helper(v)
                res.append((v,count))

    graph = make_graph()
    visited = set()
    res = []
    dfs()
    print(res)
    return res

'''
n = [('John', 15), ('Jon',12), ('Chris',13), ('Kris',4), ('Christopher',19), ('Chrissy',3)]
e = [('Jon', 'John'), ('John', 'Johnny'), ('Chris', 'Kris'), ('Chris', 'Christopher')]

baby_names(n,e)'''

#---------------------------------------------------------------------------------------------------------
# 17.8 Circus Tower: A circus is designing a tower routine consisting of people standing atop one another's
# shoulders. For practical and aesthetic reasons, each person must be both shorter and lighter than
# the person below him or her. Given the heights and weights of each person in the circus, write a
# method to compute the largest possible number of people in such a tower.

# Time Complexity: O(N^2), Space Complexity: O(N), N=# of people in circus
# Used a variation of box stacking problem, basically sort, then LIS
def circus_tower_dp(people):

    people.sort(key=lambda x: x[0], reverse=True)
    print(people)
    dp = [people[i][0] for i in range(len(people))]
    for i in range(1,len(people)):
        for j in range(i):
            if people[i][1]< people[j][1] and people[i][0]<people[j][0]: dp[i] = max(dp[i], dp[j]+people[i][0])

    print(dp)
    print(max(dp))
    return max(dp)


'''
p = [(6,100),(2,110),(5,95),(3,90),(4,100),(6,110),(6,95),(6,85), (5,115), (2,80)]
circus_tower_dp(p)'''
# version to show actual list of people
def circus_tower_dp_2(people):

    people.sort(key=lambda x: x[0], reverse=True)
    print(people)
    dp = [[people[i][0],[i]] for i in range(len(people))]
    for i in range(1,len(people)):
        for j in range(i):
            if people[i][1]< people[j][1] and people[i][0]<people[j][0]:
                if dp[j][0]+people[i][0]> dp[i][0]:
                    dp[i][0] = dp[j][0]+people[i][0]
                    dp[i][1] = dp[j][1] + [i]


    print(dp)
    print(max(dp))
    return max(dp)

'''
p = [(6,100),(2,110),(5,95),(3,90),(4,100),(6,110),(6,95),(6,85), (5,115), (2,80)]
circus_tower_dp_2(p)'''

#---------------------------------------------------------------------------------------------------------
# 17.9 Kth Multiple: Design an algorithm to find the kth number such that the only prime factors are 3, 5,
# and 7. Note that 3, 5, and 7 do not have to be factors, but it should not have any other prime factors.
# For example, the first several multiples would be (in order) 1, 3, 5, 7, 9, 15, 21.

# kth multiple = (3^a) * (3^b) * (3^c), some combination of a b and c
# 1,   3, 5, 7,   9, 15, 21,   25, 27, 35   45, 49, 63
 
from collections import deque 
# Time Complexity: O(k), Space Complexity: O(k)
def kth_multiple(k):
    threes, fives, sevens = deque(),deque(),deque() # could make our own queue class as well, this was just easier to implement
    x = 1
    for i in range(1,k):
        threes.append(x*3)
        fives.append(x*5)
        sevens.append(x*7)
        x = min(threes[0], fives[0], sevens[0])
        if x == threes[0]: 
            x = threes.popleft()
        if x == fives[0]:
            x = fives.popleft()
        if x == sevens[0]: 
            x = sevens.popleft()

    print(x)
    return x

#kth_multiple(40)

#---------------------------------------------------------------------------------------------------------
# 17.10 Majority Element: A majority element is an element that makes up more than half of the items in
# an array. Given a positive integers array, find the majority element. If there is no majority element,
# return -1. Do this in O(N) time and 0(1) space.
# Input: 1 2 5 9 5 9 5 5 5
# Output: 5

# Time Complexity: O(N), Space Complexity: O(1)
def majority_element(arr):
    
    length = len(arr)
    max_consecutive = 0
    max_consecutive_val = 0
    curr_consecutive = 1
    for i in range(1,len(arr)):
        if arr[i-1] == arr[i]: 
            curr_consecutive+=1
            if curr_consecutive>max_consecutive: 
                max_consecutive = curr_consecutive
                max_consecutive_val = arr[i]
        else: curr_consecutive = 1

    if length%2==1:
        if arr[-1] == arr[0]:
            curr_consecutive+=1
            if curr_consecutive>max_consecutive: 
                max_consecutive = curr_consecutive
                max_consecutive_val = arr[-1]
    count=0
    for i in range(len(arr)):
        if arr[i] == max_consecutive_val: count+=1

    if count>= length//2 + 1: return max_consecutive_val
    return -1

a = [1,2,9, 5, 9, 5, 5, 5]
print(majority_element(a))