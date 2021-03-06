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
import re
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
'''
a = [1, 2, 5, 9, 5, 9, 5, 5, 5]
print(majority_element(a))'''

# Time Complexity: O(N), Space Complexity: O(1)
def majority_element_clean(arr):
    ''' This works since like above, we are looking for a consecutive element (the longest continuous one). This will ALWAYS 
    be 1 if there is a majority element, the final max_count will be equal to 1, and would not have reset to a random
    index, but rather and index of the majority element. It can be 1 on other occasions aswell, so we check the count of our
    proposed major element in the second for loop. Since majority element count is n//2 + 1, while all others are
    n//2 - 1, so if majority element, max_count will have incremented and decremented but at the end would be at least 1.'''

    max_count, max_consecutive_idx = 1,0
    for i in range(len(arr)):
        if arr[i] == arr[max_consecutive_idx]: max_count+=1
        else: max_count-=1

        if not max_count:
            max_count = 1
            max_consecutive_idx=i

    count = 0
    for i in range(len(arr)):
        if arr[i] == arr[max_consecutive_idx]: count+=1

    return arr[max_consecutive_idx] if count>len(arr)//2 else -1

'''
a = [5,3,5,3,5,3,5]
print(majority_element_clean(a))'''

#---------------------------------------------------------------------------------------------------------
# 17.11 Word Distance: You have a large text file containing words. Given any two words, find the shortest
# distance (in terms of number of words) between them in the file. If the operation will be repeated
# many times for the same file (but different pairs of words), can you optimize your solution?

# Time Complexity: O(N) per word pair search, Space Complexity: O(1)
def word_distance_simple(words, word_pairs):

    def helper(word1, word2):
        
        pos_1,pos_2 = -1,-1
        f_word, s_word = word1, word2
        dist = float('inf')
        for i in range(len(words)):
            if words[i] == word1: pos_1 = i
            if words[i] == word2: pos_2 = i
            if pos_1>=0 and pos_2>=0: dist = min(dist, abs(pos_1-pos_2))

        return dist-1
    
    # using this to repeat operation
    for pair in word_pairs:
        r = helper(pair[0], pair[1])
        print(r)
'''
a = ['this', 'is', 'a', 'test', 'paragraph', 'that', 'I', 'will', 'be', 'using', 'The', 'paragraph', 'will', 'contain', 'words']
word_pairs = [('this', 'test'), ('using', 'paragraph'), ('will', 'paragraph')]

word_distance_simple(a, word_pairs)'''

# Time Complexity: O(N) for first search O(A+B) for all subsequent where A=#occurences word1, B=#occurences word2, Space Complexity: O(N)
def word_distance(words, word_pairs):

    # O(N)
    def build_location_dict():
        for i in range(len(words)):
            if words[i] not in locations: locations[words[i]] = [i]
            else: locations[words[i]].append(i)
    
    # O(A+B)
    def helper(word1, word2):
        i,j = 0,0
        min_distance = float('inf')
        while i<len(locations[word1]) and j<len(locations[word2]):
            min_distance = min(min_distance, abs(locations[word1][i] - locations[word2][j]))
            if locations[word1][i]<= locations[word2][j]: i+=1
            else: j+=1
        
        return min_distance-1
        
    locations = {}
    build_location_dict()

    # using this to repeat operation
    for pair in word_pairs:
        r = helper(pair[0], pair[1])
        print(r)

'''
a = ['this', 'is', 'a', 'test', 'paragraph', 'that', 'I', 'will', 'be', 'using', 'The', 'paragraph', \
     'will', 'contain', 'words', 'it', 'will', 'have', 'a', 'paragraph']
word_distance(a, [('this', 'test'), ('using', 'paragraph'), ('will', 'paragraph')])'''

#---------------------------------------------------------------------------------------------------------
# 17.12 BiNode: Consider a simple data structure called BiNode, which has pointers to two other nodes. The
# data structure BiNode could be used to represent both a binary tree (where node1 is the left node
# and node2 is the right node) or a doubly linked list (where node1 is the previous node and node2
# is the next node). Implement a method to convert a binary search tree (implemented with BiNode)
# into a doubly linked list. The values should be kept in order and the operation should be performed
# in place (that is, on the original data structure).

# Tough question, need to think hard as a MF
class BiNode:
    def __init__(self, val):
        self.val = val
        self.pointer1 = None
        self.pointer2 = None

# Time Complexity: O(N), Space Complexity: O(1)
def convert_bst_to_dll(root):

    def traverse(root, parent):
        if not root: return None

        if not root.pointer2 and parent:
            root.pointer2 = parent 
            parent.pointer1 = root
        right= traverse(root.pointer2, parent)
        left = traverse(root.pointer1, root)
        
        if right:
            root.pointer2 = right
            right.pointer1 = root 
            if parent: 
                right.pointer2 = parent
                parent.pointer1 = right

        return left if left else root

    return traverse(root, None)
'''
n1 = BiNode(5)
n2 = BiNode(4)
n3 = BiNode(3)
n4 = BiNode(2)
n5 = BiNode(7)
n6 = BiNode(8)
n7 = BiNode(9)
n1.pointer1 = n2
n1.pointer2 = n7
n2.pointer1 = n4
n4.pointer2 = n3
n7.pointer1 = n5
n5.pointer2 = n6

def inorder(root):
    if not root: return None
    inorder(root.pointer1)
    print(root.val, end=" ")
    inorder(root.pointer2)

inorder(n1)
print()

dll = convert_bst_to_dll(n1)
while dll:
    prev = dll.pointer1.val if dll.pointer1 else "NONE"
    next = dll.pointer2.val if dll.pointer2 else "NONE"
    curr = dll.val if dll else "NONE"
    print(f"Prev: {prev}, Current:{curr}, Next: {next}")

    dll=dll.pointer2
'''

#---------------------------------------------------------------------------------------------------------
# 17.13 Re-Space: Oh, no! You have accidentally removed all spaces, punctuation, and capitalization in a
# lengthy document. A sentence like "I reset the computer. It still didn't boot!"
# became "iresetthecomputeritstilldidntboot''. You'll deal with the punctuation and capitalization
# later; right now you need to re-insert the spaces. Most of the words are in a dictionary but
# a few are not. Given a dictionary (a list of strings) and the document (a string), design an algorithm
# to unconcatenate the document in a way that minimizes the number of unrecognized characters.
# EXAMPLE
# Input jesslookedjustliketimherbrother
# Output: 'jess' looked just like 'tim' her brother (7 unrecognized characters)

# in the above example jess + tim == 7 chars, since they are names, presumably they are not a part of the dictionary

# Time Complexity: O(N^2), Space Complexity: O(N)
def re_space(document, dictionary):

    def helper(document, count, us_list):
        global res
        if count in cache:
            if count<res[0]: res = (count, cache[count])
            return
        if not document:
            if count<res[0]: res = (count, us_list)
            cache[count] = us_list
            return
            
        for i in range(1,len(document)+1): 
            if document[:i] in dictionary:
                helper(document[i:],count,us_list+[document[:i]])
                
        helper(document[1:],count+1,us_list)
        

    global res
    res = (float('inf'), None)
    cache={}
    helper(document,0,[])
    print(res)

'''   
doc = "jesslookedjustliketimherbrother"
dic = {'looked', 'like', 'just', 'her', 'brother', 'look', 'other', 'bro'}

re_space(doc,dic)'''

#---------------------------------------------------------------------------------------------------------
# 17.14 Smallest K: Design an algorithm to find the smallest K numbers in an array.

# Time Complexity: O(NlogN), Space Complexity: O(k), or O(1) if we just return indexes (ie, 0,k) since sorted array
def smallest_k_naive(arr, k):

    arr.sort()
    return arr[0:k]

'''
a = [5, 9, -44, 16, 3, -4, 76, 66, -8, 240, 44, 37, -8, 9, 5]
print(smallest_k_naive(a,5))'''

# Time Complexity: O(N*logk), Space Complexity: O(k)
def smallest_k(arr, k):

    # O(logk)
    def maxHeapify(heap, idx):

        left_child = idx*2 +1
        right_child = idx*2 +2

        largest = idx
        if left_child<len(heap) and heap[left_child]>heap[largest]: largest = left_child
        if right_child<len(heap) and heap[right_child]>heap[largest]: largest = right_child
        if largest !=idx: 
            heap[idx], heap[largest] = heap[largest], heap[idx]
            maxHeapify(heap, largest)

    def insertHelper(heap, idx):
        parent = (idx-1)//2
        if parent>=0:
            if heap[parent]<heap[idx]:
                heap[parent], heap[idx] = heap[idx], heap[parent]
                insertHelper(heap, parent)

    def insert(heap,val):
        heap.append(val)
        insertHelper(heap,len(heap)-1)

    def deleteMax(heap):
        heap[0] = heap.pop()
        maxHeapify(heap,0)

    # O(k) (looks like it should be klogk, but is actually k)
    def buildMaxHeap(heap):
        for i in range((len(heap)//2)-1,-1,-1):
            maxHeapify(heap,i)
 
    heap = [arr[i] for i in range(k)]
    buildMaxHeap(heap)
    for i in range(k,len(arr)):
        if arr[i]<heap[0]:
            deleteMax(heap)
            insert(heap, arr[i])

    print(heap)
    return heap

'''
a = [5, 16, -44, 9, 3, -4, 76, 66, -8, 240, 44, 37, -8, 9, 5]
smallest_k(a,5)'''

#---------------------------------------------------------------------------------------------------------
# 17.15 Longest Word: Given a list of words, write a program to find the longest word made of other words
# in the list.

# Time Complexity: O(N*L), Space Complexity: O(N), where L is the length of longest word
def longest_word(words):
    
    def helper(word, pre, count):
        global longest
        if not word:
            if not longest or len(pre) > len(longest) and count>1: longest = pre
            return
        for i in range(1,len(word)+1):
            if word[:i] in word_dic: helper(word[i:], pre+word[:i], count+1)
    
    global longest
    longest = None
    word_dic = {w:[] for w in words}
    words.sort(key=lambda x: len(x), reverse=True)
    print(words)
    for w in words:
        helper(w,'',0)
    print(longest)
    return longest

'''
w = ['long', 'longer', 'toothpick', 'abrarearrangingtoothpickrearrearranging', 'rearranging', 'tooth', 'pick', 'rear', 'ranging', 'abra','cada','bra','shoot', 'abracadabrashoot']
longest_word(w)'''

#---------------------------------------------------------------------------------------------------------
# 17.16 The Masseuse: A popular masseuse receives a sequence of back-to-back appointment requests
# and is debating which ones to accept. She needs a 15-minute break between appointments and
# therefore she cannot accept any adjacent requests. Given a sequence of back-to-back appointment
# requests (all multiples of 15 minutes, none overlap, and none can be moved), find the optimal
# (highest total booked minutes) set the masseuse can honor. Return the number of minutes.
# EXAMPLE
# Input: {30, 15, 60, 75, 45, 15, 15, 45}
# Output: 180 minutes ({30, 60, 45, 45}).

# Time Complexity: O(2^N), Space Complexity: O(N)
def masseuse_naive(requests):

    def helper(requests,mins, appts):
        global optimal
        if not requests:
            if not optimal[0] or optimal[0]<mins: optimal = (mins, appts)
            return

        helper(requests[2:],mins+requests[0],appts+[requests[0]])
        helper(requests[1:],mins,appts)

    global optimal
    optimal = (None,None)
    helper(requests,0,[])
    print(optimal)
    return optimal
'''
#a=[30, 15, 60, 75, 45, 15, 15, 45]
b = [15,15,15,75,15]
masseuse_naive(b)'''

# Time Complexity: O(N), Space Complexity: O(N) 
def masseuse_memo(requests):

    def helper(idx):
        
        if idx >= len(requests): return 0
        if idx in memo: return memo[idx]
        choose = requests[idx]+helper(idx+2)
        dont_choose = helper(idx+1)
        memo[idx] = max(choose,dont_choose)

        return memo[idx]

    memo = {}
    optimal = helper(0)
    print(optimal)
    return optimal

'''
a = [30, 15, 60, 75, 45, 15, 15, 45]
b = [75,105,120,75,90,135]
masseuse_memo(a)'''

def masseuse_optimized(requests):
    '''TODO'''
    pass
#---------------------------------------------------------------------------------------------------------
# 17.17 Multi Search: Given a string b and an array of smaller strings T, design a method to search b for
# each small string in T.

# use trie
# Time Complexity: O(b*k + t*k), Space Complexity: O(bt) ? (to store all occurences)
# where b=length of string b, t=# of strings in T, k=length of longest t

def multi_search(b, T):
    
    # O(tk)
    def makeTrie():
        tr = {}
        for word in T:
            t = tr
            for c in word:
                if c not in t:
                    t[c] ={}
                    t=t[c]
                else: t = t[c]
            t[0] = -1
        return tr

    trie = trie_c = makeTrie()
    results = []
    word = ''

    # O(bk)
    for i in range(len(b)):
        if b[i] in trie:
            j = i
            while j<len(b) and b[j] in trie:
                word+=b[j]
                trie = trie[b[j]]
                if 0 in trie: results.append((word, (i,j)))
                j+=1
        
            trie = trie_c
            word=''
    
    print(results)
    return results

'''
b = "This is a regular string to be searched"
t = ['This', 'is', 'searched', 'search', 'string']
multi_search(b,t)'''

#---------------------------------------------------------------------------------------------------------
# 17.18 Shortest Supersequence: You are given two arrays, one shorter (with all distinct elements) and one
# longer. Find the shortest subarray in the longer array that contains all the elements in the shorter
# array. The items can appear in any order.
# EXAMPLE
# Input:
# {1, 5, 9}
# {7, 5, 9, 0, 2, 1, 3, 5, 7, 9, 1, 1, 5, 8, 8, 9, 7}
# Output:[7, 10] (the underlined portion above)

# Time Complexity: O(s*l^2), Space Complexity: O(N) where l = length of longer array, and s = length of shorter array
def shortest_supersequence_naive(shorter, longer):
    
    def helper(idx, start):
        global shortest
        if not d:
            if not shortest[1] or ((idx-1)-start+1 <shortest[0]): shortest = ((idx-1)-start+1,(start, idx-1))
            return
        if idx>=len(longer): return None
        if longer[idx] in d:

            d[longer[idx]] -=1
            if d[longer[idx]]==0: d.pop(longer[idx])
            
            if not start: helper(idx+1, idx)
            else: helper(idx+1, start)

            if longer[idx] not in d: d[longer[idx]] =1
            else: d[longer[idx]] +=1
        

        helper(idx+1, start)
    
    d = {}
    for n in shorter:
        if n in d: d[n]+=1
        else: d[n]=1

    global shortest
    shortest = (float('inf'), None)
    helper(0,None)
    print(shortest)
    print(d)
    return shortest

'''
s = [1, 5, 9]
l = [7, 5, 9, 0, 2, 1, 3, 5, 7, 9, 1, 1, 5, 8, 8, 9, 7]
shortest_supersequence(s,l)'''

# {7, 5, 9, 0, 2, 1, 3, 5, 7, 9, 1, 1, 5, 8, 8, 9, 7}
#5:1, 1, 7, 7, 7, 7, 7, 7,12,12,12,12,12,-1,-1,-1,-1
#1:5, 5, 5, 5, 5, 5,10,10,10,10,10,11,-1,-1,-1,-1,-1
#9:2, 2, 2, 9, 9, 9, 9, 9, 9, 9,15,15,15,15,15,15,-1
# :5, 5, 7, 9, 9, 9,10,10,12,12,15,15,-1,-1,-1,-1,-1  
# Time Complexity: O(s*l), Space Complexity: O(s*l) where l = length of longer array, and s = length of shorter array
def shortest_supersequence(shorter,longer):
    locations = [[-1 for _ in range(len(longer))] for _ in range(len(shorter))]
    for i in range(len(shorter)):
        for j in range(len(longer)-1, -1, -1):
            if longer[j] == shorter[i]: locations[i][j] = j
            else:
                if j<len(longer)-1: locations[i][j] = locations[i][j+1]
    
    shortest = (float('inf'), None)
    for j in range(len(longer)):
        max_idx, valid = float('-inf'), True
        for i in range(len(shorter)):
            if locations[i][j] == -1: 
                valid = False
                break
            max_idx = max(locations[i][j], max_idx)
        
        if valid and (max_idx - j + 1) < shortest[0]: shortest = (max_idx - j + 1, (j, max_idx))

    print(shortest)
    return shortest

'''
s = [1, 5, 9]
l = [7, 5, 9, 0, 2, 1, 3, 5, 7, 9, 1, 1, 5, 8, 8, 9, 7]

shortest_supersequence(s,l)'''

# Time Complexity: O(s*l), Space Complexity: O(l) where l = length of longer array, and s = length of shorter array
def shortest_supersequence_optimized(shorter,longer):

    locations = [0 for _ in range(len(longer))]

    for i in range(len(shorter)):
        carry = -1
        for j in range(len(longer)-1,-1,-1):
                if longer[j] == shorter[i]: carry = j
                if locations[j]!=-1:
                    if longer[j] == shorter[i]: locations[j] = max(locations[j],j)
                    else: locations[j] = max(locations[j],carry) if carry != -1 else -1 # override max when element is not present yet
                    
    shortest = (float('inf'), None)
    for i in range(len(longer)):
        if locations[i] != -1 and locations[i] - i + 1< shortest[0]: shortest = (locations[i] - i + 1, (i, locations[i]))

    print(shortest)
    return shortest

'''
s = [1, 5, 9]
l = [7, 8, 9, 0, 2, 1, 3, 8, 7, 9, 1, 1, 4, 8, 5, 9, 7]

shortest_supersequence_optimized(s,l)'''

#---------------------------------------------------------------------------------------------------------
# 17.19 Missing Two: You are given an array with all the numbers from 1 to N appearing exactly once,
# except for one number that is missing. How can you find the missing number in O(N) time and
# 0(1) space? What if there were two numbers missing?

# [4,1,5,2,6]  4, 1, 5, 2, 0
# [1,3,6,5,2]  1, 3, 0, 5, 2 

# Time Complexity: O(N), Space Complexity: O(N) (stack depth)
def missing_two_naive(arr):

    def calculate_factorial(n):
        if n==1: return 1
        return n*calculate_factorial(n-1)

    n = len(arr)+1
    fac = calculate_factorial(n)
    
    for i in range(len(arr)): fac = fac/arr[i]

    print(fac)
    return fac
'''
a = [4,1,3,2,6,7,8]
missing_two_naive(a)'''

# Time Complexity: O(N), Space Complexity: O(1) , will cause overflow though, since we're doing factorial
def missing_two(arr):
    if not arr: return None

    def calculate_factorial(n):
        fac = 1
        for i in range(1, n+1): fac = fac*i
        return fac

    n = len(arr)+1
    fac = calculate_factorial(n)
    
    for i in range(len(arr)): fac = fac/arr[i]

    print(fac)
    return fac

'''
a = [2,1,4]
missing_two(a)'''

# Time Complexity: O(N), Space Complexity: O(1)
def missing_two_optimized(arr):
    if not arr: return None
    
    n = len(arr)+1
    sum_v = 0
    for i in range(n+1): sum_v += i
    for i in range(len(arr)): sum_v -= arr[i]

    print(sum_v)
    return sum_v

'''
a = [2,4,3,8,5,6,1]
missing_two_optimized(a)'''

# Time Complexity: O(N), Space Complexity: O(1)
# x+y = sum_v, x*y = fac_v, x = sum_v - y -> (sum_v - y)*y = fac_v -> -y^2 + sum_v*y = fac_v -> 0 = y^2 - sum_v*y + fac_v
def missing_two_two(arr):

    def quad_formula(a,b,c):

        v1 = (-b + ((b**2) - 4*a*c)**(1/2))/2
        v2 = (-b - ((b**2) - 4*a*c)**(1/2))/2
        return max(v1, v2)

    if not arr: return None
    
    n = len(arr)+2
    sum_v, fac_v = 0, 1
    for i in range(1, n+1): fac_v *= i
    for i in range(n+1): sum_v += i
    for i in range(len(arr)): 
        sum_v -= arr[i]
        fac_v /= arr[i]

    zero = quad_formula(1, -sum_v, fac_v)

    print((sum_v - zero, zero))
    return (sum_v - zero, zero)
'''
a = [1,8,2,7,3,5]
missing_two_two(a)'''

#---------------------------------------------------------------------------------------------------------
# 17.20 Continuous Median: Numbers are randomly generated and passed to a method. Write a program
# to find and maintain the median value as new values are generated.

class MaxHeap:
    def __init__(self):
        self.heap = []
        self.size = 0

    def heapify(self, idx):
        l_child = idx*2 + 1
        r_child = idx*2 + 2
        greatest = idx

        if l_child < len(self.heap) and self.heap[l_child] > self.heap[idx]: greatest = l_child
        if r_child < len(self.heap) and self.heap[r_child] > self.heap[idx]: greatest = r_child
        if greatest != idx: 
            self.heap[idx], self.heap[greatest] = self.heap[greatest], self.heap[idx]
            self.heapify(greatest)

    def insertHelper(self, idx):
        parent = (idx-1) // 2
        if parent >= 0 and self.heap[parent] < self.heap[idx]:
            self.heap[idx], self.heap[parent] = self.heap[parent], self.heap[idx]
            self.insertHelper(parent)

    def insert(self, val):
        self.heap.append(val)
        self.size +=1
        self.insertHelper(self.size-1)

    def deleteRoot(self):
        root = self.heap[0]
        self.heap[0] = self.heap.pop()
        self.heapify(0)
        self.size -=1
        return root

    def peek(self):
        return self.heap[0]
        

class MinHeap:
    def __init__(self):
        self.heap = []
        self.size = 0

    def heapify(self, idx):
        l_child = idx*2 + 1
        r_child = idx*2 + 2
        smallest = idx

        if l_child < len(self.heap) and self.heap[l_child] < self.heap[idx]: smallest = l_child
        if r_child < len(self.heap) and self.heap[r_child] < self.heap[idx]: smallest = r_child
        if smallest != idx: 
            self.heap[idx], self.heap[smallest] = self.heap[smallest], self.heap[idx]
            self.heapify(smallest)

    def insertHelper(self, idx):
        parent = (idx-1) // 2
        if parent >= 0 and self.heap[parent] > self.heap[idx]:
            self.heap[idx], self.heap[parent] = self.heap[parent], self.heap[idx]
            self.insertHelper(parent)

    def insert(self, val):
        self.heap.append(val)
        self.size +=1
        self.insertHelper(self.size-1)

    def deleteRoot(self):
        root = self.heap[0]
        self.heap[0] = self.heap.pop()
        self.heapify(0)
        self.size-=1
        return root

    def peek(self):
        return self.heap[0]
        
# Time Complexity: O(logN) to insert, O(1) to get median, Space Complexity: O(N), where N is the # of numbers streamed so far
def continuous_median(numbers):

    def helper(n):
        global median
        if min_heap.size == 0 and max_heap.size ==0:
            max_heap.insert(n)
            median = n
            return

        elif max_heap.size == min_heap.size:
            if n <= median: max_heap.insert(n)
            else:
                max_heap.insert(min_heap.deleteRoot()) 
                min_heap.insert(n)

            median = max_heap.peek()
        else:
            if n <= median:
                min_heap.insert(max_heap.deleteRoot())
                max_heap.insert(n)
            else: min_heap.insert(n)

            median = (max_heap.peek() + min_heap.peek())/2

    global median
    min_heap = MinHeap()
    max_heap = MaxHeap()
    # simualte stream
    for i in range(len(numbers)): 
        helper(n[i])
        print(f"MEDIAN: {median}")
'''
n = [3, 66, 12, 99, -23]
continuous_median(n)'''

# Time Complexity: O(logN) to insert, O(1) to get median, Space Complexity: O(N), where N is the # of numbers streamed so far
def continuous_median_clean(numbers):

    def helper(n):
        global median
        if n <= median: max_heap.insert(n)
        else: min_heap.insert(n)
        
        if min_heap.size-1> max_heap.size: max_heap.insert(min_heap.deleteRoot())
        if max_heap.size-1> min_heap.size: min_heap.insert(max_heap.deleteRoot())
        
        if min_heap.size==max_heap.size: median = (max_heap.peek() + min_heap.peek())/2 
        elif min_heap.size>max_heap.size: median = min_heap.peek()
        else: median = max_heap.peek()

    global median
    median = float('inf')
    min_heap = MinHeap()
    max_heap = MaxHeap()
    # simulate stream
    for i in range(len(numbers)): 
        helper(n[i])
        print(f"MEDIAN: {median}")

'''
n = [3, 66, 12, 99, -23, -2, 88]
continuous_median_clean(n)'''

#---------------------------------------------------------------------------------------------------------
# 17.21 Volume of Histogram: Imagine a histogram (bar graph). Design an algorithm to compute the
# volume of water it could hold if someone poured water across the top. You can assume that each
# histogram bar has width 1.
# EXAMPLE
# Input: {0, 0, 4, 0, 0, 6, 0, 0, 3, 0, 5, 0, 1, 0, 0, 0}
#
#           |
#           |         |
#     |     |         |
#     |     |     |   |
#     |     |     |   |
# _ _ | _ _ | _ _ | _ | _ | _ _ _
#
# Output: 26

# Time Complexity: O(N), Space Complexity: O(N)
def volume_of_histogram(histogram):
    '''Sweep left then right, and get max possible volume at location, then subtract out bars'''
    l_max = [0 for i in range(len(histogram))]
    r_max = [0 for _ in range(len(histogram))]
    min_v = [0 for _ in range(len(histogram))]
    l_max[0], r_max[-1] = histogram[0], histogram[-1]
    for i in range(1,len(histogram)): l_max[i] = max(l_max[i-1], histogram[i])
    for i in range(len(histogram)-2,-1,-1): r_max[i] = max(r_max[i+1], histogram[i])
    for i in range(len(histogram)): min_v[i] = min(l_max[i], r_max[i])
    
    final_vol = 0
    for i in range(len(histogram)): final_vol += min_v[i] - histogram[i]
    
    print(final_vol)
    return final_vol

'''
h = [0, 0, 4, 0, 0, 6, 0, 0, 3, 0, 5, 0, 1, 0, 0, 0]
volume_of_histogram(h)'''

# Time Complexity: O(N), Space Complexity: O(N)
def volume_of_histogram_clean(histogram):

    l_max = [0 for i in range(len(histogram))]
    l_max[0] = histogram[0]
    for i in range(1,len(histogram)): l_max[i] = max(l_max[i-1], histogram[i])
    
    r_max = histogram[-1]
    final_vol = 0
    for i in range(len(histogram)-1,-1,-1):
        r_max = max(r_max, histogram[i])
        final_vol += min(r_max, l_max[i]) - histogram[i]
        

    print(final_vol)
    return final_vol
'''
h = [0, 0, 4, 0, 0, 6, 0, 0, 3, 0, 5, 0, 1, 0, 0, 0]
volume_of_histogram_clean(h)'''

#           |
#           |         |
#     |     |         |
#     |     |     |   |
#     |     |     |   |
# _ _ | _ _ | _ _ | _ | _ | _ _ _

# Time Complexity: O(N), Space Complexity: O(1)
def volume_of_histogram_optimized(histogram):

    i,j = 0, len(histogram)-1
    l_max = r_max = volume = 0

    while i<j:
        if histogram[i]<histogram[j]:
            if histogram[i]<l_max: volume += l_max - histogram[i]
            else: l_max = histogram[i]  
            i+=1
        else:
            if histogram[j]<r_max: volume += r_max - histogram[j]
            else: r_max = histogram[j]
            j-=1

    print(volume)
    return volume
'''
h = [0, 0, 4, 0, 0, 6, 0, 0, 3, 0, 5, 0, 1, 0, 0, 0]
volume_of_histogram_optimized(h)'''

#---------------------------------------------------------------------------------------------------------
# 17.22 Word Transformer: Given two words of equal length that are in a dictionary, write a method to
# transform one word into another word by changing only one letter at a time. The new word you get
# in each step must be in the dictionary.
# EXAMPLE
# Input: DAMP, LIKE
# Output: DAMP-> LAMP-> LIMP-> LIME-> LIKE

# Time Complexity: O(N*26*N) -> O(N^2 * K) ?, Space Complexity: O(N) recursion stack
def word_transformer_naive(word1, word2, dictionary):
    '''THIS WONT WORK FOR PATH > LEN(WORD1/2) !!!!!'''
    def helper(word, path, count):
        
        if word not in dictionary: return
        if count == len(word1):
            if word == word2: res.append(path)
            return
        
        for i in range(len(word)):
            for a in avail:
                check = word[:i] + a + word[i+1:]
                if check in dictionary: helper(check, path+[check], count+1)

    res = []
    avail = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    
    helper(word1,[word1],0)
    print(res)
    return res

'''
w1 = 'DAMP'
w2 = 'LIKE'
d = {'DAMP', 'LAMP', 'LIMP', 'LIME', 'LIKE', 'DAME', 'LAME', }
word_transformer_naive(w1,w2,d)'''

from collections import deque
# Time Complexity: O(N^2 * M ), Space Complexity: O(N*M) where N=length of word and M=# words in dict
def word_transformer(word1, word2, dictionary):
    
    # check if two words are only one edit away
    def checkOneEdit(w1, w2):
        diff_count=0
        for i in range(len(w1)):
            if w1[i]!=w2[i]: diff_count+=1
            if diff_count>1: return False
        return True
    
    queue = deque()
    queue.append([word1])
    res = []
    visited = set()
    visited.add(word1)
    while queue:
        currWords = set()
        for i in range(len(queue)):
            path = queue.popleft()
            lastword = path[-1]
            if lastword == word2: res.append(path)
            for word in dictionary:
                if checkOneEdit(word, lastword) and word not in visited:
                    queue.append(path+[word])
                    currWords.add(word)

        visited.update(currWords)
    
    print(res)
    return res
'''
w1 = 'DAMP'
w2 = 'LIKE'
d = {'DAMP', 'LAMP', 'LIMP', 'LIME', 'LIKE', 'DAME', 'LAME'}
word_transformer(w1,w2,d)'''

#---------------------------------------------------------------------------------------------------------
# 17.23 Max Square Matrix: Imagine you have a square matrix, where each cell (pixel) is either black or
# white. Design an algorithm to find the maximum subsquare such that all four borders are filled with
# black pixels.

# B B B B B  -> 4*4 = 16
# B B W W B
# B B W W B
# W B B B B
# W W W B B

#   1 2 3
# 1 1 2 3
# 2 2 4 6
# 3 3 6 9

# Time Complexity: O(N^2), Space Complexity: O(N^2) where N = dimension of square
def max_square_matrix(matrix):
    
    counts = [[(0,0) for _ in range(len(matrix))] for _ in range(len(matrix))]
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if matrix[i][j] == 'B': counts[i][j] = (counts[i-1][j][0]+1, counts[i][j-1][1]+1)

    max_square = (0,None)
    for i in range(len(matrix)-1,-1,-1):
        for j in range(len(matrix)-1,-1,-1):
            min_span = min(counts[i][j][0],counts[i][j][1])
            up, left = i-min_span+1, j-min_span+1

            if matrix[i][j] == 'B' and counts[up][j][1] >= min_span and counts[i][left][0] >= min_span and min_span**2 > max_square[0]:
                max_square = (min_span**2, ((i-min_span+1, j-min_span+1), (i,j)))

    print(max_square)
    return max_square

'''
m = [['B', 'B', 'B', 'B', 'B'],['B', 'B', 'W', 'W', 'B'],['B', 'B', 'W', 'W', 'B'],['W', 'B', 'B', 'B', 'B'],['W', 'W', 'W', 'B', 'B']]
max_square_matrix(m)'''

#---------------------------------------------------------------------------------------------------------
# 17.24 Max Submatrix: Given an NxN matrix of positive and negative integers, write code to find the
# submatrix with the largest possible sum.
# 
#  1 -5  7  7 -4  -> 33
#  2  7  7  7  1  
# -2 -3 -1 -1 -2  
# -2 -2  1  1 -2  
# -1  3 -2  2 -2 

# Brute force -> N^2 possible subrows, N^2 possible subcols, N^4 time for submatrices, N^2 to compute sum for each,
# therefore, O(N^6) time for brute force

# Time Complexity: O(N^4), Space Complexity: O(N^2)
def max_submatrix(matrix):

    '''LOOKS SIMPLE BUT PAY CLOSE ATTENTION !!!'''
    
    sub_sum = [[0 for _ in range(len(matrix))] for _ in range(len(matrix))]
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            up = sub_sum[i-1][j] if i>0 else 0 # actually dont even need these if statement, will wrap around
            left = sub_sum[i][j-1] if j>0 else 0 # actually dont even need these if statement, will wrap around
            union = sub_sum[i-1][j-1] if j>0 and i>0 else 0 # actually dont even need these if statement, will wrap around
            sub_sum[i][j] = matrix[i][j] + up + left - union
    
    max_sub = (0, None)
    for i in range(len(matrix)):
        for i_c in range(i,len(matrix)):
            for j in range(len(matrix)):
                for j_c in range(j,len(matrix)):
                    # carefull attention to the indexing !!!
                    sum_t = sub_sum[i-1][j_c] if i>0 else 0
                    sum_l = sub_sum[i_c][j-1] if j>0 else 0
                    sum_tl = sub_sum[i-1][j-1] if i>0 and j>0 else 0
                    sum_whole = sub_sum[i_c][j_c] - sum_t - sum_l + sum_tl
                    
                    if sum_whole > max_sub[0]: max_sub = (sum_whole, ((i,j), (i_c,j_c)))
                    
    print(max_sub)
    return max_sub

'''
m = [[1, -5, 7, 7, -4],[2, 7, 7, 7, 1],[-2, -3, -1, -1, -2],[-2, -2, 1, 1, -2],[-1, 3, -2, 2, -2]]
max_submatrix(m)'''

#--------------------------------------------------------------------------------------------------------------
# 17.25 Word Rectangle: Given a list of millions of words, design an algorithm to create the largest possible
# rectangle of letters such that every row forms a word (reading left to right) and every column forms
# a word (reading top to bottom). The words need not be chosen consecutively from the list, but all
# rows must be the same length and all columns must be the same height.

# p h a t
# h o p e
# o p e n

# note: possible largest rectangle is longest word in list, also: can use bag of words (trie)

# Time Complexity: ??, Space Complexity: O(N) ??
def word_rectangle(words):

    '''TRICKY AS A MF !!!'''

    def build_trie(length):
        tr = {}
        for word in dic[length]:
            t = tr
            for w in word:
                if w not in t: t[w] = {}
                t = t[w]
            t[-1] = -1
        return tr

    def check_valid(width, prefix):
        if width not in trie_dict: return False
        if len(prefix) == 1: return True
        tr = trie_dict[width]
        for c in prefix:
            if c in tr: tr = tr[c]
            else: return False

        return True

    def create_rec(rec, visited, l, w, w_list):

        if len(rec) == w: return rec
        if len(visited) == len(w_list): return None
        
        for word in w_list:
            if word not in visited:
                rec.append(list(word))
                visited.add(word)
                for j in range(len(word)):
                    prefix = ''
                    for i in range(len(rec)):
                        prefix += rec[i][j]
                        if not check_valid(w,prefix): return None
        
        return create_rec(rec, visited, l, w, w_list)
        
    dic = {}
    max_l = 0
    for w in words:
        if len(w) in dic: dic[len(w)].append(w)
        else: dic[len(w)] = [w]
        max_l = max(max_l, len(w))
    
    trie_dict = {k:build_trie(k) for k in dic}
    
    for i in range(max_l,0,-1):
        for j in range(i,0,-1):
            if i in dic and j in dic:
                r = create_rec([],set(), i, j, dic[i])
                if r:
                    print(len(r)*len(r[0]),r)
                    return (len(r)*len(r[0]),r)
    
    return None

'''
#w = ['tester', 'phat', 'hope', 'open', 'pho', 'hop', 'ape', 'ten']
w = ['spears', 'planet', 'easily', 'animal', 'relate', 'styles']
word_rectangle(w)'''

#--------------------------------------------------------------------------------------------------------------
# 17.26 Sparse Similarity: The similarity of two documents (each with distinct words) is defined to be the
# size of the intersection divided by the size of the union. For example, if the documents consist of
# integers, the similarity of {1, 5, 3} and { 1, 7, 2, 3} is 0. 4, because the intersection has size
# 2 and the union has size 5.
# We have a long list of documents (with distinct values and each with an associated ID) where the
# similarity is believed to be "sparse:'That is, any two arbitrarily selected documents are very likely to
# have similarity O. Design an algorithm that returns a list of pairs of document IDs and the associated
# similarity.
# Print only the pairs with similarity greater than 0. Empty documents should not be printed at all. For
# simplicity, you may assume each document is represented as an array of distinct integers.
# EXAMPLE
#
# Input:
# 13: {14, 15, 100, 9, 3}
# 16: {32, 1, 9, 3, 5}
# 19: {15, 29, 2, 6, 8, 7}
# 24: {7, 10}
#
# Output:
# ID1, ID2: SIMILARITY
# 13,  19:  0.1
# 13,  16:  0.25
# 19,  24:  0.14285714285714285


# Time Complexity: O(N^2 * W^2), Space Complexity: O(1) where N=# of documents and W=# of words in longest doc
def sparse_similarity_naive(documents):

    #O(W^2)
    def compute(d_a, d_b):
        intersection = 0
        for i in range(len(d_a[1])):
            for j in range(len(d_b[1])):
                if d_a[1][i]==d_b[1][j]: intersection += 1
        if intersection:
            union = len(d_a[1]) + len(d_b[1]) - intersection
            print(f"{d_a[0]}, {d_b[0]}: {intersection/union}")
    
    #O(N^2)
    for i in range(len(documents)):
        doc_a = documents[i]
        for j in range(i+1,len(documents)):
            doc_b = documents[j]
            compute(doc_a, doc_b)


            
'''
d = [[13, [14, 15, 100, 9, 3]], [16, [32, 1, 9, 3, 5]], [19, [15, 29, 2, 6, 8, 7]], [24, [7, 10]]]
sparse_similarity_naive(d)'''

# Time Complexity: O(N^2 * W + N * WlogW), Space Complexity: O(1) where N=# of documents and W=# of words in longest doc
def sparse_similarity(documents):

    #O(W)
    def compute(d_a, d_b):
        intersection = i = j = 0
        while i<len(d_a[1])-1 and i<len(d_b[1])-1:
            if d_a[1][i] == d_b[1][j]: 
                intersection += 1
                i+=1
                j+=1
            if d_a[1][i] > d_b[1][j]: j+=1
            else: i+=1
        while i<len(d_a[1])-1: 
            if d_a[1][i] == d_b[1][j]: intersection += 1
            i+=1
        while j<len(d_b[1])-1: 
            if d_a[1][i] == d_b[1][j]: intersection += 1
            j+=1

        if intersection:
            union = len(d_a[1]) + len(d_b[1]) - intersection
            print(f"{d_a[0]}, {d_b[0]}: {intersection/union}")

    #O(N * WlogW)
    for i in range(len(documents)): documents[i][1].sort()
    
    #O(N^2)
    for i in range(len(documents)):
        doc_a = documents[i]
        for j in range(i+1,len(documents)):
            doc_b = documents[j]
            compute(doc_a, doc_b)

'''
d = [[13, [14, 15, 100, 9, 3]], [16, [32, 1, 9, 3, 5]], [19, [15, 29, 2, 6, 8, 7]], [24, [7, 10]]]
sparse_similarity(d)'''


# Time Complexity: O(N*W + P*W), Space Complexity: O(N*W), where N=# of documents, W=# of words in longest doc, P= pairs of docs with intersection
def sparse_similarity_optimized(documents):

    # O(N*W)
    word_map = {}
    doc_length = {}
    for doc in documents:
        for w in doc[1]:
            if w not in word_map: word_map[w] = [doc[0]]
            else: word_map[w].append(doc[0])
        doc_length[doc[0]] = len(doc[1])

    # O(P*W)
    pairs_of_docs = {} # intersection map
    for w in word_map:
        for i in range(len(word_map[w])):
            for j in range(i+1,len(word_map[w])):
                if (word_map[w][i],word_map[w][j]) not in pairs_of_docs: pairs_of_docs[(word_map[w][i],word_map[w][j])] = 1
                else: pairs_of_docs[(word_map[w][i],word_map[w][j])] += 1

   
    # O(P)
    for pair in pairs_of_docs:
        intersection = pairs_of_docs[pair]
        union = doc_length[pair[0]] + doc_length[pair[1]] - intersection
        print(f"{pair[0]}, {pair[1]}: {intersection/union}")
       

'''
d = [[13, [14, 15, 100, 9, 3]], [16, [32, 1, 9, 3, 5]], [19, [15, 29, 2, 6, 8, 7]], [24, [7, 10]]]
sparse_similarity_optimized(d)'''