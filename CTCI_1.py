#------------------------------------------------------------------------------
# 1.1 Implement an alforithm to determine if a string has all unique characters. What if you cannot use additional data structures

# Naive solution
from calendar import c
from inspect import _ParameterKind, stack
from re import I, X
from tkinter import N
from matplotlib.pyplot import box
from numpy import char
from psutil import OSX


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
