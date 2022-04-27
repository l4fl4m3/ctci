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