#---------------------------------------------------------------------------------------------------------
# 11.1 Mistake: Find the mistake(s) in the following code:
# unsigned int i;
# for (i = 100; i >= 0; --i)
# 	printf("%d\n", i);

# Ans: Unsigned int is always greater than or equal to zero, so infinite for loop. And use %u instead of %d, since unsigned int.
# for (i = 100; i > 0; --i)
# 	printf("%u\n", i);


#---------------------------------------------------------------------------------------------------------
# 11.2 Random Crashes: You are given the source to an application which crashes when it is run. After
# running it ten times in a debugger, you find it never crashes in the same place. The application is
# single threaded, and uses only the C standard library. What programming errors could be causing
# this crash? H ow would you test each one?


#---------------------------------------------------------------------------------------------------------
# 11.3 ChessTest:We have the following method used in a chess game: boolean canMoveTo(int x,
# int y). This method is part of the Piece class and returns whether or not the piece can move to
# position (x, y). Explain how you would test this method.


#---------------------------------------------------------------------------------------------------------
# 11.4 No Test Tools: How would you load test a webpage without using any test tools?


#---------------------------------------------------------------------------------------------------------
# 11.5 Test a Pen: How would you test a pen?


#---------------------------------------------------------------------------------------------------------
# 11.6 Test an ATM: How would you test an ATM in a distributed banking system?