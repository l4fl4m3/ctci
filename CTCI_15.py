#---------------------------------------------------------------------------------------------------------
# 15.1 Thread vs. Process: What's the difference between a thread and a process?

#---------------------------------------------------------------------------------------------------------
# 15.2 Context Switch: How would you measure the time spent in a context switch?

#---------------------------------------------------------------------------------------------------------
# 15.3 Dining Philosophers: In the famous dining philosophers problem, a bunch of philosophers are
# sitting around a circular table with one chopstick between each of them. A philosopher needs
# both chopsticks to eat, and always picks up the left chopstick before the right one. A deadlock
# could potentially occur if all the philosophers reached for the left chopstick at the same time. Using
# threads and locks, implement a simulation of the dining philosophers problem that prevents deadlocks.

#---------------------------------------------------------------------------------------------------------
# 15.4 Deadlock-Free Class: Design a class which provides a lock only if there are no possible deadlocks.

#---------------------------------------------------------------------------------------------------------
# 15.5 Call In Order: Suppose we have the following code:
# public class Foo {
# public Foo() { ... }
# public void first() { ... }
# public void second() { ... }
# public void third() { ... }
# }
# The same instance of F oo will be passed to three different threads. ThreadA will call first,
# threads will call second, and thread( will call third. Design a mechanism to ensure that
# first is called before second and second is called before third.

#---------------------------------------------------------------------------------------------------------
# 15.6 Synchronized Methods: You are given a class with synchronized method A and a normal method
# B. If you have two threads in one instance of a program, can they both execute A at the same time?
# Can they execute A and B at the same time?

#---------------------------------------------------------------------------------------------------------
# 15.7 FizzBuzz: In the classic problem FizzBuzz, you are told to print the numbers from 1 to n. However,
# when the number is divisible by 3, print "Fizz''. When it is divisible by 5, print "Buzz''. When it is divisible
# by 3 and 5, print "FizzBuzz''. In this problem, you are asked to do this in a multithreaded way.
# Implement a multithreaded version of FizzBuzz with four threads. One thread checks for divisibility
# of 3 and prints "Fizz''. Another thread is responsible for divisibility of 5 and prints"Buzz''. A third thread
# is responsible for divisibility of 3 and 5 and prints "FizzBuzz''. A fourth thread does the numbers.