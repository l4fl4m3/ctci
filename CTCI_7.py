#---------------------------------------------------------------------------------------------------------
# 7.1 Deck of Cards: Design the data structures for a generic deck of cards. Explain how you would
# subclass the data structures to implement blackjack.

class Card:
	# ranks and suits to be used
	RANKS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)
	SUITS = ('C', 'D', 'H', 'S')

	def __init__(self,rank,suit):
		self.rank = rank
		self.suit = suit

	# when str called on Card, we print face cards accordingly
	def __str__ (self):
		if (self.rank == 1):
			rank = 'A'
		elif (self.rank == 13):
			rank = 'K'
		elif (self.rank == 12):
			rank = 'Q'
		elif (self.rank == 11):
			rank = 'J'
		else:
			rank = str (self.rank)
		return rank + self.suit

class CardDeck:
	def __init__(self):
		self.cards = []
		self.makeDeck()

	#Make New Deck
	def makeDeck(self):
		for rank in Card.RANKS:
			for suit in Card.SUITS:
				self.cards.append(Card(rank,suit))
	#Shuffle
	def shuffle(self):
		for i in range(len(self.cards)):
			s = random.randint(0,i)
			self.cards[s],self.cards[i] = self.cards[i],self.cards[s]

		# or could juse use this
		#random.shuffle(self.cards)
	
	#Get Top Card
	def get_card(self):
		return self.cards.pop()

class Player:
	'''Player class'''
	def __init__(self,cards):
		self.cards = cards
		self.bust = False

	# get score of player
	def get_score(self):
		score = 0
		for c in self.cards:
			if c.rank >9: score+=10
			elif c.rank==1: score+=11
			else: score+=c.rank
		for c in self.cards:
			if score<=21: return score
			if c.rank==1: score-=10

		return score
	
	# return comma seperated string of cards the player holds
	def get_cards(self):
		cards = []
		for c in self.cards:
			cards.append(str(c))
		return ','.join(cards)

	# add to the players cards/hand
	def hit(self,card):
		self.cards.append(card)
	
	# set bust flag
	def make_bust(self):
		self.bust = True

class Dealer(Player):
	''' Same as Player, can add other functions as needed'''
	def __init__(self,cards):
		super().__init__(cards)

class BlackJack(CardDeck):
	def __init__(self,n_players):
		# create a deck, shuffle it
		self.deck = CardDeck()
		self.deck.shuffle()
		self.players = []
		self.cards_in_hand = 2

		#Create Players with their initial hands
		for i in range(n_players):
			hand = []
			for j in range(self.cards_in_hand): hand.append(self.deck.get_card())
			self.players.append(Player(hand))
		
		#Create Dealers with initial hand
		dealer_hand=[]
		for i in range(self.cards_in_hand): dealer_hand.append(self.deck.get_card())
		self.dealer = Dealer(dealer_hand)

	def play(self):
		# print player hands
		for i in range(len(self.players)):
			print(f"Player: {i}, Cards: {self.players[i].get_cards()}, Score: {self.players[i].get_score()}")
		# print dealer first card
		print(f"Dealer, First Card: {str(self.dealer.cards[0])}")

		# Let each player play
		for i in range(len(self.players)):
			player = self.players[i]
			hit = True
			while hit and player.get_score()<21:
				resp = input(f"Player {i}: Want to hit? (y/n")
				hit = resp=='y'
				if hit: 
					player.hit(self.deck.get_card())
					print(f"Player: {i}, Cards: {self.players[i].get_cards()}, Score: {self.players[i].get_score()}")
				
			if player.get_score()>21: player.make_bust()

		# Play Dealer
		while self.dealer.get_score()<17:
			self.dealer.hit(self.deck.get_card())

		# check if dealer is busted
		if self.dealer.get_score()>21: self.dealer.make_bust()

		# print dealer score
		print(f"Dealer, Cards: {self.dealer.get_cards()}, Score: {self.dealer.get_score()}")
		

		# if dealer bust, check who, if any, players win
		if self.dealer.bust:
			for i in range(len(self.players)):
				if not self.players[i].bust: print(f"Player {i} wins!")
				else: print(f"Player {i} loses.")

		# otherwise check who has higher score than dealer
		else: 
			for i in range(len(self.players)):
				if not self.players[i].bust:
					if self.players[i].get_score()>self.dealer.get_score(): print(f"Player {i} wins!")
					elif self.players[i].get_score()==self.dealer.get_score(): print(f"Player {i} ties.")
					else: print(f"Player {i} loses.")

				else: print(f"Player {i} loses.")


#b = BlackJack(2)
#b.play()


#---------------------------------------------------------------------------------------------------------
# 7.2 Call Center: Imagine you have a call center with three levels of employees: respondent, manager,
# and director. An incoming telephone call must be first allocated to a respondent who is free. If the
# respondent can't handle the call, he or she must escalate the call to a manager. If the manager is not
# free or not able to handle it, then the call should be escalated to a director. Design the classes and
# data structures for this problem. Implement a method dispatchCall() which assigns a call to
# the first available employee.

class Employee:
	def __init__(self,rank=None):
		self.rank = rank
		self.call = None

	def take_call(self,call):
		self.call = call

	def escalate(self,call):
		if self.rank == 'R': self.call.rank = "M"
		if self.rank == 'M': self.call.rank = "D"
		call.dispatchCall(call)
	
	def getRank(self):
		return self.rank

	def isFree(self):
		return self.call


class Respondent(Employee):
	def __init__(self,emp_number):
		super().__init__(emp_number, "R")

class Manager(Employee):
	def __init__(self,emp_number):
		super().__init__(emp_number, "M")

class Director(Employee):
	def __init__(self,emp_number):
		super().__init__(emp_number, "D")

class Call:
	def __init__(self,caller):
		self.caller = caller
		self.rank = "R"
		self.handler = None
	
	def setHandler(self,employee):
		self.handler = employee

	def setRank(self,rank):
		self.rank = rank

	def getRank(self):
		return self.rank
	

class CallCenter:

	RANKS = ('R','M','D')

	def __init__(self,num_r=10,num_m=2,num_d=2):
		self.respondents = []
		self.managers = []
		self.directors = []
		self.employees = 0
		self.makeEmployees(num_r,'R')
		self.makeEmployees(num_m,'M')
		self.makeEmployees(num_d,'D')
		self.queue = []

	def makeEmployees(self,num,rank):
		if not rank: return None
		emp = []
		for i in range(num): 
			if rank == 'R':
				emp.append(Respondent(self.employees))
			elif rank == 'M':
				emp.append(Manager(self.employees))
			elif rank == 'D':
				emp.append(Manager(self.employees))
			self.employees+=1
		
		if rank == 'R': self.respondents.append(emp)
		elif rank == 'M': self.managers.append(emp)
		elif rank == 'D': self.directors.append(emp)
	
	def callHandler(self,call):
		if call.rank == 'R': handler = self.respondents
		if call.rank == 'M': handler = self.managers
		if call.rank == 'D': handler = self.directors
		for i in range(len(handler)):
			if handler[i].isFree(): return handler[i]

		# for manager ranked calls, try to see if director is free
		if call.rank == 'M':
			handler = self.directors
			for i in range(len(handler)):
				if handler[i].isFree():
					call.rank = 'D' 
					return handler[i]
		
		return None

	def dispatchCall(self,call):
		c = call
		if self.queue:
			self.queue.append(c)
			c = self.queue.pop(0)
		handler = self.callHandler(c)
		if not handler:
			print("Call in Queue")
			self.queue.append(c)
			return

		handler.take_call(c)
		call.setHandler = handler


#---------------------------------------------------------------------------------------------------------
# 7.3 Jukebox: Design a musical jukebox using object-oriented principles.

class JukeBox:
	def __init__(self):
		self.cdplayer = None
		self.cds = []
	def play(self):
		self.cdplayer.playSong()
	def selectCD(self,cd):
		self.cdplayer.selectCD(cd)
	def selectSong(self,song):
		self.cdplayer.playSong(song)
	def addCDs(self,cds):
		for c in cds: self.cds.append(c)
	def removeCDs(self,cds):
		for c in cds: self.cds.remove(c)

class CDPlayer:
	def __init__(self,cd,playlist):
		self.song = cd
		self.cd = playlist
		self.playlist = None

	def playSong(self,song):
		#send to music/audio buffer, and play
		pass
	def setPlaylist(self, playlist):
		self.playlist = playlist
	def getPlaylist(self, playlist):
		return self.playlist
	def setCD(self,cd):
		self.cd = cd
	def getCD(self):
		return self.cd
	def setPlaylist(self, playlist):
		self.playlist = playlist
	
class CD:
	def __init__(self,name,songs):
		self.name = name
		self.songs = songs
	
class Song:
	def __init__(self,name,artist):
		self.name = name
		self.artist = artist

class Playlist:
	def __init__(self,name):
		self.name = name
		self.songs = []

	def addSong(self,song):
		self.songs.append(song)

	def removeSong(self,song):
		self.songs.remove(song)

	def shuffle(self):
		self.songs.random.shuffle()

	def getNextSong(self):
		return self.songs[0]

# Cleaner way if no other info is givens
class JukeBox:
	def __init__(self,songs):
		self.songs={}
		for s in songs:
			self.songs[s] = s.name
		self.current_song = None
	
	def playSong(self, name):
		if self.current_song: self.stopSong()
		self.current_song = name
		self.current_song.play() 

	def stopSong(self):
		if self.current_song: self.current_song.stop()

class Song:
	def __init__(self,name):
		self.name = name
		self.playing = False

	def play(self):
		self.playing = True

	def stop(self):
		self.playing = False


#---------------------------------------------------------------------------------------------------------
# 7.4 Parking Lot: Design a parking lot using object-oriented principles.

# Assume lot has multiple levels. Can park motorcycles, cars and buses. Has motorcycle, and regular. A motorcycle
# can park in any spot. A car can park in a regular spot. Bus can park in five regular spots that are consecutive and within
# the same row.

# The answer in the book is what this one follows. However the answer gets convoluted. There are many random things/assumptions
# that are introduced. I think its best to jsut create a Vehicle base class, then a few types of vehicles. Then create a Parking Lot    
# wrapper class, and a Level and Parking Spot class, that should suffice. All the other stuff becomes too much needlessly.

VEHICLE_TYPES = ('M', 'C', 'B')
class Vehicle:
	def __init__(self, vehicle_type):
		self.vehicle_type = vehicle_type
		#self.spots_required = spots_required
		#self.parking_spots = set()
	'''
	def parkInSpot(self, spot):
		self.parking_spots.add(spot)

	def unParkFromSpot(self, spot):
		self.parking_spots.remove(spot)
	'''

class Motorcycle(Vehicle):
	def __init__(self):
		super().__init__('M')

class Car(Vehicle):
	def __init__(self):
		super().__init__('C')

class Bus(Vehicle):
	def __init__(self):
		super().__init__('B')

class ParkingLot:

	def __init__(self, num_levels=3):
		self.levels = []
		self.num_levels = num_levels
		self.makeLevels()

	def makeLevels(self):
		for i in range(self.num_levels):
			self.levels.append(Level(i,100,10))
	
	def parkVehicle(self,vehicle):
		for level in self.levels:
			spot = level.findAvailableSpot(vehicle)
			if spot:
				level.parkAtSpot(spot,vehicle)
				return True
		return False

class Level:
	def __init__(self,floor,num_spots=100,spots_per_row=10):
		self.floor = floor
		self.parking_spots = []
		self.available_spots = 0
		self.spots_per_row = spots_per_row
		self.makeSpots(num_spots)

	def makeSpots(self, num_spots):
		num_rows = num_spots//self.spots_per_row
		for i in range(num_rows):
			for _ in range(self.spots_per_row):
				self.parking_spots.append(ParkingSpot(self.floor, i, self.available_spots,'C'))
				self.available_spots+=1

	def parkAtSpot(self,spot,vehicle):
		spot.parkInSpot(vehicle)

	def findAvailableSpot(self, vehicle):
		if not self.available_spots: return None
		for spot in self.parking_spots: 
			if spot.vehicle_type == vehicle.vehicle_type: return spot
		return None

	def spotFreed(self):
		self.available_spots+=1

class ParkingSpot:
	def __init__(self, level, row, spot_number, vehicle_type):
		self.level = level
		self.row = row
		self.spot_number = spot_number
		self.vehicle_type = vehicle_type
		self.vehicle = None

	def isAvailable(self):
		return self.vehicle

	def canFitVehicle(self, vehicle):
		return self.vehicle_type==vehicle.vehicle_type

	def parkInSpot(self, vehicle):
		self.vehicle = vehicle
		#self.vehicle.parkInSpot(self)

	def unParkFromSpot(self):
		if not self.vehicle: return None
		#self.vehicle.unParkFromSpot(self)
		self.vehicle = None
		self.level.spotFreed()

	def getRow(self):
		return self.row
	
	def getSpotNumber(self):
		return self.spotNumber

#---------------------------------------------------------------------------------------------------------
# 7.5 Online Book Reader: Design the data structures for an online book reader system.

class OnlineBookReader:
	def __init__(self):
		self.userManager = UserManager()
		self.library = Library()
		self.display = Display()
		self.current_book = None
		self.current_user = None

	def getLibrary(self):
		return self.library
	
	def getDisplay(self):
		return self.display
	
	def getCurrentBook(self):
		return self.current_book
	
	def getCurrentUser(self):
		return self.current_user

	def setCurrentBook(self,book):
		self.current_book = book
		self.display.displayBook(book)

	def setCurrentUser(self,user):
		self.current_user = user
		self.display.displayUser(user)
	


class UserManager:
	def __init__(self):
		self.users = {}

	def addUser(self,id, info):
		if id in self.users: return None
		new_user = User(id, info)
		self.users[id] = new_user
		return new_user

	def deleteUser(self,user):
		if user.id not in self.users: return False
		self.users.pop(user.id)
		return True
	
	def getUser(self,id):
		if id not in self.users: return None
		return self.users[id]
	
class User:
	def __init__(self,id,info):
		self.id = id
		self.info = info
	def getUserId(self):
		return self.id
	def setUserId(self,id):
		self.id = id
	def getInfo(self):
		return self.info
	def setInfo(self,info):
		self.info = info

class Display:
	def __init__(self):
		self.current_user= None
		self.current_book= None
		self.page_num = 0
	
	def displayUser(self, user):
		self.current_user = user.id
		self.reRender()
	
	def displayBook(self,book):
		self.current_book = book
		self.reRender()

	def nextPage(self):
		if self.page_num < self.current_book.getNumPages(): self.page_num+=1
		else: return None
		self.reRender()

	def previousPage(self):
		if self.page_num>0: self.page_num-=1
		else: return None
		self.reRender()

	def reRender(self):
		#display(self.current_book, self.current_user, self.page_num)
		# peripheral / buffers / drivers that handle 
		pass

class Library:
	def __init__(self):
		self.books = {}

	def addBook(self,id,info):
		if id in self.books: return None
		new_book = Book(id, info)
		self.books[id] = new_book
		return new_book

	def deleteBook(self,book):
		if book.id not in self.books: return False
		self.books.pop(book.id)
		return True
	
	def getBook(self,id):
		if id not in self.books: return None
		return self.books[id]

class Book:
	def __init__(self, id, info):
		self.id = id
		self.info = info

	def getBookId(self):
		return self.id
	def setUserId(self,id):
		self.id = id
	def getInfo(self):
		return self.info
	def setInfo(self,info):
		self.info = info
	def getNumPages(self):
		return self.info.getNumPages


#---------------------------------------------------------------------------------------------------------
# 7.6 Jigsaw: Implement an NxN jigsaw puzzle. Design the data structures and explain an algorithm to
# solve the puzzle. You can assume that you have a fitsWith method which, when passed two
# puzzle edges, returns true if the two edges belong together.

# This is not as convoluted as the official answer. Basically we create a puzzle, and for each piece, have a set 
# of the (max 4) edges/surrounding pieces that it fits with and is currently connected to. The piece class also 
# has a method that allows you to connect another piece to it, this just adds the other piece to the connectedset 
# within the current piece and vice versa. To solve the puzzle we iterate through each piece in the puzzle, and if 
# it is in the other fits set, we connect the two.

class JigsawPuzzle:
	def __init__(self, size):
		self.pieces = []
		self.size = size
		self.makePuzzle()

	def makePuzzle(self):
		pieces = [[Piece() for _ in range(self.size)] for _ in range(self.size)]
		for i in range(self.size):
			for j in range(self.size):
				if i: pieces[i][j].fitsWith(pieces[i-1][j])
				if j: pieces[i][j].fitsWith(pieces[i][j-1])
				self.pieces.append(pieces[i][j])
	
	def isSolved(self):
		for piece in self.pieces:
			if piece.connected != piece.fits: return False
		return True

	def solvePuzzle(self):
		for a in self.pieces:
			for b in self.pieces:
				if b in a.fits: a.connect(b)

class Piece:
	def __init__(self):
		self.fits = set()
		self.connected = set()

	def fitsWith(self,piece):
		self.fits.add(piece)
		piece.fits.add(self)

	def connect(self,piece):
		self.connected.add(piece)
		piece.connected.add(self)

#---------------------------------------------------------------------------------------------------------
# 7.7 Chat Server: Explain how you would design a chat server. In particular, provide details about the
# various backend components, classes, and methods. What would be the hardest problems to solve?

# Could be done cleaner with sets, but oh well

class ChatServer:

	def __init__(self):
		self.chats = {}
		self.chat_count = 0

	def createChat(self, info):
		chat = Chat(self.chat_count, info)
		self.chats[chat.id] = chat
		self.chat_count+=1
		return chat

	def deleteChat(self,chatId):
		if not chatId in self.chats: return False
		for user in self.chats[chatId].users: user.leaveChat(chatId)
		self.chats.pop(chatId)
		return True

class Chat:

	def __init__(self, id, info):
		self.id = id
		self.info = info
		self.users = {}
		self.messages = []

	def addUser(self,user):
		self.users[user.id] = user

	def removeUser(self,userId):
		if not userId in self.users: return False
		self.users.pop(userId.id)
		return True
	
	def addMessage(self,userId, message):
		self.messages.append((userId,message))

class User:

	def __init__(self,id,info):
		self.id = id
		self.info = info
		self.chats = {}
		self.sent_messages = []
		self.current_chat = None
	
	def getId(self):
		return self.id
	def setID(self,id):
		self.id = id
	def getChats(self):
		return self.chats

	def createChat(self,info):
		chat = ChatServer.createChat(info)
		chat.addUser(self)
		self.chats[chat.id] = chat
		self.current_chat = chat

	def leaveChat(self,chatId):
		self.chats[chatId].removeUser(self.id)
		self.chats.pop(chatId)

	def setCurrentChat(self, chat):
		self.current_chat = chat

	def sendMessage(self, message):
		self.current_chat.addMessage((self.id,message))
		self.sent_messages = ((self.current_chat.id,message))
	

#---------------------------------------------------------------------------------------------------------
# 7.8 Othello: Othello is played as follows: Each Othello piece is white on one side and black on the other.
# When a piece is surrounded by its opponents on both the left and right sides, or both the top and
# bottom, it is said to be captured and its color is flipped. On your turn, you must capture at least one
# of your opponent's pieces. The game ends when either user has no more valid moves. The win is
# assigned to the person with the most pieces. Implement the object-oriented design for Othello.

PIECE_COLOR = ('B', 'W')

class Othello:

	def __init__(self,rows,cols):
		self.rows = rows
		self.cols = cols
		self.board = Board(rows,cols)
		self.players = [Player(1,'B',self), Player(2,'W',self)]

	def getBoard(self):
		return self.board

	def play(self):
		"""Logic for game play in here"""
		pass


class Board:

	def __init__(self,rows,cols):
		self.board = [[None for _ in range(cols)] for _ in range(rows)]
		self.score_white = 0
		self.score_black = 0

	def setPiece(self,row,col,piece):
		self.board[row][col] = piece
		self.flipPieces(row,col,piece.color,direction)

	def flipPieces(self, row, color, direction):
		# add logic for flipping consecutive pieces and get score_delta
		self.updateScore(color, score_delta)

	def getScore(self,color):
		if color == 'B': return self.score_black
		return self.score_white
	
	def updateScore(self,color, score_delta):
		if color == 'B': 
			self.score_black += score_delta
			self.score_white -= score_delta
		else: 
			self.score_black -= score_delta
			self.score_white += score_delta

class Piece:

	def __init__(self,color):
		self.color = color
	def getColor(self):
		return self.color
	def setColor(self, color):
		self.color = color
	def toggleColor(self):
		self.setColor('B') if self.color == 'W' else self.setColor('W')

class Player:

	def __init__(self,id,color):
		self.id = id
		self.color = color

	def getId(self):
		return self.id
	def getColor(self):
		return self.color


#---------------------------------------------------------------------------------------------------------
# 7.9 Circular Array: Implement a CircularArray class that supports an array-like data structure which
# can be efficiently rotated. If possible, the class should use a generic type (also called a template), and
# should support iteration via the standard for (Obj o : circularArray) notation.


#---------------------------------------------------------------------------------------------------------
# 7.10 Minesweeper: Design and implement a text-based Minesweeper game. Minesweeper is the classic
# single-player computer game where an NxN grid has B mines (or bombs) hidden across the grid. The
# remaining cells are either blank or have a number behind them. The numbers reflect the number of
# bombs in the surrounding eight cells. The user then uncovers a cell. If it is a bomb, the player loses.
# If it is a number, the number is exposed. If it is a blank cell, this cell and all adjacent blank cells (up to
# and including the surrounding numeric cells) are exposed. The player wins when all non-bomb cells
# are exposed. The player can also flag certain places as potential bombs. This doesn't affect game
# play, other than to block the user from accidentally clicking a cell that is thought to have a bomb.
# (Tip for the reader: if you're not familiar with this game, please play a few rounds on line first.)

#---------------------------------------------------------------------------------------------------------
# 7.11 File System: Explain the data structures and algorithms that you would use to design an in-memory
# file system. Illustrate with an example in code where possible.

#---------------------------------------------------------------------------------------------------------
# 7.12 Hash Table: Design and implement a hash table which uses chaining (linked lists) to handle collisions.