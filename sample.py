"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
PY GAME

https://pygame-learning-environment.readthedocs.io/en/latest/user/games/monsterkong.html

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import os
from os.path import exists

import tensorflow as tf

import ple
from ple import PLE
from ple.games.monsterkong import MonsterKong as MonsterKong_Game
from ple.games import base

###
from ple.games.monsterkong.person import Person as Person_Game
from ple.games.monsterkong.player import Player as Player_Game
from ple.games.monsterkong.monsterPerson import MonsterPerson as MonsterPerson_Game
###

from pygame.constants import K_a, K_s, K_d, K_w, K_h, K_SPACE

import matplotlib.pyplot as plt

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
None
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
print(physical_devices)
print(config)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Variables
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
actions = { "none_1": K_h, "left": K_a, "down": K_s, "right": K_d, "up": K_w, "action": K_SPACE }
nb_frames = 100000000000

global lives
global reward
global steps
global gamescores
	
steps = 0
lives = 0
reward = 0
gamescores = 0

################ Mixed of data input  ###############
global DATA
DATA = tf.zeros([1, 1, 1, 30], dtype=tf.float32)
global LABEL
LABEL = tf.zeros([1, 1, 1, 1], dtype=tf.float32)

for i in range(15):
	DATA_row = tf.constant([ 9999, -9999, 9999, -9999, 9999, -9999, 9999, -9999, 9999,
				-9999, 9999, -9999, 9999, -9999, 9999, -9999, 9999, -9999, 9999, -9999, 9999, -9999, 9999, -9999, 9999, -9999, 9999, -9999, 9999, -9999 ], shape=(1, 1, 1, 30), dtype=tf.float32)		
	DATA = tf.experimental.numpy.vstack([DATA, DATA_row])
	LABEL = tf.experimental.numpy.vstack([LABEL, tf.constant(0, shape=(1, 1, 1, 1))])
	
for i in range(15):
	DATA_row = tf.constant([ -9999, 9999, -9999, 9999, -9999, 9999, -9999, 9999, -9999,
				9999, -9999, 9999, -9999, 9999, -9999, 9999, -9999, 9999, -9999, 9999, -9999, 9999, -9999, 9999, -9999, 9999, -9999, 9999, -9999, 9999 ], shape=(1, 1, 1, 30), dtype=tf.float32)		
	DATA = tf.experimental.numpy.vstack([DATA, DATA_row])
	LABEL = tf.experimental.numpy.vstack([LABEL, tf.constant(9, shape=(1, 1, 1, 1))])	
	
DATA = DATA[-30:,:,:,:]
LABEL = LABEL[-30:,:,:,:]
####################################################

momentum = 0.1
learning_rate = 0.00001
batch_size=10

checkpoint_path = "F:\\models\\checkpoint\\" + os.path.basename(__file__).split('.')[0] + "\\TF_DataSets_01.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)

if not exists(checkpoint_dir) : 
	os.mkdir(checkpoint_dir)
	print("Create directory: " + checkpoint_dir)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Class / Functions
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# take memeber element for sort
def elementPhase(elem):

    return elem[1]
	
def elementListCreate( list_item, bRelativePlayer=False ):

	player = read_current_state("player")
	list_temp = [ ( 6, int( pow( pow( x - player[0][0], 2 ) + pow( y - player[0][1], 2 ), 0.5 ) ), x, y ) for ( x, y ) in list_item if y <= player[0][1] ]
	
	if len( list_temp ) > 0 :
		pass
	else :
		list_temp.append( [ 5, -999, -999, -999 ] )
		
	list_temp.sort(key=elementPhase)

	return list_temp
	
def	read_current_state( string_gamestate ):
		
	if string_gamestate in ['score']:
		return game_console.getScore()
		
	elif string_gamestate in ['game_over']:	# False
		return game_console.game_over()
		
	elif string_gamestate in ['fireballGroup']:
		temp = []
		for fireball in game_console.newGame.fireballGroup :
			temp.append( fireball.getPosition() )	
		return temp
		
	elif string_gamestate in ['coinGroup']:
		temp = []
		for coin in game_console.newGame.coinGroup :
			temp.append( coin.getPosition() )	
		return temp
		
	elif string_gamestate in ['player']:
		Player_Game = game_console.newGame.Players
		temp = []
		for player in game_console.newGame.Players :
			temp.append( player.getPosition() )	
		return temp
		
	elif string_gamestate in ['monster']:
		MonsterPerson_Game = game_console.newGame.Enemies
		temp = []
		for Enemies in MonsterPerson_Game :
			temp.append( Enemies.getPosition() )	
		return temp
		
	elif string_gamestate in ['lives']:
		return game_console.newGame.lives
		
	elif string_gamestate in ['allies']:
		Allies_Game = game_console.newGame.allyGroup
		temp = []
		for allies in Allies_Game :
			temp.append( allies.getPosition() )	
		return temp
		
	elif string_gamestate in ['wall']:
		Wall_Game = game_console.newGame.wallGroup
		temp = []
		for wall in Wall_Game :
			temp.append( wall.getPosition() )	
		return temp
		
	elif string_gamestate in ['ladder']:
		Ladder_Game = game_console.newGame.ladderGroup
		temp = []
		for ladder in Ladder_Game :
			temp.append( ladder.getPosition() )	
		return temp
		
	else:
		return None
		
	return None

def predict_action( ):
	global DATA
	
	predictions = model.predict(tf.expand_dims(tf.squeeze(DATA), axis=1 ))
	score = tf.nn.softmax(predictions[0])

	return int(tf.math.argmax(score))

def random_action( ): 
	
	temp = tf.random.normal([1, 6], 0.1, 0.1, tf.float32)
	temp = tf.math.multiply(temp, tf.constant([ 0.0000099, 99999999, 0.0000099, 99999999, 0.0000099, 99999999 ], shape=(6, 1), dtype=tf.float32))
	temp = tf.nn.softmax(temp[0])
	action = int(tf.math.argmax(temp))

	return action

def update_DATA( action ):
	global lives
	global reward
	global steps
	global gamescores
	global DATA
	global LABEL
	
	steps = steps + 1
	
	gamescores = read_current_state("score")
	game_over = read_current_state("game_over")
	fireballGroup = read_current_state("fireballGroup")
	coinGroup = read_current_state("coinGroup")
	player = read_current_state("player")
	monsters = read_current_state("monster")
	lives = read_current_state("lives")
	allies = read_current_state("allies")
	ladders = read_current_state("ladder")
	wall = read_current_state("wall")
	
	list_player = [ ( 0, x, y ) for ( x, y ) in player if y >= player[0][1] ]
	list_monster = [ ( 1, x, y ) for ( x, y ) in monsters if y <= player[0][1] ]
	list_coin = [ ( 2, x, y ) for ( x, y ) in coinGroup if y <= player[0][1] ]
	list_fireball = [ ( 4, x, y ) for ( x, y ) in fireballGroup if y <= player[0][1] ]
		
	if len( list_coin ) > 2 :
		pass
	else :
		list_coin.append( [ 2, -999, -999 ] )
		
	if len( list_coin ) > 2 :
		pass
	else :
		list_coin.append( [ 2, -999, -999 ] )
		
	if len( list_fireball ) > 2 :
		pass
	else :
		list_fireball.append( [ 4, -999, -999 ] )
		
	if len( list_fireball ) > 2 :
		pass
	else :
		list_fireball.append( [ 4, -999, -999 ] )
		
	if len( list_monster ) > 2 :
		pass
	else :
		list_monster.append( [ 1, -999, -999 ] )
		
	if len( list_monster ) > 2 :
		pass
	else :
		list_monster.append( [ 1, -999, -999 ] )
		
		
	### determine alley allies ###
	list_alley = elementListCreate( allies, bRelativePlayer=True )
	### end ###	
	
	### determine ladder distance ###
	list_ladder = elementListCreate( ladders, bRelativePlayer=True )
	### end ###	
	
	### determine wall distance ###
	list_wall = elementListCreate( wall, bRelativePlayer=False )
	### end ###	
	
	steps + gamescores + ( 50 * reward )
	
	contrl = steps + gamescores + ( 50 * reward )
	contr2 = list_ladder[len(list_ladder) - 1][1]
	contr3 = 1
	
	coff_0 = list_player[0][1]
	coff_1 = list_player[0][2]
	coff_2 = list_alley[0][1]
	coff_3 = list_alley[0][3]
	coff_4 = list_coin[0][1]
	coff_5 = list_coin[0][2]
	coff_6 = list_coin[1][1]
	coff_7 = list_coin[1][2]
	coff_8 = list_fireball[0][1]
	coff_9 = list_fireball[0][2]
	coff_10 = list_fireball[1][1]
	coff_11 = list_fireball[1][2]
	coff_12 = list_ladder[0][1]
	coff_13 = list_ladder[1][1]
	coff_14 = list_ladder[2][1]
	coff_15 = list_ladder[3][1]
	coff_16 = list_monster[0][2]
	coff_17 = list_monster[0][2]
	coff_18 = list_monster[1][2]
	coff_19 = list_monster[1][2]
	
	coff_20 = list_wall[0][1]
	coff_21 = list_wall[1][1]
	coff_22 = list_wall[2][1]
	coff_23 = list_wall[3][1]
	coff_24 = list_wall[4][1]
	coff_25 = list_wall[5][1]
	coff_26 = 1

	action_name = [ x for ( x, y ) in actions.items() if y == action]
	
	print( "steps: " + str( steps ).zfill(6) + " action: " + str(action_name) + " coff_0: " + str(int(coff_0)).zfill(6) + " coff_1: " + str(int(coff_1)).zfill(6) + " coff_2: " 
			+ str(int(coff_2)).zfill(6) + " coff_3: " + str(int(coff_3)).zfill(6) + " coff_4: " + str(int(coff_4)).zfill(6) + " coff_5: " + str(int(coff_5)).zfill(6)
	)
	
	DATA_row = tf.constant([ contrl, contr2, contr3, coff_0, coff_1, coff_2, coff_3, coff_4, coff_5, coff_6, coff_7, coff_8, coff_9, coff_10, coff_11, coff_12, coff_13, coff_14, coff_15, coff_16, coff_17, coff_18, coff_19, 
			coff_20, coff_21, coff_22, coff_23, coff_24, coff_25, coff_26 ], shape=(1, 1, 1, 30), dtype=tf.float32)

	
	DATA = tf.experimental.numpy.vstack([DATA, DATA_row])
	DATA = DATA[-30:,:,:,:]
	
	LABEL = tf.experimental.numpy.vstack([LABEL, tf.constant(action, shape=(1, 1, 1, 1))])
	LABEL = LABEL[-30:,:,:,:]
	
	DATA = DATA[-30:,:,:,:]
	LABEL = LABEL[-30:,:,:,:]
	
	return DATA, LABEL, steps

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Environment
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
game_console = MonsterKong_Game()
p = PLE(game_console, fps=30, display_screen=True, reward_values={})
p.init()

obs = p.getScreenRGB()	# (500, 465, 3)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Callback
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
class custom_callback(tf.keras.callbacks.Callback):

	def __init__(self, patience=0):
		self.best_weights = None
		self.best = 999999999999999
		self.patience = patience
	
	def on_train_begin(self, logs={}):
		self.best = 999999999999999
		self.wait = 0
		self.stopped_epoch = 0

	def on_epoch_end(self, epoch, logs={}):
		if(logs['accuracy'] == None) : 
			pass
		
		if logs['loss'] < self.best :
			self.best = logs['loss']
			self.wait = 0
			self.best_weights = self.model.get_weights()
		else :
			self.wait += 1
			if self.wait >= self.patience:
				self.stopped_epoch = epoch
				self.model.stop_training = True
				print("Restoring model weights from the end of the best epoch.")
				self.model.set_weights(self.best_weights)
		
		# if logs['loss'] <= 0.2 and self.wait > self.patience :
		if self.wait > self.patience :
			self.model.stop_training = True

custom_callback = custom_callback(patience=8)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: DataSet
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
dataset = tf.data.Dataset.from_tensor_slices((DATA, LABEL))

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Model Initialize
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
input_shape = (1, 30)

model = tf.keras.models.Sequential([
	tf.keras.layers.InputLayer(input_shape=input_shape),
	
	tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True, return_state=False)),
	tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))

])
		
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(192))
model.add(tf.keras.layers.Dense(6))
model.summary()

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Optimizer
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
optimizer = tf.keras.optimizers.SGD(
    learning_rate=learning_rate,
    momentum=momentum,
    nesterov=False,
    name='SGD',
)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Loss Fn
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""								
lossfn = tf.keras.losses.MeanSquaredLogarithmicError(reduction=tf.keras.losses.Reduction.AUTO, name='mean_squared_logarithmic_error')

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Model Summary
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
model.compile(optimizer=optimizer, loss=lossfn, metrics=['accuracy'])

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: FileWriter
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
if exists(checkpoint_path) :
	model.load_weights(checkpoint_path)
	print("model load: " + checkpoint_path)
	input("Press Any Key!")

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Training
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
history = model.fit(dataset, epochs=1, callbacks=[custom_callback])
model.save_weights(checkpoint_path)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Tasks
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
for i in range(nb_frames):
	
	if p.game_over():
		p.reset_game()
		steps = 0
		lives = 0
		reward = 0
		gamescores = 0
		
	if ( steps == 1 ):
		print('start ... ')
		
	action = predict_action( )	
	action = list(actions.values())[action]

	reward = p.act(action)
	obs = p.getScreenRGB()
	
	gamescores = gamescores + reward
	DATA, LABEL, steps = update_DATA( action )
	

	if ( reward > 0 or steps % 15 == 0  ):
		dataset = tf.data.Dataset.from_tensor_slices((DATA, LABEL))
		history = model.fit(dataset, epochs=2, batch_size=batch_size, callbacks=[custom_callback])
		model.save_weights(checkpoint_path)
