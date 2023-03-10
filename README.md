# distance_attention_networks_input
For study distance input networks for machine learning problems, snake ladders, or tunnel shared distance. We can add the problem parameters into our AI training templates for study and solve the problem with the conditions you input.

In some problems, we cannot input all parameters and wait until AI can find the relationship of all distances, ```( X1 - X2 ) and ( Y1 - Y2 )``` but the AI can simply find their relationship from minimum or maximum of the distance input ```( item1, 5 ), ( item2, 10 ) and ( item3, 15 )```. The AI networks learn priority and the object's distance the problem is AI learns of the object's types and actions, not equations formula to find its distance or colors.

π§Έπ¬ It challenges conditions as most of the robot does or sort algorithms you can implement further, distance is something quick to determine the priority or it is the priority but it is not only distance and object types. Some of the traveling path problems are considered steps and relativity of each object as rules or ladder which is why we create a list of items as ```( types, relative distance, position X, position Y )```

## Objectives ##

π§Έπ¬ How do we reach the last room in the mazes from the current room is to find a tunnel and paths when the tunnel is a ladder and the path is distance.

```
contrl = steps + gamescores + ( 50 * reward )
contr2 = list_ladder[len(list_ladder) - 1][1]
contr3 = 1
```

## List and relative distance create ##

π§π¬ With simple problem type we know that ```distance``` value can be estimate from x and y values of player curser and target as ```pow( X1 - X2 ) + pow( Y1 - Y2 )```.

```
# take memeber element for sort
def elementPhase(elem):

    return elem[1]
	
def elementListCreate( list_item, bRelativePlayer=False ):

	player = read_current_state("player")
	list_temp = [ ( 6, int( pow( pow( x - player[0][0], 2 ) + 
                              pow( y - player[0][1], 2 ), 0.5 ) ), x, y ) for ( x, y ) in list_item if y <= player[0][1] ]
	
	if len( list_temp ) > 0 :
		pass
	else :
		list_temp.append( [ 5, -999, -999, -999 ] )
		
	list_temp.sort(key=elementPhase)

	return list_temp
  
ladders = read_current_state("ladder")
list_ladder = elementListCreate( ladders, bRelativePlayer=True )
```

## Result ##

MonsterKong player curser try all leftside are, right side and tunnel.

![sample picutre](https://github.com/jkaewprateep/distance_attention_networks_input/blob/main/01.png?raw=true "sample picutre")

Random action play on items, ladder.

![sample picutre](https://github.com/jkaewprateep/distance_attention_networks_input/blob/main/02.png?raw=true "sample picutre")

Gamescores condition still be consider.

![sample picutre](https://github.com/jkaewprateep/distance_attention_networks_input/blob/main/MonsterKong.gif?raw=true "sample picutre")
