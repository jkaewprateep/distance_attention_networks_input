# distance_attention_networks_input
For study distance input networks for machine learning problems, snake ladders, or tunnel shared distance. We can add the problem parameters into our AI training templates for study and solve the problem with the conditions you input.

## Objectives ##

🧸💬 How do we reach the last room in the mazes from the current room is to find a tunnel and paths when the tunnel is a ladder and the path is distance.

```
contrl = steps + gamescores + ( 50 * reward )
contr2 = list_ladder[len(list_ladder) - 1][1]
contr3 = 1
```

## List and relative distance crate ##

👧💬 With simple problem type we know that ```distance``` value can be estimate from x and y values of player curser and target as ```pow( X1 - X2 ) + pow( Y1 - Y2 )```.

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

![sample picutre](https://github.com/jkaewprateep/distance_attention_networks_input/blob/main/01.png?raw=true "sample picutre")

![sample picutre](https://github.com/jkaewprateep/distance_attention_networks_input/blob/main/02.png?raw=true "sample picutre")

![sample picutre](https://github.com/jkaewprateep/distance_attention_networks_input/blob/main/MonsterKong.gif?raw=true "sample picutre")
