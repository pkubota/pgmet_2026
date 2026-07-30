'reinit'
'set display color white'
'clear'
'open barogp.ctl'

it=1
while( it <=200)

'd z(t='it')'
'!sleep 1'
'clear'
it=it+5
endwhile
