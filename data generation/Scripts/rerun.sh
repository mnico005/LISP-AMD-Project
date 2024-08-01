num=0
while [ -d "$num/" ]; 
do 
    cd $num
    rm params.in
    cp ~/macsimParams/params.in params.in
    ~/simulators/macsimB4cycleStamp/bin/macsim &
    cd ../
    num=$(( $num + 1 )) 
done
