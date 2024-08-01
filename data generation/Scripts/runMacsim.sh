format=trace_0_*.raw 
num=0
cp -r ~/macsimParams/tools $PWD/
for file in $format; do 
    mkdir $num
    cp $file $num/trace_0.raw
    rm $file
    cp trace.txt $num/trace.txt
    cp ~/macsimParams/params.in $num/params.in
    cp trace_0_$num.txt $num/trace_0.txt
    rm trace_0_$num.txt
    printf "1\n$PWD/$num/trace.txt" > $num/trace_file_list
    cd $num
    ~/simulators/macsimB4cycleStamp/bin/macsim &
    cd ../
    num=$(( $num + 1 )) 
done
