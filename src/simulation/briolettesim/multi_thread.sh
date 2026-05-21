

for i in {0..15}
do
    target/release/briolette-sim sobol -p results/FAIR_params_4_final/FAIR_params_percent_$i.txt -r 10 -s 10000 -t 2 > output_params_$i.log 2>&1 &
    
done
