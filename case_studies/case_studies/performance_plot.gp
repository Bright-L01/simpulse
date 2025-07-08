
# Gnuplot script for performance visualization
set terminal png size 800,600
set output 'performance_comparison.png'

set title 'Lean 4 Simp Optimization Performance'
set ylabel 'Compilation Time (seconds)'
set xlabel 'Mathlib4 Modules'
set xtic rotate by -45 scale 0

set style data histogram
set style histogram cluster gap 1
set style fill solid border -1
set boxwidth 0.9

plot 'performance_data.dat' using 2:xtic(1) title 'Baseline' lt rgb "#ff7f7f",      '' using 3 title 'Optimized' lt rgb "#7f7fff"
