# zips the simulation plots

rm pnglist.txt
touch pnglist.txt

for NPRODUCTS in 100
do
    for SCENARIO in 3 4
    do

	for CASE_DONE in "exo" "endo"
	do
	    LOCAL_DIR=J${NPRODUCTS}/${CASE_DONE}_v${SCENARIO}/figures_paper
	    echo "${LOCAL_DIR}/new_pseudo_vals_${CASE_DONE}_J=${NPRODUCTS}_v${SCENARIO}_T=5000.png" >> pnglist.txt
	    echo "${LOCAL_DIR}/new_semi_elast_${CASE_DONE}_J=${NPRODUCTS}_v${SCENARIO}_T=5000.png" >> pnglist.txt
	done

	for CASE_DONE in  "exo_demog" "endo_demog"
		  do
	    LOCAL_DIR=J${NPRODUCTS}/${CASE_DONE}_v${SCENARIO}/figures_paper
	    for PI_VAL in 0 1 2
	     do
	      echo "${LOCAL_DIR}/new_pseudo_vals_${CASE_DONE}_J=${NPRODUCTS}_v${SCENARIO}_T=5000_pi${PI_VAL}.png"  >> pnglist.txt
	      echo "${LOCAL_DIR}/new_semi_elast_${CASE_DONE}_J=${NPRODUCTS}_v${SCENARIO}_T=5000_pi${PI_VAL}.png" >> pnglist.txt
	     done
	    done

done

#echo "show_paper_plots.py"  >> pnglist.txt

zip -@ - < pnglist.txt > pngs.zip

done
