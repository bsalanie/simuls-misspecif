# copies the simulation results from the AWS EC2 instance that created them
SIMULS_DIR_AWS=ubuntu@35.175.107.214:simuls_MNL


for NPRODUCTS in 50
do
    for SCENARIO in 3 4
    do

	for CASE_DONE in "exo" "endo" "exo_demog" "endo_demog"
	do
	    LOCAL_DIR=J${NPRODUCTS}/${CASE_DONE}_v${SCENARIO}
	    mkdir -p ${LOCAL_DIR}
	    echo "scp -i ~/.ssh/devenv-key.pem ${SIMULS_DIR_AWS}/${LOCAL_DIR}/simul_results_${CASE_DONE}_J=${NPRODUCTS}_v${SCENARIO}_T=10000.pkl ${LOCAL_DIR}"
	    scp -i ~/.ssh/devenv-key.pem ${SIMULS_DIR_AWS}/${LOCAL_DIR}/simul_results_${CASE_DONE}_J=${NPRODUCTS}_v${SCENARIO}_T=10000.pkl ${LOCAL_DIR}
	done
    done
done
