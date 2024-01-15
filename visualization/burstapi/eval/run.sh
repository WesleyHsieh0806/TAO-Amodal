python burstapi/eval/create_dirtree.py $@

if [ $? -eq 0 ]; then
    if [ -f "${TRACKEVAL_DIR}/scripts/_tmp_burst_eval.sh" ]; then
        cd ${TRACKEVAL_DIR}/scripts
        bash _tmp_burst_eval.sh
        rm _tmp_burst_eval.sh
    fi
fi