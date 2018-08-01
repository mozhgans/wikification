echo "5 started"
python wsd_eval.py -t 5  -w 5 -v  >> 5.log 2>&1 
echo "done"

echo "10 started"
python wsd_eval.py -t 10 -w 5 -v >> 10.log 2>&1
echo "done"

echo "15 started"
python wsd_eval.py -t 15 -w 5 -v >> 15.log 2>&1 
echo "done"

#nohup python wsd_eval.py -t 20 -w 5 -v > 20.log 2>&1 &

