echo R1
python3 TransR_new.py --i 2 --d 100 --e 600 --lr .00001 > out/TransR_11.txt
echo R0
python3 TransE_new.py --i 0 --d 100 --e 550 --lr .0002 --p 0 --s 1 > out/TransR_00.txt
python3 TransR_new.py --i 2 --d 100 --e 600 --lr .00001 > out/TransR_01.txt
