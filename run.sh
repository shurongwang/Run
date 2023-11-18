echo E0
python3 TransE_new.py --i 0 --d 100 --e 1500 --lr .0001 > out/TransE_0.txt
echo H0
python3 TransH_new.py --i 0 --d 100 --e 1500 --lr .0001 > out/TransH_0.txt
echo M0
python3 DistMult_new.py --i 0 --d 100 --e 1500 --lr .00005 > out/DistMult_0.txt
echo E1
python3 TransE_new.py --i 1 --d 100 --e 1800 --f 100 --lr .0001 > out/TransE_1.txt
echo H1
python3 TransH_new.py --i 1 --d 100 --e 1800 --f 100 --lr .0001 > out/TransH_1.txt
echo M1
python3 DistMult_new.py --i 1 --d 100 --e 2000 --lr .00005 > out/DistMult_1.txt
echo R0
python3 TransE_new.py --i 0 --d 100 --e 550 --lr .0002 --p 0 --s 1 > out/TransR_00.txt
python3 TransR_new.py --i 2 --d 100 --e 600 --lr .00001 > out/TransR_01.txt
echo R1
python3 TransE_new.py --i 1 --d 100 --e 750 --f 100 --lr .0002 --p 0 --s 1 > out/TransR_10.txt
python3 TransR_new.py --i 2 --d 100 --e 600 --lr .00001 > out/TransR_11.txt
