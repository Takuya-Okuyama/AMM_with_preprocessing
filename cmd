./AMM -m 4096 -k 4096 -n 4096 -c 4096 -r 100 -kernel exact -seed 1 -alpha 1.0 -beta 0.0 -verbose 2 -sanity > log_exact

./AMM -m 4096 -k 4096 -n 4096 -c 256 -r 1000 -kernel previousAMM -seed 1 -verbose 2 -sanity


./AMM -m 4096 -k 4096 -n 4096 -c 4096 -r 100 -kernel previousAMM -seed 1 -alpha 1.0 -beta 0.0 -verbose 2

./AMM -m 4096 -k 4096 -n 4096 -c 4096 -r 100 -kernel exact -seed 1 -alpha 1.0 -beta 0.0 -verbose 2 -sanity

nsys nvprof ./AMM -m 4096 -k 4096 -n 4096 -c 256 -r 10 -kernel previousAMM -seed 1 -alpha 1.0 -beta 0.0 -verbose 2

./AMM -m 4096 -k 4096 -n 4096 -c 256 -r 1000 -kernel previousAMM -matrixType_A gaussian_1.0_1.0 -matrixType_B gaussian_1.0_1.0 -seed 1 -sanity -verbose 2

./AMM -m 4096 -k 4096 -n 4096 -c 256 -r 1000 -kernel proposedAMM -matrixType_A gaussian_1.0_1.0 -matrixType_B gaussian_1.0_1.0 -seed 1 -sanity -verbose 2 > log.csv

./AMM -m 16384 -k 16384 -n 16384 -c 16384 -r 100 -kernel exact -seed 1 -alpha 1.0 -beta 0.0 -sanity -verbose 2
./AMM -m 16384 -k 16384 -n 16384 -c 1024 -r 100 -kernel previousAMM -seed 1 -alpha 1.0 -beta 0.0 -sanity -verbose 2

32,16384,16384,16384,16384,100,exact,gaussian_0.0_1.0,gaussian_0.0_1.0,1,1.000000,0.000000,16385.015625,16382.773438,2097038.500000,46130.175781,461.301758,0.000000,0.000000,0.000000
32,16384,16384,16384,1024,100,previousAMM,gaussian_0.0_1.0,gaussian_0.0_1.0,1,1.000000,0.000000,16385.015625,16382.773438,2097038.500000,3590.081543,35.900814,8379848.500000,8355510.000000,8412500.000000

./AMM -m 4096 -k 4096 -n 4096 -c 1024 -r 1000 -kernel previousAMM -matrixType_A gaussian_1.0_1.0 -matrixType_B gaussian_1.0_1.0 -seed 1

./AMM -m 4096 -k 4096 -n 4096 -c 256 -r 1000 -kernel previousAMM -matrixType_A gaussian_1.0_1.0 -matrixType_B gaussian_1.0_1.0 -seed 1 -sanity > log_previous.csv
./AMM -m 4096 -k 4096 -n 4096 -c 256 -r 1000 -kernel proposedAMM -matrixType_A gaussian_1.0_1.0 -matrixType_B gaussian_1.0_1.0 -seed 1 -sanity > log_proposed.csv