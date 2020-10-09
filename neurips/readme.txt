The code provides can be executed in Python and has the following requirements.

Python Package Requirement
Python version: 3.7.6

Package	Version
numpy		1.18.1
argparse 	1.1
scipy  		1.4.1
networkx  	2.4
pickle  	4.0


To run the synthetic experiments, please use the following command:

python main_algorithm -d 10 -noiselevel 0.2 -n 1000 -k 500 -output ./

-d 					integer, dimensionality of samples
-noiselevel			float between 0 and 1, the fraction of comparison labels being flipped.
-n 					integer, number of samples, if provided, the value provided by -m will be ignored.
-k 					integer, number of times a sample is involved in comparisons.
-m 					integer, number of comparisons
-output 			string, utput file folder