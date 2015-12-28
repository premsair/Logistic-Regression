# Logistic-Regression
Logistic Regression Classifier to predict the genre of an audio file

README
Prem Sai Kumar Reddy Gangana 
(psreddy@unm.edu)
04/10/2015


My Program is written on Ananconda Distribution of Python v2.7 (64 bit) on the Integration Development Environment (IDE) Spyder v2.3.4 
over operating system Windows8.1. 

Following packages are imported to aid at different stages in programming and in code.
matplotlib==1.4.3
numpy==1.9.2
scikit-learn==0.15.2
scipy==0.15.1
scikits-talkbox==0.2.3

Usage:
To change the parameters, please edit the following lines.
main_pgm.py:	learning_rate=0.01 -> 36
		          penalty_term=0.001 -> 37
	          	epochs=300         -> 38

fft_data.py:    path of the audio files -> 21,23 
mfcc_daya.py:   path of the audio files -> 21,23

Note: Path of the files is set as the current working directory in the code which means the code files (.py) and the genre folders are 
in the same working direcotry. If you need to change the path then replace ./ with the absolute path of the genre folders only not the files.

If you already have fft and mfcc data ready with you in the order ['classical','country','jazz','metal','pop','rock'], then you can put them in
the same directory as the code files and then uncomment below lines in the file main_pgm.py
91-->  fft_Data=np.load('../data/fftdata.npy')
113--> fft_Data=np.load('../data/fftdata.npy')
136--> mfcc_Data=np.load('../data/mfccdata.npy')

Note: If the data is not in the order mentioned above, then the accuracies you get might differ from what i have reported.

Execution:

On Windows, just double click the main_pgm.py, it will load the other modules and creates .pyc files for them.
On Linux, use this command: python27 main_pgm.py or python2.7 main_pgm.py (depending on the softlink provided for python)

Note: Please make sure that all the required packages mentioned above exists in the system before execution
