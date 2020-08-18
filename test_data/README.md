<h1>Learning</h1>
Primeclue's learning powers were checked on publicly available data listed below.

Testing procedure:

1. Split data randomly into training and testing sets.
2. Train classifiers on training data for 10 minutes.
3. Take the best classifier and note its result on test data.
4. Repeat above steps 19 times.
5. Record median result on test data. 

<h2>Tested data</h2>

Table below shows median best result ([AUC - Area Under the Curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve)) for test data.

Data name | Median AUC
--- | ---
Adult income | 0.9 
Banknote verification | 1.0
Breast cancer diagnosis | 0.77 
Companies 1 year bankruptcy | 0.91
Credit card default | 0.77
Crime rate | 0.88
Heart disease | 0.8                                    
Hepatitis deaths | 0.76                   
Online news popularity | 1.0     
Stocks | 0.72 
Fizz Buzz | 0.96    
Breast cancer with mistake: | 1.0
Random | 0.50

Tests were done by running `./run_median_check.sh`

<h2>More about data</h2>
<h3>Adult income</h3>
Predict whether income exceeds $50K/yr based on census data. Also known as "Census Income" dataset.<br>
https://archive.ics.uci.edu/ml/datasets/Adult

<h3>Banknote verification</h3>
Data were extracted from images that were taken from genuine and forged banknote-like specimens. For digitization, an industrial camera usually used for print inspection was used. The final images have 400x 400 pixels. Due to the object lens and distance to the investigated object gray-scale pictures with a resolution of about 660 dpi were gained. Wavelet Transform tool were used to extract features from images.<br>
http://archive.ics.uci.edu/ml/datasets/banknote+authentication

<h3>Breast cancer diagnosis</h3>
Clinical features were observed or measured for 64 patients with breast cancer and 52 healthy controls.
The predictors are anthropometric data and parameters which can be gathered in routine blood analysis.<br>
http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Coimbra<br> 
[Patricio, 2018] Patrício, M., Pereira, J., Crisóstomo, J., Matafome, P., Gomes, M., Seiça, R., & Caramelo, F. (2018). Using Resistin, glucose, age and BMI to predict the presence of breast cancer. BMC Cancer, 18(1)

<h3>Companies 1 year bankruptcy</h3>
The dataset is about bankruptcy prediction of Polish companies.The bankrupt companies were analyzed in the period 2000-2012, while the still operating companies were evaluated from 2007 to 2013.<br>
http://archive.ics.uci.edu/ml/datasets/Polish+companies+bankruptcy+data<br>
Zieba, M., Tomczak, S. K., & Tomczak, J. M. (2016). Ensemble Boosted Trees with Synthetic Features Generation in Application to Bankruptcy Prediction. Expert Systems with Applications.

<h3>Credit card default</h3>
http://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients<br>
Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. Expert Systems with Applications, 36(2), 2473-2480.

<h3>Crime rate</h3>
Communities in the US. Data combines socio-economic data from the '90 Census, law enforcement data from the 1990 Law Enforcement Management and Admin Stats survey, and crime data from the 1995 FBI UCR<br>
http://archive.ics.uci.edu/ml/datasets/Communities+and+Crime+Unnormalized <br>
UCI Machine Learning Repository
U. S. Department of Commerce, Bureau of the Census, Census Of Population And Housing 1990 United States: Summary Tape File 1a & 3a (Computer Files), 
U.S. Department Of Commerce, Bureau Of The Census Producer, Washington, DC and Inter-university Consortium for Political and Social Research Ann Arbor, Michigan. (1992) 
U.S. Department of Justice, Bureau of Justice Statistics, Law Enforcement Management And Administrative Statistics (Computer File) U.S. Department Of Commerce, Bureau Of The Census Producer, Washington, DC and Inter-university Consortium for Political and Social Research Ann Arbor, Michigan. (1992) 
U.S. Department of Justice, Federal Bureau of Investigation, Crime in the United States (Computer File) (1995) 

<h3>Heart disease</h3>                                            
Data on cardiac Single Proton Emission Computed Tomography (SPECT) images. Each patient classified into two categories: normal and abnormal.<br>
http://archive.ics.uci.edu/ml/datasets/SPECTF+Heart

<h3>Hepatitis deaths</h3>                             
From G.Gong: CMU; Mostly Boolean or numeric-valued attribute types; Includes cost data (donated by Peter Turney)<br>
http://archive.ics.uci.edu/ml/datasets/Hepatitis

<h3>Online news popularity</h3>      
This dataset summarizes a heterogeneous set of features about articles published by Mashable in a period of two years.<br>
http://archive.ics.uci.edu/ml/datasets/Online+News+Popularity <br>
K. Fernandes, P. Vinagre and P. Cortez. A Proactive Intelligent Decision Support System for Predicting the Popularity of Online News. Proceedings of the 17th EPIA 2015 - Portuguese Conference on Artificial Intelligence, September, Coimbra, Portugal.
        
<h3>Stocks</h3>
Predicting whether a stock will double in price within next year. Data collected from Polish stock market. Each line represents features
of a single stock and WIG (Polish broad market index) at the beginning of a year. Features include things like PE, PBV,
Williams R indicator and past price changes. The objective was to predict whether the stock's price will at least double during the year.     
          
<h3>Fizz buzz</h3>
Variation of [popular game](https://en.wikipedia.org/wiki/Fizz_buzz). Objective was to predict one of four classes:<br>
`Fizz` for numbers % 3 == 0<br>
`Buzz` for numbers % 5 == 0<br>
`FizzBuzz` for numbers % 15 == 0<br>
`Value` for every other number.<br>
To make things more complicated, data was not shuffled. In other words, lower numbers went to training data and higher numbers to 
test data. This way Primeclue had to learn the rules instead of approximating. 

  
<h3>Breast cancer with mistake</h3>
This test was done to see what happens if output column is imported as input by mistake (i.e. Primeclue has access to outcome during training).
 Primeclue should be smart enough to figure out that one of the columns gives 100% accuracy.
This test was only done for 1 minute in each round.
                                     
<h3>Random</h3>
This is totally random data generated by a computer. First 10 columns are random numbers, last column is a random bit.
As there is nothing to predict it is a good way to check for data leak / curve fitting. Results on training data
can go to around 65-70% correctness. Obviously, result on test data stay around 50%.   

<h2>Files description</h2>
Folder _original_: contains original files<br>
Folder _processed_: contains processed files before import; last column is target<br>
Folder _primeclue_data_: contains files in Primeclue's format; this can be moved to ~/Primeclue/data<br>                 
