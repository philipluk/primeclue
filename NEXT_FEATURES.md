If you'd like some feature implemented, and it's not on the list below, 
let me know at [lukasz.wojtow@gmail.com](mailto:lukasz.wojtow@gmail.com).<br>

<h4>Display cost per data point</h4>
For Cost objective it would be interesting to see value of cost / data point

<h4>Include data header + option to view imported data</h4>
This would be some sort of list at the beginning of DataSet. Then, GUI would give option to view data with column names. 

<h4>Show false positive rate</h4>
Sometimes only 'true' predictions are interesting, so showing false positive rate is very informative.

<h4>Add MinFalsePositives to training objectives</h4>
For some data only 'true' predictions are interesting, and sometimes it's essential to have low false positive rate.
It's possible to do it as a bool option - then only 'true' predictions would be scored.
 
<h4>Allow appending data</h4>
This could be done by allowing more than one file in serializator directory.

<h4>Use GPU for executing trees</h4>
GPU seems to be much faster for single precision floats. I achieved 70x speedup on GeForce GT 710 vs 1 core on Ryzen 7 1800X. 
With a decent card speedup could go into 1000x

<h4>Display data coordinates for a trained classifier</h4>
It would be interesting to see what data the most successful classifier is using. If some columns
aren't used at all then data collection can be simplified.
    
<h4>Hint when a single data point compares two items</h4>
This seems advisable in learning sports results or comparing two items. A boolean flag would tell Primeclue that one data point
with n columns actually consists of two items with n/2 columns each. Such a flag would then cause Primeclue
to create nodes with two inputs shifted by n/2. So, for example, with 16 columns describing two teams (with 
8 columns for each team) Primeclue would create nodes that compare (subtract?) Data(0, 4) with Data(0, 12).

<h4>Train the worst class only</h4>
Train only the worst performing class. This could save resources and increase classifier score.

<h4>Load data from a file on server</h4>
Used when file is too big for a browser to handle.

<h4>Data dictionary</h4>
When importing data that have some non-numbers values Primeclue should automatically change words to numbers and use
this dictionary for classification later.

<h4>Apache Kafka integration</h4>
Receive data from an external source and add to existing data.

<h4>Read data gradually for training</h4>
Useful when there is too much data to hold in memory at once. 

<h4>Show correlation between training data score and verification data score for best trees</h4>
This could work as sort of warning that a problem is too hard, and a good score on one set does not correspond to
a good score on another set. 
