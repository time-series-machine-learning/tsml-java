# UEA Time Series Classification - CAWPE Stable Branch

.. image:: https://travis-ci.com/uea-machine-learning/tsml.svg?branch=paper/cawpe
    :target: https://travis-ci.com/uea-machine-learning/tsml

This is a stable branch in support of the paper 'A probabilistic classifier ensemble weighting scheme based on cross-validated accuracy estimates'. It is intended to provde a snapshot of the code at the time of final submission, as well as a stable location for summaries of results. See the associated [website](http://www.timeseriesclassification.com/CAWPE.php) for more details. 

### Supplementary Material

Results summaries can be found at [CAWPEResults/](https://github.com/TonyBagnall/uea-tsc/tree/paper/cawpe/CAWPEResults), or downloaded directly via the above site. 

Code for using our analysis pipeline (contantly being updated) in order to produce the results for the paper is in [src/main/java/evaluation/CAWPEResultsCollationCode.java](https://github.com/TonyBagnall/uea-tsc/blob/paper/cawpe/src/main/java/evaluation/CAWPEResultsCollationCode.java). A stable snapshot of the code used to create results and analysis at the time of initial creation (2017) is available via the associated site too.

The [CAWPE class](https://github.com/TonyBagnall/uea-tsc/blob/paper/cawpe/src/main/java/vector_classifiers/CAWPE.java) itself also contains example code to produce all base classifier and post-processed ensemble results for figure 3, buildCAWPEPaper_AllResultsForFigure3().

### Authors

* james.large@uea.ac.uk - lead author 
* j.lines@uea.ac.uk - secondary author
* ajb@uea.ac.uk - corresponding author
