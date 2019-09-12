

UEA Time Series Classification
===============================

.. image:: https://travis-ci.com/tonybagnall/uea-tsc.svg?branch=master
    :target: https://travis-ci.com/tonybagnall/uea-tsc   

Find more info on our `website <http://www.timeseriesclassification.com>`__.

A `Weka <https://svn.cms.waikato.ac.nz/svn/weka/branches/stable-3-8/>`__ compatible Java toolbox for
time series classification, clustering and transformation. For the python, sklearn compatible version, see 
_sktime <https://github.com/alan-turing-institute/sktime>`__

Installation
------------
We are looking at getting this on Maven. For now there are two options:

* download the `Jar File <https://github.com/TonyBagnall/uea-tsc/TSC jar 31_5_20.zip>`__
* download the source file and include in a project in your favourite IDE
you can then construct your own experiment (see `BasicExamples.java <https://github.com/TonyBagnall/uea-tsc/blob/master/src/main/java/examples/BasicExamples.java>`__) or 
the experimental structure we use (see `Experiments.java <https://github.com/TonyBagnall/uea-tsc/blob/master/src/main/java/experiments/Experiments.java>`__) 

Classifiers
------------
We have implemented the following bespoke classifiers for univariate, equal length time series classification

Distance Based

* DD_DTW 
* DTD_C
* ElasticEnsemble
* NN_CID
* SAX_1NN
* SAXVSM
* ProximityForest

Dictionary Based

* BOSS
* BOP
* WEASEL

Spectral Based

* RISE
* CRISE

Shaplet Based

* LearnShapelets
* ShapeletTransformClassifier
* FastShapelets

(to do: recover original ShapeletTree)

Interval Based

* TSF
* TSBF
* LPS

Ensembles

* FlatCote
* HiveCote

We have implemented the following bespoke classifiers for multivariate, equal length time series classification

* NN_ED_D
* NN_ED_I
* NN_DTW_D
* NN_DTW_I
* NN_DTW_A
* MultivariateShapeletTransformClassifier
* ConcatenateClassifier



Clusterers
------------
Currently quite limited. Standard approach would be to perform an unsupervised 

* UnsupervisedShapelets


Filters/Transformations
------------
SimpleBatchFilters that take an Instances (the set of time series), transforms them
and returns a new Instances object

* ACF
* ACF_PACF
* ARMA
* BagOfPatternsFilter
* BinaryTransform
* Clipping
* Correlation
* Cosine
* DerivativeFilter
* Differences
* FFT
* Hilbert
* MatrixProfile
* NormalizeAttribute
* NormalizeCase
* PAA
* PACF
* PowerCepstrum
* PowerSepstrum
* RankOrder
* RunLength
* SAX
* Sine
* SummaryStats

