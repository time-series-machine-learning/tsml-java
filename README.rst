

UEA Time Series Classification
===============================

.. image:: https://travis-ci.com/uea-machine-learning/tsml.svg?branch=master
    :target: https://travis-ci.com/uea-machine-learning/tsml

A `Weka <https://svn.cms.waikato.ac.nz/svn/weka/branches/stable-3-8/>`__-compatible Java toolbox for
**time series classification, clustering and transformation**. For the python sklearn-compatible version, see 
`sktime <https://github.com/alan-turing-institute/sktime>`__

Find out more info about our broader work and dataset hosting for the UCR univariate and UEA multivariate time series classification archives on our `website <http://www.timeseriesclassification.com>`__.

This codebase is actively being developed for our research. The dev branch will contain the most up-to-date, but stable, code. 

Installation
------------
We are looking into deploying this project on Maven or Gradle in the future. For now there are two options:

* download the `jar file <https://github.com/TonyBagnall/uea-tsc/TSC jar 31_5_20.zip>`__ and include as a dependency in your project, or you can run experiments through command line, see the `examples on running experiments <https://github.com/uea-machine-learning/tsml/blob/dev/src/main/java/examples/Ex04_ThoroughExperiments.java>`__
* fork or download the source files and include in a project in your favourite IDE you can then construct your own experiments (see our `examples <https://github.com/uea-machine-learning/tsml/tree/dev/src/main/java/examples>`__) and implement your own classifiers.

Overview
--------

This codebase mainly represents the implementation of different algorithms in a common framework, which at the time leading up to the `Great Time Series Classification Bake Off <https://link.springer.com/article/10.1007/s10618-016-0483-9>`__ in particular was a real problem, with implementations being in any of Python, C/C++, Matlab, R, Java, etc. or even combinations thereof. 

We therefore mainly provide implementations of different classifiers as well as experimental and results analysis pipelines with the hope of promoting and streamlining open source, easily comparable, and easily reproducible results, specifically within the TSC space. 

While they are obviously very important methods to study, we shall very likely not be implementing any kind of deep learning methods in our codebase, and leave those rightfully in the land of optimised languages and libraries for them, such as `sktime-dl <https://github.com/uea-machine-learning/sktime-dl>`__ , the Keras-enabled extension to `sktime <https://github.com/alan-turing-institute/sktime>`__. 

Our `examples <https://github.com/uea-machine-learning/tsml/tree/dev/src/main/java/examples>`__ run through the basics of using the code, however the basic layout of the codebase is this:

`evaluation/ <https://github.com/uea-machine-learning/tsml/tree/master/src/main/java/evaluation>`__ 
    contains classes for generating, storing and analysing the results of your experiments
    
`experiments/ <https://github.com/uea-machine-learning/tsml/tree/master/src/main/java/experiments>`__ 
    contains classes specifying the experimental pipelines we utilise, and lists of classifier and dataset specifications. The 'main' class is `Experiments.java <https://github.com/uea-machine-learning/tsml/blob/master/src/main/java/experiments/Experiments.java>`__, however other experiments classes exist for running on simulation datasets or for generating transforms of time series for later classification, such as with the Shapelet Transform. 

`timeseriesweka/ <https://github.com/uea-machine-learning/tsml/tree/master/src/main/java/timeseriesweka>`__ and `multivariate_timeseriesweka/ <https://github.com/uea-machine-learning/tsml/tree/master/src/main/java/multivariate_timeseriesweka>`__ 
    contain the TSC algorithms we have implemented, for univariate and multivariate classification respectively. 

`weka_extras/ <https://github.com/uea-machine-learning/tsml/tree/master/src/main/java/weka_extras>`__ 
    contains extra algorithm implementations that are not specific to TSC, such as generalised ensembles or classifier tuners. 

Implemented Algorithms
----------------------

Classifiers
```````````

The lists of implemented TSC algorithms shall continue to grow over time. These are all in addition to the standard Weka classifiers and non-TSC algorithms defined under the weka_extras package. 

We have implemented the following bespoke classifiers for univariate, equal length time series classification:

===============  ================  ==============  =================  ==============  =========
Distance Based   Dictionary Based  Spectral Based  Shapelet Based     Interval Based  Ensembles         
===============  ================  ==============  =================  ==============  =========
DD_DTW           BOSS              RISE            LearnShapelets     TSF             FlatCote
DTD_C            cBOSS             cRISE           ShapeletTransform  TSBF            HiveCote
ElasticEnsemble  BOP                               FastShapelets      LPS
NN_CID           WEASEL        
SAX_1NN          SAXVSM
ProximityForest              
===============  ================  ==============  =================  ==============  =========

And we have implemented the following bespoke classifiers for multivariate, equal length time series classification:

========  =============================
NN_ED_D   MultivariateShapeletTransform
NN_ED_I   ConcatenateClassifier
NN_DTW_D  NN_DTW_A
NN_DTW_I
========  =============================

Clusterers
``````````

Currently quite limited, aside from those already shipped with Weka. 

=====================  =======
UnsupervisedShapelets
=====================  =======

Filters/Transformations
```````````````````````

SimpleBatchFilters that take an Instances (the set of time series), transforms them
and returns a new Instances object

===================  ===================  ===================
ACF                  ACF_PACF             ARMA
BagOfPatternsFilter  BinaryTransform      Clipping
Correlation          Cosine               DerivativeFilter
Differences          FFT                  Hilbert
MatrixProfile        NormalizeAttribute   NormalizeCase
PAA                  PACF                 PowerCepstrum
PowerSepstrum        RankOrder            RunLength
SAX                  Sine                 SummaryStats
===================  ===================  ===================

Paper-Supporting Branches
-------------------------

This project acts as the general open-source codebase for our research, especially the `Great Time Series Classification Bake Off <https://link.springer.com/article/10.1007/s10618-016-0483-9>`__. We are also trialling a process of creating stable branches in support of specific outputs. 

Current branches of this type are: 

* `paper/cawpe/ <https://github.com/uea-machine-learning/tsml/tree/paper/cawpe>`__ in support of `"A probabilistic classifier ensemble weighting scheme based on cross-validated accuracy estimates" <https://link.springer.com/article/10.1007/s10618-019-00638-y>`__

* `paper/cawpeExtension/ <https://github.com/uea-machine-learning/tsml/tree/paper/cawpeExtension>`__ in support of "Mixing hetero- and homogeneous models in weighted ensembles" (Accepted/in-press)

Contributors
------------

Lead: Anthony Bagnall (@TonyBagnall, `@tony_bagnall <https://twitter.com/tony_bagnall>`__, ajb@uea.ac.uk)

* James Large (@James-Large, `@jammylarge <https://twitter.com/jammylarge>`__, james.large@uea.ac.uk)
* Jason Lines (@jasonlines), 
* George Oastler (@goastler), 
* Matthew Middlehurst (@MatthewMiddlehurst), 
* Michael Flynn (@Michael Flynn), 
* Aaron Bostrom (@ABostrom, `@_Groshh_ <https://twitter.com/_Groshh_>`__, a.bostrom@nua.ac.uk), 
* Patrick Schäfer (@patrickzib)
* Chang Wei Tan (@ChangWeiTan)

We welcome anyone who would like to contribute their algorithms! 

License 
-------

GNU General Public License v3.0
