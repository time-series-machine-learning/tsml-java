/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package experiments;


import evaluation.tuning.ParameterSpace;
import experiments.Experiments.ExperimentalArguments;
import machine_learning.classifiers.tuned.TunedClassifier;
import tsml.classifiers.distance_based.elastic_ensemble.ElasticEnsemble;
import tsml.classifiers.distance_based.knn.KNNLOOCV;
import tsml.classifiers.hybrids.HIVE_COTE;
import tsml.classifiers.dictionary_based.*;
import tsml.classifiers.dictionary_based.boss_variants.BOSSC45;
import tsml.classifiers.dictionary_based.SpatialBOSS;
import tsml.classifiers.dictionary_based.boss_variants.BoTSWEnsemble;
import tsml.classifiers.distance_based.*;
import tsml.classifiers.frequency_based.RISE;
import tsml.classifiers.legacy.COTE.FlatCote;
import tsml.classifiers.legacy.COTE.HiveCote;
import tsml.classifiers.hybrids.TSCHIEFWrapper;
import tsml.classifiers.interval_based.TSF;
import tsml.classifiers.shapelet_based.ShapeletTransformClassifier;
import tsml.classifiers.shapelet_based.FastShapelets;
import tsml.classifiers.shapelet_based.LearnShapelets;
import tsml.classifiers.interval_based.LPS;
import tsml.classifiers.multivariate.MultivariateShapeletTransformClassifier;
import tsml.classifiers.multivariate.NN_DTW_A;
import tsml.classifiers.multivariate.NN_DTW_D;
import tsml.classifiers.multivariate.NN_DTW_I;
import tsml.classifiers.multivariate.NN_ED_I;
import tsml.classifiers.shapelet_based.ShapeletTree;
import weka.core.EuclideanDistance;
import weka.core.Randomizable;
import machine_learning.classifiers.ensembles.CAWPE;
import machine_learning.classifiers.PLSNominalClassifier;
import machine_learning.classifiers.kNN;
import machine_learning.classifiers.tuned.TunedXGBoost;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.classifiers.meta.RotationForest;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;

import java.util.Arrays;
import java.util.HashSet;
import java.util.concurrent.TimeUnit;

/**
 *
 * @author James Large (james.large@uea.ac.uk) and Tony Bagnall
 */
public class ClassifierLists {

    //All implemented classifiers in tsml
    //<editor-fold defaultstate="collapsed" desc="All univariate time series classifiers">
    public static String[] allUnivariate={
//Distance Based
            "DTW","DTWCV","ApproxElasticEnsemble","ProximityForest","ElasticEnsemble","FastElasticEnsemble",
            "DD_DTW","DTD_C", "NN_CID","MSM","TWE","WDTW",
//Dictionary Based
            "BOSS", "BOP", "SAXVSM", "SAX_1NN", "WEASEL", "cBOSS", "BOSSC45", "S-BOSS","BoTSWEnsemble",
//Interval Based
            "LPS","TSF","cTSF",
//Frequency Based
            "RISE",
//Shapelet Based
            "FastShapelets","LearnShapelets","ShapeletTransformClassifier",
//Hybrids
            "HiveCote","FlatCote"
};
    //</editor-fold>
    public static HashSet<String> allClassifiers=new HashSet<String>( Arrays.asList(allUnivariate));

    /**
     * DISTANCE BASED: classifiers based on measuring the distance between two classifiers
     */
    public static String[] distance= {
        "DTW","DTWCV","ApproxElasticEnsemble","ProximityForest","FastElasticEnsemble",
            "DD_DTW","DTD_C","NN_CID",
        "EE",
        "LEE",
        "TUNED_DTW_1NN_V1"
    };
    public static HashSet<String> distanceBased=new HashSet<String>( Arrays.asList(distance));
    private static Classifier setDistanceBased(Experiments.ExperimentalArguments exp){
        String classifier=exp.classifierName;
        Classifier c = null;
        int fold=exp.foldId;
        switch(classifier) {
            case "EE":
                c = ElasticEnsemble.FACTORY.EE_V2.build();
                break;
            case "LEE":
                c = ElasticEnsemble.FACTORY.LEE.build();
                break;
            case "ApproxElasticEnsemble":
                c = new ApproxElasticEnsemble();
                break;
            case "ProximityForest":
                c = new ProximityForestWrapper();
                break;
            case "FastElasticEnsemble":
                c=new FastElasticEnsemble();
                break;
            case "DD_DTW":
                c=new DD_DTW();
                break;
            case "DTD_C":
                c=new DTD_C();
                break;
            case "CID_DTW":
                c=new NN_CID();
                ((NN_CID)c).useDTW();
                break;
            case "NN_CID":
                c = new NN_CID();
                break;
            default:
                System.out.println("Unknown distance based classifier "+classifier+" should not be able to get here ");
                System.out.println("There is a mismatch between array distance and the switch statement ");
                throw new UnsupportedOperationException("Unknown distance based  classifier "+classifier+" should not be able to get here. "
                        + "There is a mismatch between array distance and the switch statement.");

        }
        return c;
    }
    /**
     * DICTIONARY BASED: classifiers based on counting the occurrence of words in series
     */
    public static String[] dictionary= {
        "BOSS", "BOP", "SAXVSM", "SAX_1NN", "WEASEL", "cBOSS", "BOSSC45", "S-BOSS", "SpatialBOSS", "BoTSWEnsemble",

            "cS-BOSS","BcS-BOSS","KTune-cS-BOSS","WeightTune-cS-BOSS","Bigram-BcS-BOSS","Bigram-BcBOSS",
            "HI-Bigram-BcS-BOSS","HI-BcS-BOSS","HI-cS-BOSS","FBcS-BOSS","WinLenScale-FBcS-BOSS","HI-FBcS-BOSS",
            "Wise-FBcS-BOSS","Wise2-FBcS-BOSS","Logistic-FBcS-BOSS","FCNN-BcS-BOSS","HI-FCNN-BcS-BOSS",
            "HI-LimitFCNN-BcS-BOSS","HI-SLimitFCNN-BcS-BOSS","HI-CompFCNN-BcS-BOSS",
            "IGB-BcS-BOSS","Anova-BcS-BOSS","MFT-BcS-BOSS","DFT-BcS-BOSS","AnovaMFT-BcS-BOSS","AnovaIGB-BcS-BOSS",
            "pIGB-BcS-BOSS","pAnova-BcS-BOSS","pAnovaIGB-BcS-BOSS","HI-SLimitFCNN-pIGB-BcS-BOSS","HI-pIGB-BcS-BOSS",
            "HI-100pIGB-BcS-BOSS","HI-100pAnovaIGB-BcS-BOSS","HI-Bigram-pIGB-BcS-BOSS","HI-TuneWeight-pIGB-BcS-BOSS",
            "HI-nrBigram-pIGB-BcS-BOSS","HI-nr-pIGB-BcS-BOSS","HI-pIGB-cS-BOSS","HI-pIGB-750cS-BOSS",
            "HI-BIpIGB-cS-BOSS","HI-BIpIGB-750cS-BOSS","HI-Bigram-pIGB-Bc-BOSS","Bigram-pIGB-BcS-BOSS",
    "HI-Cutoff-Bigram-pIGB-BcS-BOSS","HI-tp80-Bigram-pIGB-BcS-BOSS","HI-tp60-Bigram-pIGB-BcS-BOSS","HI-tp50-Bigram-pIGB-BcS-BOSS","HI-l4-Bigram-pIGB-BcS-BOSS",
            "HI-pBigram-pIGB-BcS-BOSS","HI-fs-pIGB-BcS-BOSS","HI-500s-pIGB-BcS-BOSS","HI-100m-pIGB-BcS-BOSS",
            "HI-tp100-Bigram-pIGB-BcS-BOSS","cBOSS-Max100","TDE-1H","TDE-4H","TDE-12H", "TDE-Cutoff70", "TDE-WordLength"};

    public static HashSet<String> dictionaryBased=new HashSet<String>( Arrays.asList(dictionary));
    private static Classifier setDictionaryBased(Experiments.ExperimentalArguments exp){
        String classifier=exp.classifierName;
        Classifier c;
        int fold=exp.foldId;
        switch(classifier) {
            case "BOP":
                c=new BagOfPatterns();
                break;
            case "SAXVSM":
                c=new SAXVSM();
                break;
            case "SAX_1NN":
                c=new SAXVSM();
                break;
            case "BOSS":
                c=new BOSS();
                break;
            case "cBOSS":
                c = new cBOSS();
                break;
            case "BOSSC45":
                c = new BOSSC45();
                break;
            case "SpatialBOSS": case "S-BOSS":
                c = new SpatialBOSS();
                break;
            case "BoTSWEnsemble":
                c = new BoTSWEnsemble();
                break;
            case "WEASEL":
                c = new WEASEL();
                break;


            case "cBOSS-Max100":
                c = new cBOSS();
                ((cBOSS)c).setMaxEnsembleSize(100);
                break;


            case "cS-BOSS":
                c = new cBOSSSP();
                ((cBOSSSP) c).chiLimits = new double[]{0};
                ((cBOSSSP) c).setBayesianParameterSelection(false);
                break;
            case "BcS-BOSS":
                c = new cBOSSSP();
                ((cBOSSSP) c).chiLimits = new double[]{0};
                break;
            case "KTune-cS-BOSS":
                c = new cBOSSSP();
                ((cBOSSSP) c).chiLimits = new double[]{0};
                ((cBOSSSP) c).setBayesianParameterSelection(false);
                ((cBOSSSP) c).tuneK = true;
                break;
            case "WeightTune-cS-BOSS":
                c = new cBOSSSP();
                ((cBOSSSP) c).chiLimits = new double[]{0};
                ((cBOSSSP) c).setBayesianParameterSelection(false);
                ((cBOSSSP) c).tuneWeight = true;
                break;
            case "Bigram-BcS-BOSS":
                c = new cBOSSSP();
                ((cBOSSSP) c).chiLimits = new double[]{0};
                ((cBOSSSP) c).bigrams = new boolean[]{true};
                break;
            case "Bigram-BcBOSS":
                c = new cBOSSSP();
                ((cBOSSSP) c).chiLimits = new double[]{0};
                ((cBOSSSP) c).levels = new Integer[]{1};
                ((cBOSSSP) c).bigrams = new boolean[]{true};
                break;
            case "HI-Bigram-BcS-BOSS":
                c = new cBOSSSP();
                ((cBOSSSP) c).chiLimits = new double[]{0};
                ((cBOSSSP) c).bigrams = new boolean[]{true};
                ((cBOSSSP) c).histogramIntersection = true;
                break;
            case "HI-BcS-BOSS":
                c = new cBOSSSP();
                ((cBOSSSP) c).chiLimits = new double[]{0};
                ((cBOSSSP) c).histogramIntersection = true;
                break;
            case "HI-cS-BOSS":
                c = new cBOSSSP();
                ((cBOSSSP) c).chiLimits = new double[]{0};
                ((cBOSSSP) c).setBayesianParameterSelection(false);
                ((cBOSSSP) c).histogramIntersection = true;
                break;
            case "FBcS-BOSS":
                c = new cBOSSSP();
                ((cBOSSSP) c).bigrams = new boolean[]{true};
                ((cBOSSSP) c).featureSelection = true;
                break;
            case "WinLenScale-FBcS-BOSS":
                c = new cBOSSSP();
                ((cBOSSSP) c).bigrams = new boolean[]{true};
                ((cBOSSSP) c).featureSelection = true;
                ((cBOSSSP) c).chiLimits = new double[]{0.3};
                ((cBOSSSP) c).limitOp = 1;
                break;
            case "HI-FBcS-BOSS":
                c = new cBOSSSP();
                ((cBOSSSP) c).bigrams = new boolean[]{true};
                ((cBOSSSP) c).featureSelection = true;
                ((cBOSSSP) c).histogramIntersection = true;
                break;
            case "Wise-FBcS-BOSS":
                c = new cBOSSSP();
                ((cBOSSSP) c).setSeed(fold);
                ((cBOSSSP) c).bigrams = new boolean[]{true};
                ((cBOSSSP) c).featureSelection = true;
                ((cBOSSSP) c).limitVal = 200000;
                break;
            case "Wise2-FBcS-BOSS":
                c = new cBOSSSP();
                ((cBOSSSP) c).setSeed(fold);
                ((cBOSSSP) c).bigrams = new boolean[]{true};
                ((cBOSSSP) c).featureSelection = true;
                ((cBOSSSP) c).chiLimits = new double[]{0.1, 0.2, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1};
                ((cBOSSSP) c).limitVal = 200000;
                break;
            case "Logistic-FBcS-BOSS":
                c = new cBOSSSP();
                ((cBOSSSP) c).bigrams = new boolean[]{true};
                ((cBOSSSP) c).featureSelection = true;
                ((cBOSSSP) c).useLogistic = true;
                break;
            case "FCNN-BcS-BOSS":
                c = new cBOSSSP();
                ((cBOSSSP) c).chiLimits = new double[]{0};
                ((cBOSSSP) c).FCNN = true;
                break;
            case "HI-FCNN-BcS-BOSS":
                c = new cBOSSSP();
                ((cBOSSSP) c).chiLimits = new double[]{0};
                ((cBOSSSP) c).FCNN = true;
                ((cBOSSSP) c).histogramIntersection = true;
                break;
            case "HI-LimitFCNN-BcS-BOSS":
                c = new cBOSSSP();
                ((cBOSSSP) c).chiLimits = new double[]{0};
                ((cBOSSSP) c).FCNN = true;
                ((cBOSSSP) c).FCNNlimit = 100;
                ((cBOSSSP) c).histogramIntersection = true;
                break;
            case "HI-SLimitFCNN-BcS-BOSS":
                c = new cBOSSSP();
                ((cBOSSSP) c).chiLimits = new double[]{0};
                ((cBOSSSP) c).FCNN = true;
                ((cBOSSSP) c).FCNNsoftlimit = 100;
                ((cBOSSSP) c).histogramIntersection = true;
                break;
            case "HI-CompFCNN-BcS-BOSS":
                c = new cBOSSSP();
                ((cBOSSSP) c).chiLimits = new double[]{0};
                ((cBOSSSP) c).FCNN = true;
                ((cBOSSSP) c).FCNNcomp = true;
                ((cBOSSSP) c).histogramIntersection = true;
                break;

            case "IGB-BcS-BOSS":
                c = new cBOSSSP();
                ((cBOSSSP) c).chiLimits = new double[]{0};
                ((cBOSSSP) c).useIGB = new boolean[]{true};
                break;
            case "Anova-BcS-BOSS":
                c = new cBOSSSP();
                ((cBOSSSP) c).chiLimits = new double[]{0};
                ((cBOSSSP) c).useAnova = new boolean[]{true};
                break;
            case "MFT-BcS-BOSS":
                c = new cBOSSSP();
                ((cBOSSSP) c).chiLimits = new double[]{0};
                ((cBOSSSP) c).newMFT = true;
                break;
            case "DFT-BcS-BOSS":
                c = new cBOSSSP();
                ((cBOSSSP) c).chiLimits = new double[]{0};
                ((cBOSSSP) c).newDFT = true;
                break;
            case "AnovaMFT-BcS-BOSS":
                c = new cBOSSSP();
                ((cBOSSSP) c).chiLimits = new double[]{0};
                ((cBOSSSP) c).useAnova = new boolean[]{true};
                ((cBOSSSP) c).newMFT = true;
                break;
            case "AnovaIGB-BcS-BOSS":
                c = new cBOSSSP();
                ((cBOSSSP) c).chiLimits = new double[]{0};
                ((cBOSSSP) c).useAnova = new boolean[]{true};
                ((cBOSSSP) c).useIGB = new boolean[]{true};
                break;

            case "pIGB-BcS-BOSS":
                c = new cBOSSSP();
                ((cBOSSSP) c).chiLimits = new double[]{0};
                ((cBOSSSP) c).useIGB = new boolean[]{true, false};
                break;
            case "pAnova-BcS-BOSS":
                c = new cBOSSSP();
                ((cBOSSSP) c).chiLimits = new double[]{0};
                ((cBOSSSP) c).useAnova = new boolean[]{true, false};
                break;
            case "pAnovaIGB-BcS-BOSS":
                c = new cBOSSSP();
                ((cBOSSSP) c).chiLimits = new double[]{0};
                ((cBOSSSP) c).useAnova = new boolean[]{true, false};
                ((cBOSSSP) c).useIGB = new boolean[]{true, false};
                break;

            case "HI-SLimitFCNN-pIGB-BcS-BOSS":
                c = new cBOSSSP();
                ((cBOSSSP) c).chiLimits = new double[]{0};
                ((cBOSSSP) c).useIGB = new boolean[]{true, false};
                ((cBOSSSP) c).FCNN = true;
                ((cBOSSSP) c).FCNNsoftlimit = 100;
                ((cBOSSSP) c).histogramIntersection = true;

                //((cBOSSSP) c).bigrams = true;
                //((cBOSSSP) c).useAnova = new boolean[]{true, false};
                //((cBOSSSP) c).tuneWeight = true;
                break;
            case "HI-pIGB-BcS-BOSS":
                c = new cBOSSSP();
                ((cBOSSSP) c).chiLimits = new double[]{0};
                ((cBOSSSP) c).useIGB = new boolean[]{true, false};
                ((cBOSSSP) c).histogramIntersection = true;
                break;
            case "HI-100pIGB-BcS-BOSS":
                c = new cBOSSSP();
                ((cBOSSSP) c).chiLimits = new double[]{0};
                ((cBOSSSP) c).useIGB = new boolean[]{true, false};
                ((cBOSSSP) c).histogramIntersection = true;
                ((cBOSSSP) c).initialRandomParameters = 100;
                break;
            case "HI-100pAnovaIGB-BcS-BOSS":
                c = new cBOSSSP();
                ((cBOSSSP) c).chiLimits = new double[]{0};
                ((cBOSSSP) c).useIGB = new boolean[]{true, false};
                ((cBOSSSP) c).useAnova = new boolean[]{true, false};
                ((cBOSSSP) c).histogramIntersection = true;
                ((cBOSSSP) c).initialRandomParameters = 100;
                break;

            case "HI-Bigram-pIGB-BcS-BOSS":
                c = new cBOSSSP();
                ((cBOSSSP) c).chiLimits = new double[]{0};
                ((cBOSSSP) c).useIGB = new boolean[]{true, false};
                ((cBOSSSP) c).histogramIntersection = true;
                ((cBOSSSP) c).bigrams = new boolean[]{true};
                break;
            case "HI-nrBigram-pIGB-BcS-BOSS":
                c = new cBOSSSP();
                ((cBOSSSP) c).chiLimits = new double[]{0};
                ((cBOSSSP) c).useIGB = new boolean[]{true, false};
                ((cBOSSSP) c).histogramIntersection = true;
                ((cBOSSSP) c).bigrams = new boolean[]{true};
                ((cBOSSSP) c).numerosityReduction = false;
                break;
            case "HI-nr-pIGB-BcS-BOSS":
                c = new cBOSSSP();
                ((cBOSSSP) c).chiLimits = new double[]{0};
                ((cBOSSSP) c).useIGB = new boolean[]{true, false};
                ((cBOSSSP) c).histogramIntersection = true;
                ((cBOSSSP) c).numerosityReduction = false;
                break;
            case "HI-TuneWeight-pIGB-BcS-BOSS":
                c = new cBOSSSP();
                ((cBOSSSP) c).chiLimits = new double[]{0};
                ((cBOSSSP) c).useIGB = new boolean[]{true, false};
                ((cBOSSSP) c).histogramIntersection = true;
                ((cBOSSSP) c).tuneWeight = true;
                break;

            case "HI-pIGB-cS-BOSS":
                c = new cBOSSSP();
                ((cBOSSSP) c).chiLimits = new double[]{0};
                ((cBOSSSP) c).useIGB = new boolean[]{true, false};
                ((cBOSSSP) c).histogramIntersection = true;
                ((cBOSSSP) c).setBayesianParameterSelection(false);
                break;
            case "HI-pIGB-750cS-BOSS":
                c = new cBOSSSP();
                ((cBOSSSP) c).chiLimits = new double[]{0};
                ((cBOSSSP) c).useIGB = new boolean[]{true, false};
                ((cBOSSSP) c).histogramIntersection = true;
                ((cBOSSSP) c).setBayesianParameterSelection(false);
                ((cBOSSSP) c).setEnsembleSize(750);
                break;
            case "HI-BIpIGB-cS-BOSS":
                c = new cBOSSSP();
                ((cBOSSSP) c).chiLimits = new double[]{0};
                ((cBOSSSP) c).useIGB = new boolean[]{true, false};
                ((cBOSSSP) c).histogramIntersection = true;
                ((cBOSSSP) c).setBayesianParameterSelection(false);
                ((cBOSSSP) c).bigrams = new boolean[]{true};
                break;
            case "HI-BIpIGB-750cS-BOSS":
                c = new cBOSSSP();
                ((cBOSSSP) c).chiLimits = new double[]{0};
                ((cBOSSSP) c).useIGB = new boolean[]{true, false};
                ((cBOSSSP) c).histogramIntersection = true;
                ((cBOSSSP) c).setBayesianParameterSelection(false);
                ((cBOSSSP) c).setEnsembleSize(750);
                ((cBOSSSP) c).bigrams = new boolean[]{true};
                break;

            case "HI-Bigram-pIGB-Bc-BOSS":
                c = new cBOSSSP();
                ((cBOSSSP) c).chiLimits = new double[]{0};
                ((cBOSSSP) c).useIGB = new boolean[]{true, false};
                ((cBOSSSP) c).histogramIntersection = true;
                ((cBOSSSP) c).bigrams = new boolean[]{true};
                ((cBOSSSP) c).levels = new Integer[]{1};
                break;
            case "Bigram-pIGB-BcS-BOSS":
                c = new cBOSSSP();
                ((cBOSSSP) c).chiLimits = new double[]{0};
                ((cBOSSSP) c).useIGB = new boolean[]{true, false};
                ((cBOSSSP) c).bigrams = new boolean[]{true};
                break;

            case "HI-Cutoff-Bigram-pIGB-BcS-BOSS":
                c = new cBOSSSP();
                ((cBOSSSP) c).chiLimits = new double[]{0};
                ((cBOSSSP) c).useIGB = new boolean[]{true, false};
                ((cBOSSSP) c).histogramIntersection = true;
                ((cBOSSSP) c).bigrams = new boolean[]{true};
                ((cBOSSSP) c).setCutoff(true);
                break;
            case "HI-tp80-Bigram-pIGB-BcS-BOSS":
                c = new cBOSSSP();
                ((cBOSSSP) c).chiLimits = new double[]{0};
                ((cBOSSSP) c).useIGB = new boolean[]{true, false};
                ((cBOSSSP) c).histogramIntersection = true;
                ((cBOSSSP) c).bigrams = new boolean[]{true};
                ((cBOSSSP) c).setTrainProportion(0.8);
                break;
            case "HI-tp60-Bigram-pIGB-BcS-BOSS":
                c = new cBOSSSP();
                ((cBOSSSP) c).chiLimits = new double[]{0};
                ((cBOSSSP) c).useIGB = new boolean[]{true, false};
                ((cBOSSSP) c).histogramIntersection = true;
                ((cBOSSSP) c).bigrams = new boolean[]{true};
                ((cBOSSSP) c).setTrainProportion(0.6);
                break;
            case "HI-tp50-Bigram-pIGB-BcS-BOSS":
                c = new cBOSSSP();
                ((cBOSSSP) c).chiLimits = new double[]{0};
                ((cBOSSSP) c).useIGB = new boolean[]{true, false};
                ((cBOSSSP) c).histogramIntersection = true;
                ((cBOSSSP) c).bigrams = new boolean[]{true};
                ((cBOSSSP) c).setTrainProportion(0.5);
                break;
            case "HI-l4-Bigram-pIGB-BcS-BOSS":
                c = new cBOSSSP();
                ((cBOSSSP) c).chiLimits = new double[]{0};
                ((cBOSSSP) c).useIGB = new boolean[]{true, false};
                ((cBOSSSP) c).histogramIntersection = true;
                ((cBOSSSP) c).bigrams = new boolean[]{true};
                ((cBOSSSP) c).levels = new Integer[]{1,2,3,4};
                break;
            case "HI-pBigram-pIGB-BcS-BOSS":
                c = new cBOSSSP();
                ((cBOSSSP) c).chiLimits = new double[]{0};
                ((cBOSSSP) c).useIGB = new boolean[]{true, false};
                ((cBOSSSP) c).histogramIntersection = true;
                ((cBOSSSP) c).bigrams = new boolean[]{true, false};
                break;

            case "HI-fs-pIGB-BcS-BOSS":
                c = new cBOSSSP();
                ((cBOSSSP) c).chiLimits = new double[]{0.9};
                ((cBOSSSP) c).featureSelection = true;
                ((cBOSSSP) c).useIGB = new boolean[]{true, false};
                ((cBOSSSP) c).histogramIntersection = true;
                ((cBOSSSP) c).bigrams = new boolean[]{true};
                break;
            case "HI-500s-pIGB-BcS-BOSS":
                c = new cBOSSSP();
                ((cBOSSSP) c).chiLimits = new double[]{0};
                ((cBOSSSP) c).useIGB = new boolean[]{true, false};
                ((cBOSSSP) c).histogramIntersection = true;
                ((cBOSSSP) c).bigrams = new boolean[]{true};
                ((cBOSSSP) c).setEnsembleSize(500);
                break;
            case "HI-100m-pIGB-BcS-BOSS":
                c = new cBOSSSP();
                ((cBOSSSP) c).chiLimits = new double[]{0};
                ((cBOSSSP) c).useIGB = new boolean[]{true, false};
                ((cBOSSSP) c).histogramIntersection = true;
                ((cBOSSSP) c).bigrams = new boolean[]{true};
                ((cBOSSSP) c).setMaxEnsembleSize(100);
                break;

            case "HI-tp100-Bigram-pIGB-BcS-BOSS":
                c = new cBOSSSP();
                ((cBOSSSP) c).chiLimits = new double[]{0};
                ((cBOSSSP) c).useIGB = new boolean[]{true, false};
                ((cBOSSSP) c).histogramIntersection = true;
                ((cBOSSSP) c).bigrams = new boolean[]{true};
                ((cBOSSSP) c).setReduceTrainInstances(false);
                break;

            case "TDE-1H":
                c = new cBOSSSP();
                ((cBOSSSP) c).chiLimits = new double[]{0};
                ((cBOSSSP) c).useIGB = new boolean[]{true, false};
                ((cBOSSSP) c).histogramIntersection = true;
                ((cBOSSSP) c).bigrams = new boolean[]{true};
                ((cBOSSSP) c).setMaxEnsembleSize(100);
                ((cBOSSSP) c).setTrainTimeLimit(TimeUnit.HOURS, 1);
                break;
            case "TDE-4H":
                c = new cBOSSSP();
                ((cBOSSSP) c).chiLimits = new double[]{0};
                ((cBOSSSP) c).useIGB = new boolean[]{true, false};
                ((cBOSSSP) c).histogramIntersection = true;
                ((cBOSSSP) c).bigrams = new boolean[]{true};
                ((cBOSSSP) c).setMaxEnsembleSize(100);
                ((cBOSSSP) c).setTrainTimeLimit(TimeUnit.HOURS, 4);
                break;
            case "TDE-12H":
                c = new cBOSSSP();
                ((cBOSSSP) c).chiLimits = new double[]{0};
                ((cBOSSSP) c).useIGB = new boolean[]{true, false};
                ((cBOSSSP) c).histogramIntersection = true;
                ((cBOSSSP) c).bigrams = new boolean[]{true};
                ((cBOSSSP) c).setMaxEnsembleSize(100);
                ((cBOSSSP) c).setTrainTimeLimit(TimeUnit.HOURS, 12);
                break;

            case "TDE-Cutoff70":
                c = new cBOSSSP();
                ((cBOSSSP) c).chiLimits = new double[]{0};
                ((cBOSSSP) c).useIGB = new boolean[]{true, false};
                ((cBOSSSP) c).histogramIntersection = true;
                ((cBOSSSP) c).bigrams = new boolean[]{true};
                ((cBOSSSP) c).setMaxEnsembleSize(100);
                ((cBOSSSP) c).setCutoff(true);
                ((cBOSSSP) c).correctThreshold = 0.7;
                break;
            case "TDE-WordLength":
                c = new cBOSSSP();
                ((cBOSSSP) c).chiLimits = new double[]{0};
                ((cBOSSSP) c).useIGB = new boolean[]{true, false};
                ((cBOSSSP) c).histogramIntersection = true;
                ((cBOSSSP) c).bigrams = new boolean[]{true};
                ((cBOSSSP) c).setMaxEnsembleSize(100);
                ((cBOSSSP) c).wordLengths = new int[]{16, 14, 12, 10, 8, 6, 4};
                break;

            default:
                System.out.println("Unknown dictionary based classifier "+classifier+" should not be able to get here ");
                System.out.println("There is a mismatch between array dictionary and the switch statement ");
                throw new UnsupportedOperationException("Unknown dictionary based  classifier "+classifier+" should not be able to get here."
                        + "There is a mismatch between array dictionary and the switch statement ");

        }
        return c;
    }

    /**
    * INTERVAL BASED: classifiers that form multiple intervals over series and summarise
    */
    public static String[] interval= {"LPS","TSF"};
    public static HashSet<String> intervalBased=new HashSet<String>( Arrays.asList(interval));
    private static Classifier setIntervalBased(Experiments.ExperimentalArguments exp){
        String classifier=exp.classifierName;
        Classifier c;
        int fold=exp.foldId;
        switch(classifier) {
            case "LPS":
                c=new LPS();
                break;
            case "TSF":
                c=new TSF();
                break;
            default:
                System.out.println("Unknown interval based classifier "+classifier+" should not be able to get here ");
                System.out.println("There is a mismatch between array interval and the switch statement ");
                throw new UnsupportedOperationException("Unknown interval based  classifier "+classifier+" should not be able to get here."
                        + "There is a mismatch between array interval and the switch statement ");

        }
        return c;
    }

    /**
     * FREQUENCY BASED: Classifiers that work in the spectral/frequency domain
     */
    public static String[] frequency= {"RISE"};
    public static HashSet<String> frequencyBased=new HashSet<String>( Arrays.asList(frequency));
    private static Classifier setFrequencyBased(Experiments.ExperimentalArguments exp){
        String classifier=exp.classifierName;
        Classifier c;
        int fold=exp.foldId;
        switch(classifier) {
            case "RISE":
                c=new RISE();
                break;
            default:
                System.out.println("Unknown interval based classifier, should not be able to get here ");
                System.out.println("There is a mismatch between array interval and the switch statement ");
                throw new UnsupportedOperationException("Unknown interval based  classifier, should not be able to get here "
                        + "There is a mismatch between array interval and the switch statement ");

        }
        return c;
    }

    /**
     * SHAPELET BASED: Classifiers that use shapelets in some way.
     */
    public static String[] shapelet= {"FastShapelets","LearnShapelets","ShapeletTransformClassifier","ShapeletTreeClassifier"};
    public static HashSet<String> shapeletBased=new HashSet<String>( Arrays.asList(shapelet));
    private static Classifier setShapeletBased(Experiments.ExperimentalArguments exp){
        String classifier=exp.classifierName;
        Classifier c;
        int fold=exp.foldId;
        switch(classifier) {
            case "LearnShapelets":
                c=new LearnShapelets();
                break;
            case "FastShapelets":
                c=new FastShapelets();
                break;
            case "ShapeletTransformClassifier":
                c=new ShapeletTransformClassifier();
                break;
            case "ShapeletTreeClassifier":
                c=new ShapeletTree();
                break;
           default:
                System.out.println("Unknown interval based classifier, should not be able to get here ");
                System.out.println("There is a mismatch between array interval and the switch statement ");
                throw new UnsupportedOperationException("Unknown interval based  classifier, should not be able to get here "
                        + "There is a mismatch between array interval and the switch statement ");

        }
        return c;
    }

    /**
     * HYBRIDS: Classifiers that combine two or more of the above approaches
     */
    public static String[] hybrids= {"HiveCote","FlatCote","TSCHIEF"};
    public static HashSet<String> hybridBased=new HashSet<String>( Arrays.asList(hybrids));
    private static Classifier setHybridBased(Experiments.ExperimentalArguments exp){
        String classifier=exp.classifierName;
        Classifier c;
        int fold=exp.foldId;
        switch(classifier) {
            case "FlatCote":
                c=new FlatCote();
                break;
            case "HiveCote":
                c=new HiveCote();
                ((HiveCote)c).setContract(48);
                break;
            case "TSCHIEF":
                c=new TSCHIEFWrapper();
                ((TSCHIEFWrapper)c).setSeed(fold);
                break;
            default:
                System.out.println("Unknown hybrid based classifier, should not be able to get here ");
                System.out.println("There is a mismatch between array hybrids and the switch statement ");
                throw new UnsupportedOperationException("Unknown hybrid based  classifier, should not be able to get here "
                        + "There is a mismatch between array hybrids and the switch statement ");

        }
        return c;
    }

    /**
     * MULTIVARIATE time series classifiers, all in one list for now
     */
    public static String[] allMultivariate={"Shapelet_I","Shapelet_D","Shapelet_Indep","ED_I","DTW_I","DTW_D","DTW_A"};//Not enough to classify yet
    public static HashSet<String> multivariateBased=new HashSet<String>( Arrays.asList(allMultivariate));
    private static Classifier setMultivariate(Experiments.ExperimentalArguments exp){
        String classifier=exp.classifierName;
        Classifier c;
        int fold=exp.foldId;
        switch(classifier) {
            case "Shapelet_I": case "Shapelet_D": case  "Shapelet_Indep"://Multivariate version 1
                c=new MultivariateShapeletTransformClassifier();
//Default to 1 day max run: could do this better
                ((MultivariateShapeletTransformClassifier)c).setOneDayLimit();
                ((MultivariateShapeletTransformClassifier)c).setSeed(fold);
                ((MultivariateShapeletTransformClassifier)c).setTransformType(classifier);
                break;
            case "ED_I":
                c=new NN_ED_I();
                break;
            case "DTW_I":
                c=new NN_DTW_I();
                break;
            case "DTW_D":
                c=new NN_DTW_D();
                break;
            case "DTW_A":
                c=new NN_DTW_A();
                break;
            default:
                System.out.println("Unknown multivariate classifier, should not be able to get here ");
                System.out.println("There is a mismatch between multivariateBased and the switch statement ");
                throw new UnsupportedOperationException("Unknown multivariate classifier, should not be able to get here "
                        + "There is a mismatch between multivariateBased and the switch statement ");
        }
        return c;
    }


    /**
     * STANDARD classifiers such as random forest etc
     */
    public static String[] standard= {
        "XGBoostMultiThreaded","XGBoost","SmallTunedXGBoost","RandF","RotF", "PLSNominalClassifier","BayesNet","ED","C45",
            "SVML","SVMQ","SVMRBF","MLP","Logistic","CAWPE","NN"};
    public static HashSet<String> standardClassifiers=new HashSet<String>( Arrays.asList(standard));
    private static Classifier setStandardClassifiers(Experiments.ExperimentalArguments exp){
        String classifier=exp.classifierName;
        int fold=exp.foldId;
        Classifier c;
        switch(classifier) {
//TIME DOMAIN CLASSIFIERS
            case "XGBoostMultiThreaded":
                c = new TunedXGBoost();
                break;
            case "XGBoost":
                c = new TunedXGBoost();
                ((TunedXGBoost)c).setRunSingleThreaded(true);
                break;
            case "SmallTunedXGBoost":
                c = new TunedXGBoost();
                ((TunedXGBoost)c).setRunSingleThreaded(true);
                ((TunedXGBoost)c).setSmallParaSearchSpace_64paras();
                break;
            case "RandF":
                RandomForest r=new RandomForest();
                r.setNumTrees(500);
                c = r;
                break;
            case "RotF":
                RotationForest rf=new RotationForest();
                rf.setNumIterations(200);
                c = rf;
                break;
            case "PLSNominalClassifier":
                c = new PLSNominalClassifier();
                break;
            case "BayesNet":
                c = new BayesNet();
                break;
            case "ED":
                c= KNNLOOCV.FACTORY.ED_1NN_V1.build();;
                break;
            case "C45":
                c=new J48();
                break;
            case "NB":
                c=new NaiveBayes();
                break;
            case "SVML":
                c=new SMO();
                PolyKernel p=new PolyKernel();
                p.setExponent(1);
                ((SMO)c).setKernel(p);
                ((SMO)c).setRandomSeed(fold);
                ((SMO)c).setBuildLogisticModels(true);
                break;
            case "SVMQ":
                c=new SMO();
                PolyKernel poly=new PolyKernel();
                poly.setExponent(2);
                ((SMO)c).setKernel(poly);
                ((SMO)c).setRandomSeed(fold);
                ((SMO)c).setBuildLogisticModels(true);
                break;
            case "SVMRBF":
                c=new SMO();
                RBFKernel rbf=new RBFKernel();
                rbf.setGamma(0.5);
                ((SMO)c).setC(5);
                ((SMO)c).setKernel(rbf);
                ((SMO)c).setRandomSeed(fold);
                ((SMO)c).setBuildLogisticModels(true);

                break;
            case "BN":
                c=new BayesNet();
                break;
            case "MLP":
                c=new MultilayerPerceptron();
                break;
            case "Logistic":
                c= new Logistic();
                break;
            case "CAWPE":
                c=new CAWPE();
                break;
            case "NN":
                kNN k=new kNN(100);
                k.setCrossValidate(true);
                k.normalise(false);
                k.setDistanceFunction(new EuclideanDistance());
                return k;
            default:
                System.out.println("Unknown standard classifier "+classifier+" should not be able to get here ");
                System.out.println("There is a mismatch between otherClassifiers and the switch statement ");
                throw new UnsupportedOperationException("Unknown standard classifier "+classifier+" should not be able to get here "
                        + "There is a mismatch between otherClassifiers and the switch statement ");
        }
        return c;
    }

    /**
     * BESPOKE classifiers for particular set ups. Use if you want some special configuration/pipeline
     * not encapsulated within a single classifier      */
    public static String[] bespoke= {"HC-V2NoRise","HIVE-COTEV2","HIVE-COTE2","HC-TDE2","HC-WEASEL2","HIVE-COTE","HC-TDE","HC-WEASEL","HC-BcSBOSS","HC-cSBOSS","TunedHIVE-COTE"};
    public static HashSet<String> bespokeClassifiers=new HashSet<String>( Arrays.asList(bespoke));
    private static Classifier setBespokeClassifiers(Experiments.ExperimentalArguments exp){
        String classifier=exp.classifierName,resultsPath="",dataset="";
        int fold=exp.foldId;
        Classifier c;
        boolean canLoadFromFile=true;
        if(exp.resultsWriteLocation==null || exp.datasetName==null)
            canLoadFromFile=false;
        else{
            resultsPath=exp.resultsWriteLocation;
            dataset=exp.datasetName;
        }
        switch(classifier) {
            case "HC-V2NoRise":
                if(canLoadFromFile){
                    String[] cls={"CIF","TED","STC","PF"};//RotF for ST
                    c=new HIVE_COTE();
                    ((HIVE_COTE)c).setFillMissingDistsWithOneHotVectors(true);
                    ((HIVE_COTE)c).setSeed(fold);
                    ((HIVE_COTE)c).setBuildIndividualsFromResultsFiles(true);
                    ((HIVE_COTE)c).setResultsFileLocationParameters(resultsPath, dataset, fold);
                    ((HIVE_COTE)c).setClassifiersNamesForFileRead(cls);
                }
                else
                    throw new UnsupportedOperationException("ERROR: currently only loading from file for CAWPE and no results file path has been set. "
                            + "Call setClassifier with an ExperimentalArguments object exp with exp.resultsWriteLocation (contains component classifier results) and exp.datasetName set");
                break;
            case "HIVE-COTEV2":
                if(canLoadFromFile){
                    String[] cls={"CIF","TED","RISE","STC","PF"};//RotF for ST
                    c=new HIVE_COTE();
                    ((HIVE_COTE)c).setFillMissingDistsWithOneHotVectors(true);
                    ((HIVE_COTE)c).setSeed(fold);
                    ((HIVE_COTE)c).setBuildIndividualsFromResultsFiles(true);
                    ((HIVE_COTE)c).setResultsFileLocationParameters(resultsPath, dataset, fold);
                    ((HIVE_COTE)c).setClassifiersNamesForFileRead(cls);
                }
                else
                    throw new UnsupportedOperationException("ERROR: currently only loading from file for CAWPE and no results file path has been set. "
                            + "Call setClassifier with an ExperimentalArguments object exp with exp.resultsWriteLocation (contains component classifier results) and exp.datasetName set");
                break;
            case "HC-TDE2":
                if(canLoadFromFile){
                    String[] cls={"TSF","TDE","RISE","STC"};//RotF for ST
                    c=new HIVE_COTE();
                    ((HIVE_COTE)c).setFillMissingDistsWithOneHotVectors(true);
                    ((HIVE_COTE)c).setSeed(fold);
                    ((HIVE_COTE)c).setBuildIndividualsFromResultsFiles(true);
                    ((HIVE_COTE)c).setResultsFileLocationParameters(resultsPath, dataset, fold);
                    ((HIVE_COTE)c).setClassifiersNamesForFileRead(cls);
                }
                else
                    throw new UnsupportedOperationException("ERROR: currently only loading from file for CAWPE and no results file path has been set. "
                            + "Call setClassifier with an ExperimentalArguments object exp with exp.resultsWriteLocation (contains component classifier results) and exp.datasetName set");
                break;
            case "HC-WEASEL2":
                if(canLoadFromFile){
                    String[] cls={"TSF","WEASEL","RISE","STC"};//RotF for ST
                    c=new HIVE_COTE();
                    ((HIVE_COTE)c).setFillMissingDistsWithOneHotVectors(true);
                    ((HIVE_COTE)c).setSeed(fold);
                    ((HIVE_COTE)c).setBuildIndividualsFromResultsFiles(true);
                    ((HIVE_COTE)c).setResultsFileLocationParameters(resultsPath, dataset, fold);
                    ((HIVE_COTE)c).setClassifiersNamesForFileRead(cls);
                }
                else
                    throw new UnsupportedOperationException("ERROR: currently only loading from file for CAWPE and no results file path has been set. "
                            + "Call setClassifier with an ExperimentalArguments object exp with exp.resultsWriteLocation (contains component classifier results) and exp.datasetName set");
                break;
            case "HC-BcSBOSS":
                if(canLoadFromFile){
                    String[] cls={"TSF","BcS-BOSS","RISE","STC","EE"};//RotF for ST
                    c=new HIVE_COTE();
                    ((HIVE_COTE)c).setFillMissingDistsWithOneHotVectors(true);
                    ((HIVE_COTE)c).setSeed(fold);
                    ((HIVE_COTE)c).setBuildIndividualsFromResultsFiles(true);
                    ((HIVE_COTE)c).setResultsFileLocationParameters(resultsPath, dataset, fold);
                    ((HIVE_COTE)c).setClassifiersNamesForFileRead(cls);
                }
                else
                    throw new UnsupportedOperationException("ERROR: currently only loading from file for CAWPE and no results file path has been set. "
                            + "Call setClassifier with an ExperimentalArguments object exp with exp.resultsWriteLocation (contains component classifier results) and exp.datasetName set");
                break;
            case "HC-cSBOSS":
                if(canLoadFromFile){
                    String[] cls={"TSF","cS-BOSS","RISE","STC","EE"};//RotF for ST
                    c=new HIVE_COTE();
                    ((HIVE_COTE)c).setFillMissingDistsWithOneHotVectors(true);
                    ((HIVE_COTE)c).setSeed(fold);
                    ((HIVE_COTE)c).setBuildIndividualsFromResultsFiles(true);
                    ((HIVE_COTE)c).setResultsFileLocationParameters(resultsPath, dataset, fold);
                    ((HIVE_COTE)c).setClassifiersNamesForFileRead(cls);
                }
                else
                    throw new UnsupportedOperationException("ERROR: currently only loading from file for CAWPE and no results file path has been set. "
                            + "Call setClassifier with an ExperimentalArguments object exp with exp.resultsWriteLocation (contains component classifier results) and exp.datasetName set");
                break;
            case "HIVE-COTE2":
                if(canLoadFromFile){
                    String[] cls={"TSF","BOSS","RISE","STC"};//RotF for ST
                    c=new HIVE_COTE();
                    ((HIVE_COTE)c).setFillMissingDistsWithOneHotVectors(true);
                    ((HIVE_COTE)c).setSeed(fold);
                    ((HIVE_COTE)c).setBuildIndividualsFromResultsFiles(true);
                    ((HIVE_COTE)c).setResultsFileLocationParameters(resultsPath, dataset, fold);
                    ((HIVE_COTE)c).setClassifiersNamesForFileRead(cls);
                }
                else
                    throw new UnsupportedOperationException("ERROR: currently only loading from file for CAWPE and no results file path has been set. "
                            + "Call setClassifier with an ExperimentalArguments object exp with exp.resultsWriteLocation (contains component classifier results) and exp.datasetName set");
                break;
            case "HIVE-COTE":
                if(canLoadFromFile){
                    String[] cls={"TSF","BOSS","RISE","STC","EE"};//RotF for ST
                    c=new HIVE_COTE();
                    ((HIVE_COTE)c).setFillMissingDistsWithOneHotVectors(true);
                    ((HIVE_COTE)c).setSeed(fold);
                    ((HIVE_COTE)c).setBuildIndividualsFromResultsFiles(true);
                    ((HIVE_COTE)c).setResultsFileLocationParameters(resultsPath, dataset, fold);
                    ((HIVE_COTE)c).setClassifiersNamesForFileRead(cls);
                }
                else
                    throw new UnsupportedOperationException("ERROR: currently only loading from file for CAWPE and no results file path has been set. "
                            + "Call setClassifier with an ExperimentalArguments object exp with exp.resultsWriteLocation (contains component classifier results) and exp.datasetName set");
                break;
            case "HC-WEASEL":
                if(canLoadFromFile){
                    String[] cls={"TSF","WEASEL","RISE","STC","EE"};//RotF for ST
                    c=new HIVE_COTE();
                    ((HIVE_COTE)c).setFillMissingDistsWithOneHotVectors(true);
                    ((HIVE_COTE)c).setSeed(fold);
                    ((HIVE_COTE)c).setBuildIndividualsFromResultsFiles(true);
                    ((HIVE_COTE)c).setResultsFileLocationParameters(resultsPath, dataset, fold);
                    ((HIVE_COTE)c).setClassifiersNamesForFileRead(cls);
                }
                else
                    throw new UnsupportedOperationException("ERROR: currently only loading from file for CAWPE and no results file path has been set. "
                            + "Call setClassifier with an ExperimentalArguments object exp with exp.resultsWriteLocation (contains component classifier results) and exp.datasetName set");
                break;
            case "HC-TDE":
                if(canLoadFromFile){
                    String[] cls={"TSF","TDE","RISE","STC","EE"};//RotF for ST
                    c=new HIVE_COTE();
                    ((HIVE_COTE)c).setFillMissingDistsWithOneHotVectors(true);
                    ((HIVE_COTE)c).setSeed(fold);
                    ((HIVE_COTE)c).setBuildIndividualsFromResultsFiles(true);
                    ((HIVE_COTE)c).setResultsFileLocationParameters(resultsPath, dataset, fold);
                    ((HIVE_COTE)c).setClassifiersNamesForFileRead(cls);
                }
                else
                    throw new UnsupportedOperationException("ERROR: currently only loading from file for CAWPE and no results file path has been set. "
                            + "Call setClassifier with an ExperimentalArguments object exp with exp.resultsWriteLocation (contains component classifier results) and exp.datasetName set");
                break;

            case "TunedHIVE-COTE":
                if(canLoadFromFile){
                    String[] cls={"TSF","BOSS","RISE","STC","EE"};//RotF for ST
                    HIVE_COTE hc=new HIVE_COTE();
                    hc.setFillMissingDistsWithOneHotVectors(true);
                    hc.setSeed(fold);
                    hc.setBuildIndividualsFromResultsFiles(true);
                    hc.setResultsFileLocationParameters(resultsPath, dataset, fold);
                    hc.setClassifiersNamesForFileRead(cls);
                    TunedClassifier tuner=new TunedClassifier();
                    tuner.setClassifier(hc);
                    ParameterSpace pc=new ParameterSpace();
                    double[] alphaVals={1,2,3,4,5,6,7,8,9,10};
                    pc.addParameter("a",alphaVals);
                    tuner.setParameterSpace(pc);
                    c=tuner;
                }
                else
                    throw new UnsupportedOperationException("ERROR: currently only loading from file for CAWPE and no results file path has been set. "
                            + "Call setClassifier with an ExperimentalArguments object exp with exp.resultsWriteLocation (contains component classifier results) and exp.datasetName set");
                break;

            default:
                System.out.println("Unknown bespoke classifier, should not be able to get here ");
                System.out.println("There is a mismatch between bespokeClassifiers and the switch statement ");
                throw new UnsupportedOperationException("Unknown bespoke classifier, should not be able to get here "
                        + "There is a mismatch between bespokeClassifiers and the switch statement ");
        }
        return c;
    }

    /**
     *
     * setClassifier, which takes the experimental
     * arguments themselves and therefore the classifiers can take from them whatever they
     * need, e.g the dataset name, the fold id, separate checkpoint paths, etc.
     *
     * To take this idea further, to be honest each of the TSC-specific classifiers
     * could/should have a constructor and/or factory that builds the classifier
     * from the experimental args.
     *
     * previous usage was setClassifier(String classifier name, int fold).
     * this can be reproduced with setClassifierClassic below.
     *
     */
    public static Classifier setClassifier(Experiments.ExperimentalArguments exp){
        String classifier=exp.classifierName;
        Classifier c = null;
        if(distanceBased.contains(classifier))
            c=setDistanceBased(exp);
        else if(dictionaryBased.contains(classifier))
            c=setDictionaryBased(exp);
        else if(intervalBased.contains(classifier))
            c=setIntervalBased(exp);
        else if(frequencyBased.contains(classifier))
            c=setFrequencyBased(exp);
        else if(shapeletBased.contains(classifier))
            c=setShapeletBased(exp);
        else if(hybridBased.contains(classifier))
            c=setHybridBased(exp);
        else if(multivariateBased.contains(classifier))
            c=setMultivariate(exp);
        else if(standardClassifiers.contains(classifier))
            c=setStandardClassifiers(exp);
        else if(bespokeClassifiers.contains(classifier))
            c=setBespokeClassifiers(exp);
        else{
            System.out.println("Unknown classifier "+classifier+" it is not in any of the sublists ");
            throw new UnsupportedOperationException("Unknown classifier "+classifier+" it is not in any of the sublists on ClassifierLists ");
        }
        if(c instanceof Randomizable)
            ((Randomizable)c).setSeed(exp.foldId);
        return c;
    }

    /**
     * This method redproduces the old usage exactly as it was in old experiments.java.
     * If you try build any classifier that uses any experimental info other than
     * exp.classifierName or exp.foldID, an exception will be thrown.
     * In particular, any classifier that needs access to the results from others
     * e.g. CAWPEFROMFILE, will throw an UnsupportedOperationException if you try use it like this.
     *      * @param classifier
     * @param fold
     * @return
     */
    public static Classifier setClassifierClassic(String classifier, int fold){
        Experiments.ExperimentalArguments exp=new ExperimentalArguments();
        exp.classifierName=classifier;
        exp.foldId=fold;
        return setClassifier(exp);
    }






    public static void main(String[] args) throws Exception {
        System.out.println("Testing set classifier by running through the list in ClassifierLists.allUnivariate and " +
                "ClassifierLists.allMultivariate");

        for(String str:allUnivariate){
            System.out.println("Initialising "+str);
            Classifier c= setClassifierClassic(str,0);
            System.out.println("Returned classifier "+c.getClass().getSimpleName());
        }
        for(String str:allMultivariate){
            System.out.println("Initialising "+str);
            Classifier c= setClassifierClassic(str,0);
            System.out.println("Returned classifier "+c.getClass().getSimpleName());
        }
        for(String str:standard){
            System.out.println("Initialising "+str);
            Classifier c= setClassifierClassic(str,0);
            System.out.println("Returned classifier "+c.getClass().getSimpleName());
        }
        for(String str:bespoke){
            System.out.println("Initialising "+str);
            Classifier c= setClassifierClassic(str,0);
            System.out.println("Returned classifier "+c.getClass().getSimpleName());
        }


    }
}
