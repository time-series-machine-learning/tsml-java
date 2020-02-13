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


import experiments.Experiments.ExperimentalArguments;
import machine_learning.classifiers.ensembles.weightings.TrainAcc;
import tsml.classifiers.dictionary_based.*;
import tsml.classifiers.dictionary_based.boss_variants.BOSSC45;
import tsml.classifiers.dictionary_based.SpatialBOSS;
import tsml.classifiers.dictionary_based.boss_variants.BoTSWEnsemble;
import tsml.classifiers.distance_based.*;
import tsml.classifiers.frequency_based.cRISE;
import tsml.classifiers.hybrids.Catch22Classifier;
import tsml.classifiers.hybrids.FlatCote;
import tsml.classifiers.hybrids.HiveCote;
import tsml.classifiers.hybrids.TSCHIEFWrapper;
import tsml.classifiers.interval_based.C22IF;
import tsml.classifiers.interval_based.cTSF;
import tsml.classifiers.shapelet_based.ShapeletTransformClassifier;
import tsml.classifiers.shapelet_based.FastShapelets;
import tsml.classifiers.shapelet_based.LearnShapelets;
import tsml.classifiers.interval_based.TSF;
import tsml.classifiers.interval_based.LPS;
import tsml.classifiers.frequency_based.RISE;
import tsml.classifiers.multivariate.MultivariateShapeletTransformClassifier;
import tsml.classifiers.multivariate.NN_DTW_A;
import tsml.classifiers.multivariate.NN_DTW_D;
import tsml.classifiers.multivariate.NN_DTW_I;
import tsml.classifiers.multivariate.NN_ED_I;
import tsml.classifiers.distance_based.elastic_ensemble.DTW1NN;
import tsml.classifiers.distance_based.elastic_ensemble.ED1NN;
import tsml.classifiers.distance_based.elastic_ensemble.MSM1NN;
import tsml.classifiers.distance_based.elastic_ensemble.WDTW1NN;
import tsml.classifiers.shapelet_based.ShapeletTree;
import weka.classifiers.trees.RandomTree;
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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;

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
            "RISE","cRISE",
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
        "DTW","DTWCV","ApproxElasticEnsemble","ProximityForest","ElasticEnsemble","FastElasticEnsemble",
            "DD_DTW","DTD_C","NN_CID","MSM","TWE","WDTW"
//        , "CEE", "LEE", "DTWV1", "DTWCVV1"
    };
    public static HashSet<String> distanceBased=new HashSet<String>( Arrays.asList(distance));
    private static Classifier setDistanceBased(Experiments.ExperimentalArguments exp){
        String classifier=exp.classifierName;
        Classifier c;
        int fold=exp.foldId;
        switch(classifier) {
            case "DTW":
                c=new DTW1NN();
                ((DTW1NN )c).setWindow(1);
                break;
            case "DTWCV":
                c=new DTWCV();
                ((DTWCV)c).optimiseWindow(true);
                break;
            case "ApproxElasticEnsemble":
                c = new ApproxElasticEnsemble();
                break;
            case "ProximityForest":
                c = new ProximityForestWrapper();
                break;
            case "ElasticEnsemble":
                c=new ElasticEnsemble();
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
            case "MSM":
                c=new MSM1NN();
                break;
            case "TWE":
                c=new MSM1NN();
                break;
            case "WDTW":
                c=new WDTW1NN();
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
        "BOSS", "BOP", "SAXVSM", "SAX_1NN", "WEASEL", "cBOSS", "BOSSC45", "S-BOSS", "SpatialBOSS", "BoTSWEnsemble"};
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
    public static String[] interval= {"LPS","TSF","cTSF","C22IF-A","C22IF-B"};
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
            case "cTSF":
                c=new cTSF();
                break;
            case "C22IF-A":
                c=new C22IF();
                //((C22IF)c).setOutlierNorm(false);
                ((C22IF)c).setAttSubsampleSize(22);
                ((C22IF)c).setUseSummaryStats(false);
                ((C22IF)c).setBaseClassifier(new RandomTree());
                break;
            case "C22IF-B":
                c=new C22IF();
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
    public static String[] frequency= {"RISE","cRISE"};
    public static HashSet<String> frequencyBased=new HashSet<String>( Arrays.asList(frequency));
    private static Classifier setFrequencyBased(Experiments.ExperimentalArguments exp){
        String classifier=exp.classifierName;
        Classifier c;
        int fold=exp.foldId;
        switch(classifier) {
            case "RISE":
                c=new RISE();
                ((RISE) c).setTransforms("PS","ACF");
                break;
            case "cRISE":
                c=new cRISE();
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
    public static String[] hybrids= {"HiveCote","FlatCote","TSCHIEF","Catch22"};
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
            case "Catch22":
                c=new Catch22Classifier();
                RandomForest rf = new RandomForest();
                rf.setNumTrees(500);
                rf.setSeed(fold);
                ((Catch22Classifier)c).setClassifier(rf);
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
                c=new ED1NN();
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
    public static String[] bespoke= {"CAWPEPLUS","CAWPEFROMFILE","CawpeAsCote",
            "CAWPE_AS_COTE_NO_EE","HC-Standard","HC-Alpha1","HC-NewBOSS","HC-NoEE","HC-SB","HC-NoEE-SB","HC-WEASEL","HC-cBOSS", "HC-PF","HC-PF-SB","FullHC","FullHC-SB","FullHC-SB-PF","HC-Latest","HC-Catch22TSF"};
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
            case "CAWPEPLUS":
                c=new CAWPE();
                ((CAWPE)c).setupAdvancedSettings();
                break;
            case "CAWPEFROMFILE":
                if(canLoadFromFile){
                    String[] classifiers={"TSF","BOSS","RISE","STC","EE"};
                    c=new CAWPE();
                    ((CAWPE)c).setBuildIndividualsFromResultsFiles(true);
                    ((CAWPE)c).setResultsFileLocationParameters(resultsPath, dataset, fold);
                    ((CAWPE)c).setClassifiersNamesForFileRead(classifiers);
                }
                else
                    throw new UnsupportedOperationException("ERROR: Cannot load CAWPE from file since no results file path has been set. "
                            + "Call setClassifier with an ExperimentalArguments object exp with exp.resultsWriteLocation (contains component classifier results) and exp.datasetName set");
                break;
            case "CawpeAsCote":
                if(canLoadFromFile){
                    String[] cls={"TSF","BOSS","RISE","STC","EE"};//RotF for ST
                    c=new CAWPE();
                    ((CAWPE)c).setFillMissingDistsWithOneHotVectors(true);
                    ((CAWPE)c).setBuildIndividualsFromResultsFiles(true);
                    ((CAWPE)c).setResultsFileLocationParameters(resultsPath, dataset, fold);
                    ((CAWPE)c).setClassifiersNamesForFileRead(cls);
                }
                else
                    throw new UnsupportedOperationException("ERROR: currently only loading from file for CAWPE and no results file path has been set. "
                            + "Call setClassifier with an ExperimentalArguments object exp with exp.resultsWriteLocation (contains component classifier results) and exp.datasetName set");
                break;
            case "CAWPE_AS_COTE_NO_EE":
                if(canLoadFromFile){
                    String[] cls2={"TSF","BOSS","RISE","ST"};
                    c=new CAWPE();
                    ((CAWPE)c).setFillMissingDistsWithOneHotVectors(true);
                    ((CAWPE)c).setBuildIndividualsFromResultsFiles(true);
                    ((CAWPE)c).setResultsFileLocationParameters(resultsPath, dataset, fold);
                    ((CAWPE)c).setClassifiersNamesForFileRead(cls2);
                }
                else
                    throw new UnsupportedOperationException("ERROR: currently only loading from file for CAWPE and no results file path has been set. "
                            + "Call setClassifier with an ExperimentalArguments object exp with exp.resultsWriteLocation (contains component classifier results) and exp.datasetName set");
                break;
            case "HC-Standard":
                if(canLoadFromFile){
                    String[] cls={"TSF","BOSS","RISE","STC","EE"};//RotF for ST
                    c=new CAWPE();
//                    ((CAWPE)c).setWeightingScheme(new TrainAcc(4));
//                    ((CAWPE)c).setVotingScheme(new MajorityConfidence());

                    ((CAWPE)c).setFillMissingDistsWithOneHotVectors(true);
                    ((CAWPE)c).setSeed(fold);
                    ((CAWPE)c).setBuildIndividualsFromResultsFiles(true);
                    ((CAWPE)c).setResultsFileLocationParameters(resultsPath, dataset, fold);
                    ((CAWPE)c).setClassifiersNamesForFileRead(cls);
                }
                else
                    throw new UnsupportedOperationException("ERROR: currently only loading from file for CAWPE and no results file path has been set. "
                            + "Call setClassifier with an ExperimentalArguments object exp with exp.resultsWriteLocation (contains component classifier results) and exp.datasetName set");
                break;
            case "HC-Alpha1":
                if(canLoadFromFile){
                    String[] cls={"TSF","BOSS","RISE","STC","EE"};//RotF for ST
                    c=new CAWPE();
                    ((CAWPE)c).setWeightingScheme(new TrainAcc(1));
//                    ((CAWPE)c).setVotingScheme(new MajorityConfidence());
                    ((CAWPE)c).setFillMissingDistsWithOneHotVectors(true);
                    ((CAWPE)c).setSeed(fold);
                    ((CAWPE)c).setBuildIndividualsFromResultsFiles(true);
                    ((CAWPE)c).setResultsFileLocationParameters(resultsPath, dataset, fold);
                    ((CAWPE)c).setClassifiersNamesForFileRead(cls);
                }
                else
                    throw new UnsupportedOperationException("ERROR: currently only loading from file for CAWPE and no results file path has been set. "
                            + "Call setClassifier with an ExperimentalArguments object exp with exp.resultsWriteLocation (contains component classifier results) and exp.datasetName set");
                break;
            case "HC-NewBOSS":
                if(canLoadFromFile){
                    String[] cls={"TSF","BOSS-New","RISE","STC","EE"};//RotF for ST
                    c=new CAWPE();
//                    ((CAWPE)c).setWeightingScheme(new TrainAcc(4));
//                    ((CAWPE)c).setVotingScheme(new MajorityConfidence());

                    ((CAWPE)c).setFillMissingDistsWithOneHotVectors(true);
                    ((CAWPE)c).setSeed(fold);
                    ((CAWPE)c).setBuildIndividualsFromResultsFiles(true);
                    ((CAWPE)c).setResultsFileLocationParameters(resultsPath, dataset, fold);
                    ((CAWPE)c).setClassifiersNamesForFileRead(cls);
                }
                else
                    throw new UnsupportedOperationException("ERROR: currently only loading from file for CAWPE and no results file path has been set. "
                            + "Call setClassifier with an ExperimentalArguments object exp with exp.resultsWriteLocation (contains component classifier results) and exp.datasetName set");
                break;
            case "HC-NoEE":
                if(canLoadFromFile){
                    String[] cls={"TSF","BOSS","RISE","STC"};//RotF for ST
                    c=new CAWPE();
//                    ((CAWPE)c).setWeightingScheme(new TrainAcc(4));
//                    ((CAWPE)c).setVotingScheme(new MajorityConfidence());

                    ((CAWPE)c).setFillMissingDistsWithOneHotVectors(true);
                    ((CAWPE)c).setSeed(fold);
                    ((CAWPE)c).setBuildIndividualsFromResultsFiles(true);
                    ((CAWPE)c).setResultsFileLocationParameters(resultsPath, dataset, fold);
                    ((CAWPE)c).setClassifiersNamesForFileRead(cls);
                }
                else
                    throw new UnsupportedOperationException("ERROR: currently only loading from file for CAWPE and no results file path has been set. "
                            + "Call setClassifier with an ExperimentalArguments object exp with exp.resultsWriteLocation (contains component classifier results) and exp.datasetName set");
                break;

            case "HC-SB":
                if(canLoadFromFile){
                    String[] cls={"TSF","S-BOSS","RISE","STC","EE"};//RotF for ST
                    c=new CAWPE();
                    ((CAWPE)c).setFillMissingDistsWithOneHotVectors(true);
                    ((CAWPE)c).setSeed(fold);
                    ((CAWPE)c).setBuildIndividualsFromResultsFiles(true);
                    ((CAWPE)c).setResultsFileLocationParameters(resultsPath, dataset, fold);
                    ((CAWPE)c).setClassifiersNamesForFileRead(cls);
                }
                else
                    throw new UnsupportedOperationException("ERROR: currently only loading from file for CAWPE and no results file path has been set. "
                            + "Call setClassifier with an ExperimentalArguments object exp with exp.resultsWriteLocation (contains component classifier results) and exp.datasetName set");
                break;
            case "HC-NoEE-SB":
                if(canLoadFromFile){
                    String[] cls={"TSF","S-BOSS","RISE","STC"};//RotF for ST
                    c=new CAWPE();
                    ((CAWPE)c).setFillMissingDistsWithOneHotVectors(true);
                    ((CAWPE)c).setSeed(fold);
                    ((CAWPE)c).setBuildIndividualsFromResultsFiles(true);
                    ((CAWPE)c).setResultsFileLocationParameters(resultsPath, dataset, fold);
                    ((CAWPE)c).setClassifiersNamesForFileRead(cls);
                }
                else
                    throw new UnsupportedOperationException("ERROR: currently only loading from file for CAWPE and no results file path has been set. "
                            + "Call setClassifier with an ExperimentalArguments object exp with exp.resultsWriteLocation (contains component classifier results) and exp.datasetName set");
                break;
            case "HC-WEASEL":
                if(canLoadFromFile){
                    String[] cls={"TSF","WEASEL","RISE","STC","EE"};//RotF for ST
                    c=new CAWPE();
                    ((CAWPE)c).setFillMissingDistsWithOneHotVectors(true);
                    ((CAWPE)c).setSeed(fold);
                    ((CAWPE)c).setBuildIndividualsFromResultsFiles(true);
                    ((CAWPE)c).setResultsFileLocationParameters(resultsPath, dataset, fold);
                    ((CAWPE)c).setClassifiersNamesForFileRead(cls);
                }
                else
                    throw new UnsupportedOperationException("ERROR: currently only loading from file for CAWPE and no results file path has been set. "
                            + "Call setClassifier with an ExperimentalArguments object exp with exp.resultsWriteLocation (contains component classifier results) and exp.datasetName set");
                break;
            case "HC-PF-SB":
                if(canLoadFromFile){
                    String[] cls={"TSF","S-BOSS","RISE","STC","ProximityForest"};//RotF for ST
                    c=new CAWPE();
                    ((CAWPE)c).setFillMissingDistsWithOneHotVectors(true);
                    ((CAWPE)c).setSeed(fold);
                    ((CAWPE)c).setBuildIndividualsFromResultsFiles(true);
                    ((CAWPE)c).setResultsFileLocationParameters(resultsPath, dataset, fold);
                    ((CAWPE)c).setClassifiersNamesForFileRead(cls);
                }
                else
                    throw new UnsupportedOperationException("ERROR: currently only loading from file for CAWPE and no results file path has been set. "
                            + "Call setClassifier with an ExperimentalArguments object exp with exp.resultsWriteLocation (contains component classifier results) and exp.datasetName set");
                break;
            case "HC-PF":
                if(canLoadFromFile){
                    String[] cls={"TSF","BOSS","RISE","STC","ProximityForest"};//RotF for ST
                    c=new CAWPE();
                    ((CAWPE)c).setFillMissingDistsWithOneHotVectors(true);
                    ((CAWPE)c).setSeed(fold);
                    ((CAWPE)c).setBuildIndividualsFromResultsFiles(true);
                    ((CAWPE)c).setResultsFileLocationParameters(resultsPath, dataset, fold);
                    ((CAWPE)c).setClassifiersNamesForFileRead(cls);
                }
                else
                    throw new UnsupportedOperationException("ERROR: currently only loading from file for CAWPE and no results file path has been set. "
                            + "Call setClassifier with an ExperimentalArguments object exp with exp.resultsWriteLocation (contains component classifier results) and exp.datasetName set");
                break;
//            "FullHC","FullHC-SB","FullHC-SB-PF"
            case "FullHC":
                ArrayList<Classifier> cls=new ArrayList<>();
                cls.add(new TSF());
                cls.add(new BOSS());
                cls.add(new RISE());
                cls.add(new ShapeletTransformClassifier());
                cls.add(new ElasticEnsemble());
                String[] clsNames={"TSF","BOSS","RISE","STC","ElasticEnsemble"};
                ArrayList<String> names = new ArrayList<>(Arrays.asList(clsNames));
                c=new HiveCote(cls,names);
                ((HiveCote)c).setContract(4);
                break;

            case "FullHC-SB":
                ArrayList<Classifier> cls2=new ArrayList<>();
                cls2.add(new TSF());
                cls2.add(new SpatialBOSS());
                cls2.add(new RISE());
                cls2.add(new ShapeletTransformClassifier());
                cls2.add(new ElasticEnsemble());
                String[] clsNames2={"TSF","S-BOSS","RISE","STC","ElasticEnsemble"};
                ArrayList<String> names2 = new ArrayList<>(Arrays.asList(clsNames2));
                c=new HiveCote(cls2,names2);
                ((HiveCote)c).setContract(4);
                break;
            case "HC-Latest":
                if(canLoadFromFile){
                    clsNames2=new String[]{"TSF","S-BOSS","RISE","STC","PF"};//RotF for ST
                    c=new CAWPE();
                    ((CAWPE)c).setFillMissingDistsWithOneHotVectors(true);
                    ((CAWPE)c).setSeed(fold);
                    ((CAWPE)c).setBuildIndividualsFromResultsFiles(true);
                    ((CAWPE)c).setResultsFileLocationParameters(resultsPath, dataset, fold);
                    ((CAWPE)c).setClassifiersNamesForFileRead(clsNames2);
                }
                else
                    throw new UnsupportedOperationException("ERROR: currently only loading from file for CAWPE and no results file path has been set. "
                            + "Call setClassifier with an ExperimentalArguments object exp with exp.resultsWriteLocation (contains component classifier results) and exp.datasetName set");
                break;
            case "HC-Catch22TSF":
                if(canLoadFromFile){
                    String[] cls3={"Catch22TSF","BOSS","RISE","STC","EE"};//RotF for ST
                    c=new CAWPE();
                    ((CAWPE)c).setFillMissingDistsWithOneHotVectors(true);
                    ((CAWPE)c).setSeed(fold);
                    ((CAWPE)c).setBuildIndividualsFromResultsFiles(true);
                    ((CAWPE)c).setResultsFileLocationParameters(resultsPath, dataset, fold);
                    ((CAWPE)c).setClassifiersNamesForFileRead(cls3);
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
        ClassifierBuilderFactory.ClassifierBuilder classifierBuilder =
            ClassifierBuilderFactory.getGlobalInstance().getClassifierBuilderByName(classifier);
        Classifier c = null;
        if(classifierBuilder != null) {
            return classifierBuilder.build();
        } else if(distanceBased.contains(classifier))
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
