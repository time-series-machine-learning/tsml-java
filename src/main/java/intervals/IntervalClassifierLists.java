/*
 * Copyright (C) 2019 xmw13bzu
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

package intervals;

import experiments.Experiments.ExperimentalArguments;
import timeseriesweka.classifiers.dictionary_based.BOSS;
import timeseriesweka.classifiers.distance_based.ElasticEnsemble;
import timeseriesweka.classifiers.distance_based.SlowDTW_1NN;
import timeseriesweka.classifiers.distance_based.elastic_ensemble.ED1NN;
import timeseriesweka.classifiers.frequency_based.CRISE;
import timeseriesweka.classifiers.frequency_based.RISE;
import timeseriesweka.classifiers.interval_based.TSF;
import timeseriesweka.classifiers.shapelet_based.ShapeletTransformClassifier;
import vector_classifiers.CAWPE;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.meta.RotationForest;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;

/**
 *
 * @author James Large (james.large@uea.ac.uk)
 */
public class IntervalClassifierLists {

    //copied from experiments.DataSets.tscProblems2018 to have as a permanantly fixed list here
    public static final String[] datasetList = {
        "ACSF1",
        "Adiac",        // 390,391,176,37
        "AllGestureWiimoteX",       
        "AllGestureWiimoteY",       
        "AllGestureWiimoteZ",       
        "ArrowHead",    // 36,175,251,3
        "Beef",         // 30,30,470,5
        "BeetleFly",    // 20,20,512,2
        "BirdChicken",  // 20,20,512,2
        "BME",
        "Car",          // 60,60,577,4
        "CBF",                      // 30,900,128,3
        "Chinatown",
        "ChlorineConcentration",    // 467,3840,166,3
        "CinCECGTorso", // 40,1380,1639,4
        "Coffee", // 28,28,286,2
        "Computers", // 250,250,720,2
        "CricketX", // 390,390,300,12
        "CricketY", // 390,390,300,12
        "CricketZ", // 390,390,300,12
        "Crop",
        "DiatomSizeReduction", // 16,306,345,4
        "DistalPhalanxOutlineAgeGroup", // 400,139,80,3
        "DistalPhalanxOutlineCorrect", // 600,276,80,2
        "DistalPhalanxTW", // 400,139,80,6
        "DodgerLoopDay",
        "DodgerLoopGame",
        "DodgerLoopWeekend",
        "Earthquakes", // 322,139,512,2
        "ECG200",   //100, 100, 96
        "ECG5000",  //4500, 500,140
        "ECGFiveDays", // 23,861,136,2
        "ElectricDevices", // 8926,7711,96,7
        "EOGHorizontalSignal",
        "EOGVerticalSignal",
        "EthanolLevel",
        "FaceAll", // 560,1690,131,14
        "FaceFour", // 24,88,350,4
        "FacesUCR", // 200,2050,131,14
        "FiftyWords", // 450,455,270,50
        "Fish", // 175,175,463,7
        "FordA", // 3601,1320,500,2
        "FordB", // 3636,810,500,2
        "FreezerRegularTrain",
        "FreezerSmallTrain",
        "Fungi",
        "GestureMidAirD1",
        "GestureMidAirD2",
        "GestureMidAirD3",
        "GesturePebbleZ1",
        "GesturePebbleZ2",                       
        "GunPoint", // 50,150,150,2
        "GunPointAgeSpan",
        "GunPointMaleVersusFemale",
        "GunPointOldVersusYoung",                        
        "Ham",      //105,109,431
        "HandOutlines", // 1000,370,2709,2
        "Haptics", // 155,308,1092,5
        "Herring", // 64,64,512,2
        "HouseTwenty",
        "InlineSkate", // 100,550,1882,7
        "InsectEPGRegularTrain",
        "InsectEPGSmallTrain",
        "InsectWingbeatSound",//1980,220,256
        "ItalyPowerDemand", // 67,1029,24,2
        "LargeKitchenAppliances", // 375,375,720,3
        "Lightning2", // 60,61,637,2
        "Lightning7", // 70,73,319,7
        "Mallat", // 55,2345,1024,8
        "Meat",//60,60,448
        "MedicalImages", // 381,760,99,10
        "MelbournePedestrian",
        "MiddlePhalanxOutlineAgeGroup", // 400,154,80,3
        "MiddlePhalanxOutlineCorrect", // 600,291,80,2
        "MiddlePhalanxTW", // 399,154,80,6
        "MixedShapesRegularTrain",
        "MixedShapesSmallTrain",
        "MoteStrain", // 20,1252,84,2
        "NonInvasiveFetalECGThorax1", // 1800,1965,750,42
        "NonInvasiveFetalECGThorax2", // 1800,1965,750,42
        "OliveOil", // 30,30,570,4
        "OSULeaf", // 200,242,427,6
        "PhalangesOutlinesCorrect", // 1800,858,80,2
        "Phoneme",//1896,214, 1024
        "PickupGestureWiimoteZ",
        "PigAirwayPressure",
        "PigArtPressure",
        "PigCVP",
        "PLAID",
        "Plane", // 105,105,144,7
        "PowerCons",
        "ProximalPhalanxOutlineAgeGroup", // 400,205,80,3
        "ProximalPhalanxOutlineCorrect", // 600,291,80,2
        "ProximalPhalanxTW", // 400,205,80,6
        "RefrigerationDevices", // 375,375,720,3
        "Rock",
        "ScreenType", // 375,375,720,3
        "SemgHandGenderCh2",
        "SemgHandMovementCh2",
        "SemgHandSubjectCh2",
        "ShakeGestureWiimoteZ",
        "ShapeletSim", // 20,180,500,2
        "ShapesAll", // 600,600,512,60
        "SmallKitchenAppliances", // 375,375,720,3
        "SmoothSubspace",
        "SonyAIBORobotSurface1", // 20,601,70,2
        "SonyAIBORobotSurface2", // 27,953,65,2
        "StarLightCurves", // 1000,8236,1024,3
        "Strawberry",//370,613,235
        "SwedishLeaf", // 500,625,128,15
        "Symbols", // 25,995,398,6
        "SyntheticControl", // 300,300,60,6
        "ToeSegmentation1", // 40,228,277,2
        "ToeSegmentation2", // 36,130,343,2
        "Trace", // 100,100,275,4
        "TwoLeadECG", // 23,1139,82,2
        "TwoPatterns", // 1000,4000,128,4
        "UMD",
        "UWaveGestureLibraryAll", // 896,3582,945,8
        "UWaveGestureLibraryX", // 896,3582,315,8
        "UWaveGestureLibraryY", // 896,3582,315,8
        "UWaveGestureLibraryZ", // 896,3582,315,8
        "Wafer", // 1000,6164,152,2
        "Wine",//54	57	234
        "WordSynonyms", // 267,638,270,25
        "Worms", //77, 181,900,5
        "WormsTwoClass",//77, 181,900,5
        "Yoga" // 300,3000,426,2
    };
    
    
    //replaceLabelsForImages() for final labels
    public static final String[] proxyClassifiers = {  
        "ED", "SVML", "SLOWDTWCV", "DTW", "C45", "BayesNet", "RandF", "RotF",
    };
    
    public static final String[] hiveCoteMembers_uncontracted = {
        "TSF", "ST", "EE", "BOSS", "RISE",
    };
    
    public static final String[] hiveCoteMembers_contracted = {
        "TSF", "cST", "cEE", "cBOSS", "cRISE",
    };
            
    public static final String[] targetClassifiers_uncontracted = arrConcat(hiveCoteMembers_uncontracted, new String[] { "HIVE-COTE" });
    
    public static final String[] targetClassifiers_contracted = arrConcat(hiveCoteMembers_contracted, new String[] { "cHIVE-COTE" });
    
    
    public static final int contractHourLimit = 1;
    
    
    
    
    
    
    
    public static Classifier setClassifier(ExperimentalArguments exp) {    
        switch (exp.classifierName) {
            
            case "HIVE-COTE":
                CAWPE hc = new CAWPE(); //proabilities weighted by exponentiated accuracy estimates
                hc.setRandSeed(exp.foldId);
                hc.setBuildIndividualsFromResultsFiles(true);
                hc.setResultsFileLocationParameters(exp.resultsWriteLocation, exp.datasetName, exp.foldId);
                hc.setClassifiersNamesForFileRead(hiveCoteMembers_uncontracted);
                return hc;
             
            case "cHIVE-COTE":
                CAWPE chc = new CAWPE(); //proabilities weighted by exponentiated accuracy estimates
                chc.setRandSeed(exp.foldId);
                chc.setBuildIndividualsFromResultsFiles(true);
                chc.setResultsFileLocationParameters(exp.resultsWriteLocation, exp.datasetName, exp.foldId);
                chc.setClassifiersNamesForFileRead(hiveCoteMembers_contracted);
                return chc;
                
            default:
                return setClassifier(exp.classifierName, exp.foldId);
        }
        
    }
    
    
    public static Classifier setClassifier(String classifier, int fold) {                
        switch(classifier){
            
            
            ///////////////// PROXIES
            
            case "ED":
                return new ED1NN();
                
            case "SVML":
                SMO svml = new SMO();
                PolyKernel p=new PolyKernel();
                p.setExponent(1);
                svml.setKernel(p);
                svml.setRandomSeed(fold);
                svml.setBuildLogisticModels(true);
                return svml;
                
            case "SLOWDTWCV":
                SlowDTW_1NN dtwcv = new SlowDTW_1NN(); //slower, but more stable.
                dtwcv.optimiseWindow(true);
                return dtwcv;
                
            case "DTW": //without cv-ing the warp window, just setting it to.. 0.2? ask jay/george
                SlowDTW_1NN dtw = new SlowDTW_1NN();
                dtw.setMaxPercentageWarp(20);
                return dtw;
                
            case "C45": 
                return new J48();
             
            case "BayesNet":
                return new BayesNet();
                
            case "RandF": 
                RandomForest randf = new RandomForest();
                randf.setNumTrees(500);
                randf.setSeed(fold);
                return randf;
                
            case "RotF":
                RotationForest rotf = new RotationForest();
                rotf.setNumIterations(50);
                rotf.setSeed(fold);
                return rotf;
            
            //maybe more, depends on interim results
                
                
            ///////////////// TARGETS 1: UNCONTRACTED
                
            
            case "RISE":
                CRISE rise = new CRISE(fold);
                return rise;
                
            case "TSF":
                TSF tsf = new TSF();
                tsf.setSeed(fold);
                return tsf;
                
            case "ST": 
                //complete via TransformExperiments? does TransformExperiments record the time taken to transform too somewhere? 
                //will only be using rotf as final classifier, dont necessarily care about saving the transforms themselves
                //might need to end up contracting even in the 'uncontracted' version for feasibility, in that case can assume contract time (7 days e.g)
                ShapeletTransformClassifier st = new ShapeletTransformClassifier();
                st.setSeed(fold);
                return st;
                
            case "BOSS":
                BOSS boss = new BOSS();
                boss.setSeed(fold);
                return boss;
                
            case "EE":
                ElasticEnsemble ee = new ElasticEnsemble();
                //seed? or is this deterministic, and the contract version is separate class? 
                return ee;
                
//            case "HIVE-COTE":
//                see the ExperimentalArgs overload of setClassifier
//                return null;
            
                
            ///////////////// TARGETS 2: CONTRACTED
                
            case "cRISE":
                CRISE crise = new CRISE(fold);
                crise.setHourLimit(contractHourLimit);
                return null;
                
//            case "cTSF":
//                does not exist? likely would not need anyway
//                return null;
                
            case "cST": 
                ShapeletTransformClassifier cst = new ShapeletTransformClassifier();
                cst.setSeed(fold);
                cst.setHourLimit(contractHourLimit);
                return cst;
                
            case "cBOSS":
                BOSS cboss = new BOSS();
                cboss.setSeed(fold);
                cboss.setHourLimit(contractHourLimit);
                return cboss;
                
            case "cEE":
                //???
                return null;
//                
//            case "cHIVE-COTE":
//                see the ExperimentalArgs overload of setClassifier
//                return null;
                
                
            default:
                System.out.println("UNKNOWN CLASSIFIER "+classifier);
                System.exit(0);
//                throw new Exception("Unknown classifier "+classifier);
                
        }
        
        return null;
    }
    
    public static String[] replaceLabelsForImages(String[] a) {
        final String[] find = { 
            "SLOWDTWCV",
        };
        
        final String[] replace = { 
            "DTWCV",
        };
        
        String[] copy = new String[a.length];
        
        for (int i = 0; i < a.length; i++) {
            copy[i] = a[i];
        
            for (int j = 0; j < find.length; j++) {
                if (a[i].equals(find[j])) {
                    copy[i] = replace[j];
                    break;
                }
            }
        }
        
        return copy;
    }
    
    public static String[] removeClassifier(String[] a, String toremove) {
        String[] copy = new String[a.length-1];
        
        for (int i = 0, j = 0; j < a.length; i++, j++)
            if (!a[j].equals(toremove))
                copy[i] = a[j];
            else 
                i--;
        
        return copy;
    }
    
    public static String[] arrConcat(String[] a, String[] b) {
        String[] res = new String[a.length + b.length];
        
        int i = 0;
        for (int j = 0; j < a.length; j++, i++)
            res[i] = a[j];
        for (int j = 0; j < b.length; j++, i++)
            res[i] = b[j];
        
        return res;
    }
    
    public static void main(String[] args) {
        System.out.println(datasetList.length);
    }
    
}
