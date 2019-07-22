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

    // Fungi removed due to 1 case per class in train, and we need to do cv
    public static final String[] datasets_SeriesLengthAtLeast100 = {
        "ACSF1",	"Adiac",	"AllGestureWiimoteX",	"AllGestureWiimoteY",	"AllGestureWiimoteZ",	
        "ArrowHead",	"Beef",	"BeetleFly",	"BirdChicken",	"BME",	"Car",	"CBF",	"ChlorineConcentration",	
        "CinCECGTorso",	"Coffee",	"Computers",	"CricketX",	"CricketY",	"CricketZ",	
        "DiatomSizeReduction",	"DodgerLoopDay",	"DodgerLoopGame",	"DodgerLoopWeekend",	"Earthquakes",	
        "ECG5000",	"ECGFiveDays",	"EOGHorizontalSignal",	"EOGVerticalSignal",	"EthanolLevel",	"FaceAll",	
        "FaceFour",	"FacesUCR",	"FiftyWords",	"Fish",	"FordA",	"FordB",	"FreezerRegularTrain",	
        "FreezerSmallTrain",	/*"Fungi",*/	"GestureMidAirD1",	"GestureMidAirD2",	"GestureMidAirD3",	
        "GesturePebbleZ1",	"GesturePebbleZ2",	"GunPoint",	"GunPointAgeSpan",	"GunPointMaleVersusFemale",	
        "GunPointOldVersusYoung",	"Ham",	"HandOutlines",	"Haptics",	"Herring",	"HouseTwenty",	"InlineSkate",	
        "InsectEPGRegularTrain",	"InsectEPGSmallTrain",	"InsectWingbeatSound",	"LargeKitchenAppliances",	
        "Lightning2",	"Lightning7",	"Mallat",	"Meat",	"MixedShapesRegularTrain",	"MixedShapesSmallTrain",	
        "NonInvasiveFetalECGThorax1",	"NonInvasiveFetalECGThorax2",	"OliveOil",	"OSULeaf",	"Phoneme",	
        "PickupGestureWiimoteZ",	"PigAirwayPressure",	"PigArtPressure",	"PigCVP",	"PLAID",	
        "Plane",	"PowerCons",	"RefrigerationDevices",	"Rock",	"ScreenType",	"SemgHandGenderCh2",	
        "SemgHandMovementCh2",	"SemgHandSubjectCh2",	"ShakeGestureWiimoteZ",	"ShapeletSim",	"ShapesAll",	
        "SmallKitchenAppliances",	"StarLightCurves",	"Strawberry",	"SwedishLeaf",	"Symbols",	
        "ToeSegmentation1",	"ToeSegmentation2",	"Trace",	"TwoPatterns",	"UMD",	"UWaveGestureLibraryAll",	
        "UWaveGestureLibraryX",	"UWaveGestureLibraryY",	"UWaveGestureLibraryZ",	"Wafer",	"Wine",	"WordSynonyms",	
        "Worms",	"WormsTwoClass",	"Yoga",
    };
    
    
    //replaceLabelsForImages() for final labels
    public static final String[] proxyClassifiers = {  
        "ED", "SVML", "SLOWDTWCV", "DTW_20", "C45", "BayesNet", "RandF", "RotF",
    };
    
    public static final String[] hiveCoteMembers_uncontracted = {
        "TSF", "ST5Day_RotF", "EE", "BOSS", "RISE",
    };
    
    public static final String[] hiveCoteMembers_contracted = {
        "TSF", "cST_RotF", "cEE", "cBOSS", "cRISE",
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
                
            case "DTW_20": //without cv-ing the warp window, just setting it to.. 0.2? ask jay/george
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
                
            case "ST5Day_RotF": 
                //transforms saved via TransformExperiments. timings to be added on semi-manually
                //will only be using rotf as final classifier, at least as good as cawpe  on average and less variable on timings/space reqs (logistic can be a massive pig)
                //might need to end up contracting even in the 'uncontracted' version for feasibility, in that case can assume contract time (7 days e.g)
                RotationForest strotf = new RotationForest();
                strotf.setNumIterations(50);
                strotf.setSeed(fold);
                //todo find someway to load back the st timing files to add onto the rotf time.
                return strotf;
                
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
                
//            case "cST_RotF": 
//                ShapeletTransformClassifier cst = new ShapeletTransformClassifier();
//                cst.setSeed(fold);
//                cst.setHourLimit(contractHourLimit);
//                return cst;
                
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
        System.out.println(datasets_SeriesLengthAtLeast100.length);
    }
    
}
