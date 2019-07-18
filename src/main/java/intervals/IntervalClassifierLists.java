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
import weka.classifiers.Classifier;

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
        
    };
    
    public static final String[] targetClassifiers = {
        
    };
    
    public static final String[] allClassifiers = arrConcat(proxyClassifiers, targetClassifiers);
    
    
    
    
    
    
    
    
    
    
    public static Classifier setClassifier(ExperimentalArguments exp) {        
        return setClassifier(exp.classifierName, exp.foldId);
    }
    
    
    public static Classifier setClassifier(String classifier, int fold) {                
        switch(classifier){
            
            
            ///////////////// PROXIES
            
            case "ED":
                return null;
                
            case "SVML":
                return null;
                
            case "SLOWDTWCV":
                return null;
                
            case "DTW": //without cv-ing the warp window, just setting it to.. 0.2? ask jay/george
                return null;
                
            case "C45": 
                return null;
             
            case "BayesNet":
                return null;
            
            //maybe more
                
                
            ///////////////// TARGETS
                
            case "RISE": 
                return null;
                
            case "TSF":
                return null;
                
            case "ST": 
                return null;
                
            case "BOSS":
                return null;
                
            case "EE":
                return null;
                
            case "HIVE-COTE":
                return null;
                
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
