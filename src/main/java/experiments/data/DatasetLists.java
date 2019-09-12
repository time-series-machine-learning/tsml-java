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
package experiments.data;


import experiments.data.DatasetLoading;
import fileIO.InFile;
import fileIO.OutFile;
import timeseriesweka.filters.SummaryStats;
import utilities.ClassifierTools;
import weka.classifiers.Classifier;
import weka.classifiers.lazy.IBk;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.TreeSet;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

/**
 * Class containing lists of data sets in the UCR and UEA archive. 
 * @author ajb
 */
public class DatasetLists {
    
    public static String clusterPath="/gpfs/home/ajb/";
    public static String dropboxPath="C:/Users/ajb/Dropbox/";    
    public static String beastPath="//cmptscsvr.cmp.uea.ac.uk/ueatsc/Data/";
    public static  String path=clusterPath;    
    
    public static String problemPath=path+"/TSCProblems/";
    public static String resultsPath=path+"Results/";
    public static String uciPath=path+"UCIContinuous";
    
//Multivariate TSC data sets  
   //<editor-fold defaultstate="collapsed" desc="Multivariate TSC datasets 2018 release">    
    public static String[] mtscProblems2018={
        "ArticularyWordRecognition", //Index 0
        "AtrialFibrillation",//1
        "BasicMotions",
        "CharacterTrajectories",
        "Cricket",
        "DuckDuckGeese",
        "EigenWorms",
        "Epilepsy",
        "EthanolConcentration",
        "ERing",
        "FaceDetection",//10
        "FingerMovements",
        "HandMovementDirection",
        "Handwriting",
        "Heartbeat",
        "InsectWingbeat",//15
//        "KickVsPunch", Poorly formatted and very small train size
        "JapaneseVowels",
        "Libras",
        "LSST",
        "MotorImagery",
        "NATOPS",//20
        "PenDigits",
        "PEMS-SF",
        "PhonemeSpectra",
        "RacketSports",
        "SelfRegulationSCP1",//25
        "SelfRegulationSCP2",
        "SpokenArabicDigits",
        "StandWalkJump",        
        "UWaveGestureLibrary"            
};    
       //</editor-fold>       

 //TSC data sets for relaunch in 2018 
    //<editor-fold defaultstate="collapsed" desc="tsc Problems 2018 ">    
		public static String[] tscProblems2018={	
                                //Train Size, Test Size, Series Length, Nos Classes
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
      //</editor-fold>    

   public static String[] variableLength2018Problems={
        "AllGestureWiimoteX",
        "AllGestureWiimoteY",
        "AllGestureWiimoteZ",
        "GestureMidAirD1",
        "GestureMidAirD2",
        "GestureMidAirD3",        
        "GesturePebbleZ1",
        "GesturePebbleZ2",
        "PickupGestureWiimoteZ",
        "PLAID",
        "ShakeGestureWiimoteZ"
   };
   
   
   public static String[] missingValue2018Problems={
        "AllGestureWiimoteX",
        "AllGestureWiimoteY",
        "AllGestureWiimoteZ",
        "DodgerLoopDay",
        "DodgerLoopGame",
        "DodgerLoopWeekend",
        "GestureMidAirD1",
        "GestureMidAirD2",
        "GestureMidAirD3",
        "GesturePebbleZ1",
        "GesturePebbleZ2",
        "MelbournePedestrian",
        "PickupGestureWiimoteZ",
        "PLAID",
        "ShakeGestureWiimoteZ"
   };
 //TSC data sets for bakeoff redux 
    //<editor-fold defaultstate="collapsed" desc="tsc Problems 2018 ">    
		public static String[] equalLengthProblems={	
                                //Train Size, Test Size, Series Length, Nos Classes
                        "ACSF1",
			"Adiac",        // 390,391,176,37
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
//                        "Fungi", removed because only one instance per class in train. This is a query problem
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
                        "PigAirwayPressure",
                        "PigArtPressure",
                        "PigCVP",
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
      //</editor-fold>    

   
   
   public static String[] newFor2018Problems={
        "ACSF1",
        "AllGestureWiimoteX",
        "AllGestureWiimoteY",
        "AllGestureWiimoteZ",
        "BME",
        "Chinatown",
        "Crop",
        "DodgerLoopDay",
        "DodgerLoopGame",
        "DodgerLoopWeekend",
        "EOGHorizontalSignal",
        "EOGVerticalSignal",
        "EthanolLevel",
        "FreezerRegularTrain",
        "FreezerSmallTrain",
        "Fungi",
        "GestureMidAirD1",
        "GestureMidAirD2",
        "GestureMidAirD3",
        "GesturePebbleZ1",
        "GesturePebbleZ2",
        "GunPointAgeSpan",
        "GunPointMaleVersusFemale",
        "GunPointOldVersusYoung",
        "HouseTwenty",
        "InsectEPGRegularTrain",
        "InsectEPGSmallTrain",
        "MelbournePedestrian",
        "MixedShapesRegularTrain",
        "MixedShapesSmallTrain",
        "PickupGestureWiimoteZ",
        "PigAirwayPressure",
        "PigArtPressure",
        "PigCVP",
        "PLAID",
        "PowerCons",
        "Rock",
        "SemgHandGenderCh2",
        "SemgHandMovementCh2",
        "SemgHandSubjectCh2",
        "ShakeGestureWiimoteZ",
        "SmoothSubspace",
        "UMD",
   };
                

   public static String[] newFor2018Problems_noMissingValues={
        "ACSF1",
        "BME",
        "Chinatown",
        "Crop",
        "EOGHorizontalSignal",
        "EOGVerticalSignal",
        "EthanolLevel",
        "FreezerRegularTrain",
        "FreezerSmallTrain",
        "Fungi",
        "GunPointAgeSpan",
        "GunPointMaleVersusFemale",
        "GunPointOldVersusYoung",
        "HouseTwenty",
        "InsectEPGRegularTrain",
        "InsectEPGSmallTrain",
        "MelbournePedestrian",
        "MixedShapesRegularTrain",
        "MixedShapesSmallTrain",
        "PigAirwayPressure",
        "PigArtPressure",
        "PigCVP",
        "PowerCons",
        "Rock",
        "SemgHandGenderCh2",
        "SemgHandMovementCh2",
        "SemgHandSubjectCh2",
        "SmoothSubspace",
        "UMD",
   };
   
     
//TSC data sets before relaunch in 2018 
    //<editor-fold defaultstate="collapsed" desc="tsc Problems prior to relaunch in 2018 ">    
		public static String[] tscProblems2017={	
                    "AALTDChallenge",
                    "Acsf1",
                                //Train Size, Test Size, Series Length, Nos Classes
                                //Train Size, Test Size, Series Length, Nos Classes
			"Adiac",        // 390,391,176,37
			"ArrowHead",    // 36,175,251,3
			"Beef",         // 30,30,470,5
			"BeetleFly",    // 20,20,512,2
			"BirdChicken",  // 20,20,512,2
			"Car",          // 60,60,577,4
			"CBF",                      // 30,900,128,3
			"ChlorineConcentration",    // 467,3840,166,3
			"CinCECGTorso", // 40,1380,1639,4
			"Coffee", // 28,28,286,2
			"Computers", // 250,250,720,2
			"CricketX", // 390,390,300,12
			"CricketY", // 390,390,300,12
			"CricketZ", // 390,390,300,12
			"DiatomSizeReduction", // 16,306,345,4
			"DistalPhalanxOutlineCorrect", // 600,276,80,2
			"DistalPhalanxOutlineAgeGroup", // 400,139,80,3
			"DistalPhalanxTW", // 400,139,80,6
			"Earthquakes", // 322,139,512,2
                        "ECG200",   //100, 100, 96
                        "ECG5000",  //4500, 500,140
			"ECGFiveDays", // 23,861,136,2
			"ElectricDevices", // 8926,7711,96,7
			"FaceAll", // 560,1690,131,14
			"FaceFour", // 24,88,350,4
			"FacesUCR", // 200,2050,131,14
			"FiftyWords", // 450,455,270,50
			"Fish", // 175,175,463,7
			"FordA", // 3601,1320,500,2
			"FordB", // 3636,810,500,2
			"GunPoint", // 50,150,150,2
			"Ham",      //105,109,431
                        "HandOutlines", // 1000,370,2709,2
			"Haptics", // 155,308,1092,5
			"Herring", // 64,64,512,2
			"InlineSkate", // 100,550,1882,7
                        "InsectWingbeatSound",//1980,220,256
			"ItalyPowerDemand", // 67,1029,24,2
			"LargeKitchenAppliances", // 375,375,720,3
			"Lightning2", // 60,61,637,2
			"Lightning7", // 70,73,319,7
			"Mallat", // 55,2345,1024,8
			"Meat",//60,60,448
                        "MedicalImages", // 381,760,99,10
			"MiddlePhalanxOutlineCorrect", // 600,291,80,2
			"MiddlePhalanxOutlineAgeGroup", // 400,154,80,3
			"MiddlePhalanxTW", // 399,154,80,6
                        "MNIST",
			"MoteStrain", // 20,1252,84,2
			"NonInvasiveFetalECGThorax1", // 1800,1965,750,42
			"NonInvasiveFetalECGThorax2", // 1800,1965,750,42
			"OliveOil", // 30,30,570,4
			"OSULeaf", // 200,242,427,6
			"PhalangesOutlinesCorrect", // 1800,858,80,2
                        "Phoneme",//1896,214, 1024
			"Plane", // 105,105,144,7
                        "Plaid",
			"ProximalPhalanxOutlineCorrect", // 600,291,80,2
			"ProximalPhalanxOutlineAgeGroup", // 400,205,80,3
			"ProximalPhalanxTW", // 400,205,80,6
			"RefrigerationDevices", // 375,375,720,3
			"ScreenType", // 375,375,720,3
			"ShapeletSim", // 20,180,500,2
			"ShapesAll", // 600,600,512,60
			"SmallKitchenAppliances", // 375,375,720,3
			"SonyAIBORobotSurface1", // 20,601,70,2
			"SonyAIBORobotSurface2", // 27,953,65,2
			"StarlightCurves", // 1000,8236,1024,3
			"Strawberry",//370,613,235
                        "SwedishLeaf", // 500,625,128,15
			"Symbols", // 25,995,398,6
			"SyntheticControl", // 300,300,60,6
			"ToeSegmentation1", // 40,228,277,2
			"ToeSegmentation2", // 36,130,343,2
			"Trace", // 100,100,275,4
			"TwoLeadECG", // 23,1139,82,2
			"TwoPatterns", // 1000,4000,128,4
			"UWaveGestureLibraryX", // 896,3582,315,8
			"UWaveGestureLibraryY", // 896,3582,315,8
			"UWaveGestureLibraryZ", // 896,3582,315,8
			"UWaveGestureLibraryAll", // 896,3582,945,8
			"Wafer", // 1000,6164,152,2
			"Wine",//54	57	234
                        "WordSynonyms", // 267,638,270,25
			"Worms", //77, 181,900,5
                        "WormsTwoClass",//77, 181,900,5
                        "Yoga" // 300,3000,426,2
                };   
      //</editor-fold>    


    

//Bakeoff data sets, expansded in 2018  
    //<editor-fold defaultstate="collapsed" desc="tscProblems85: The new 85 UCR datasets">    
		public static String[] tscProblems85={	
                                //Train Size, Test Size, Series Length, Nos Classes
                                //Train Size, Test Size, Series Length, Nos Classes
			"Adiac",        // 390,391,176,37
			"ArrowHead",    // 36,175,251,3
			"Beef",         // 30,30,470,5
			"BeetleFly",    // 20,20,512,2
			"BirdChicken",  // 20,20,512,2
			"Car",          // 60,60,577,4
			"CBF",                      // 30,900,128,3
			"ChlorineConcentration",    // 467,3840,166,3
			"CinCECGTorso", // 40,1380,1639,4
			"Coffee", // 28,28,286,2
			"Computers", // 250,250,720,2
			"CricketX", // 390,390,300,12
			"CricketY", // 390,390,300,12
			"CricketZ", // 390,390,300,12
			"DiatomSizeReduction", // 16,306,345,4
			"DistalPhalanxOutlineCorrect", // 600,276,80,2
			"DistalPhalanxOutlineAgeGroup", // 400,139,80,3
			"DistalPhalanxTW", // 400,139,80,6
			"Earthquakes", // 322,139,512,2
                        "ECG200",   //100, 100, 96
                        "ECG5000",  //4500, 500,140
			"ECGFiveDays", // 23,861,136,2
			"ElectricDevices", // 8926,7711,96,7
			"FaceAll", // 560,1690,131,14
			"FaceFour", // 24,88,350,4
			"FacesUCR", // 200,2050,131,14
			"FiftyWords", // 450,455,270,50
			"Fish", // 175,175,463,7
			"FordA", // 3601,1320,500,2
			"FordB", // 3636,810,500,2
			"GunPoint", // 50,150,150,2
			"Ham",      //105,109,431
                        "HandOutlines", // 1000,370,2709,2
			"Haptics", // 155,308,1092,5
			"Herring", // 64,64,512,2
			"InlineSkate", // 100,550,1882,7
                        "InsectWingbeatSound",//1980,220,256
			"ItalyPowerDemand", // 67,1029,24,2
			"LargeKitchenAppliances", // 375,375,720,3
			"Lightning2", // 60,61,637,2
			"Lightning7", // 70,73,319,7
			"Mallat", // 55,2345,1024,8
			"Meat",//60,60,448
                        "MedicalImages", // 381,760,99,10
			"MiddlePhalanxOutlineCorrect", // 600,291,80,2
			"MiddlePhalanxOutlineAgeGroup", // 400,154,80,3
			"MiddlePhalanxTW", // 399,154,80,6
			"MoteStrain", // 20,1252,84,2
			"NonInvasiveFetalECGThorax1", // 1800,1965,750,42
			"NonInvasiveFetalECGThorax2", // 1800,1965,750,42
			"OliveOil", // 30,30,570,4
			"OSULeaf", // 200,242,427,6
			"PhalangesOutlinesCorrect", // 1800,858,80,2
                        "Phoneme",//1896,214, 1024
			"Plane", // 105,105,144,7
			"ProximalPhalanxOutlineCorrect", // 600,291,80,2
			"ProximalPhalanxOutlineAgeGroup", // 400,205,80,3
			"ProximalPhalanxTW", // 400,205,80,6
			"RefrigerationDevices", // 375,375,720,3
			"ScreenType", // 375,375,720,3
			"ShapeletSim", // 20,180,500,2
			"ShapesAll", // 600,600,512,60
			"SmallKitchenAppliances", // 375,375,720,3
			"SonyAIBORobotSurface1", // 20,601,70,2
			"SonyAIBORobotSurface2", // 27,953,65,2
			"StarlightCurves", // 1000,8236,1024,3
			"Strawberry",//370,613,235
                        "SwedishLeaf", // 500,625,128,15
			"Symbols", // 25,995,398,6
			"SyntheticControl", // 300,300,60,6
			"ToeSegmentation1", // 40,228,277,2
			"ToeSegmentation2", // 36,130,343,2
			"Trace", // 100,100,275,4
			"TwoLeadECG", // 23,1139,82,2
			"TwoPatterns", // 1000,4000,128,4
			"UWaveGestureLibraryX", // 896,3582,315,8
			"UWaveGestureLibraryY", // 896,3582,315,8
			"UWaveGestureLibraryZ", // 896,3582,315,8
			"UWaveGestureLibraryAll", // 896,3582,945,8
			"Wafer", // 1000,6164,152,2
			"Wine",//54	57	234
                        "WordSynonyms", // 267,638,270,25
			"Worms", //77, 181,900,5
                        "WormsTwoClass",//77, 181,900,5
                        "Yoga" // 300,3000,426,2
                };   
      //</editor-fold>    


                
                
                
 //TSC data sets for relaunch in 2018 
    //<editor-fold defaultstate="collapsed" desc="tsc Problems 2018, no missing values">    
		public static String[] tscProblems114={	
                                //Train Size, Test Size, Series Length, Nos Classes
                        "ACSF1",
			"Adiac",        // 390,391,176,37
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
                        "PigAirwayPressure",
                        "PigArtPressure",
                        "PigCVP",
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
      //</editor-fold>    

                
                
                
//Bakeoff data sets, expansded in 2018  
    //<editor-fold defaultstate="collapsed" desc="tscProblems78WithoutPigs:">    
		public static String[] tscProblems78={	
                                //Train Size, Test Size, Series Length, Nos Classes
                                //Train Size, Test Size, Series Length, Nos Classes
			"Adiac",        // 390,391,176,37
			"ArrowHead",    // 36,175,251,3
			"Beef",         // 30,30,470,5
			"BeetleFly",    // 20,20,512,2
			"BirdChicken",  // 20,20,512,2
			"Car",          // 60,60,577,4
			"CBF",                      // 30,900,128,3
			"ChlorineConcentration",    // 467,3840,166,3
			"CinCECGTorso", // 40,1380,1639,4
			"Coffee", // 28,28,286,2
			"Computers", // 250,250,720,2
			"CricketX", // 390,390,300,12
			"CricketY", // 390,390,300,12
			"CricketZ", // 390,390,300,12
			"DiatomSizeReduction", // 16,306,345,4
			"DistalPhalanxOutlineCorrect", // 600,276,80,2
			"DistalPhalanxOutlineAgeGroup", // 400,139,80,3
			"DistalPhalanxTW", // 400,139,80,6
			"Earthquakes", // 322,139,512,2
                        "ECG200",   //100, 100, 96
                        "ECG5000",  //4500, 500,140
			"ECGFiveDays", // 23,861,136,2
			"FaceAll", // 560,1690,131,14
			"FaceFour", // 24,88,350,4
			"FacesUCR", // 200,2050,131,14
			"FiftyWords", // 450,455,270,50
			"Fish", // 175,175,463,7
			"GunPoint", // 50,150,150,2
			"Ham",      //105,109,431
			"Haptics", // 155,308,1092,5
			"Herring", // 64,64,512,2
			"InlineSkate", // 100,550,1882,7
                        "InsectWingbeatSound",//1980,220,256
			"ItalyPowerDemand", // 67,1029,24,2
			"LargeKitchenAppliances", // 375,375,720,3
			"Lightning2", // 60,61,637,2
			"Lightning7", // 70,73,319,7
			"Mallat", // 55,2345,1024,8
			"Meat",//60,60,448
                        "MedicalImages", // 381,760,99,10
			"MiddlePhalanxOutlineCorrect", // 600,291,80,2
			"MiddlePhalanxOutlineAgeGroup", // 400,154,80,3
			"MiddlePhalanxTW", // 399,154,80,6
			"MoteStrain", // 20,1252,84,2
			"OliveOil", // 30,30,570,4
			"OSULeaf", // 200,242,427,6
			"PhalangesOutlinesCorrect", // 1800,858,80,2
                        "Phoneme",//1896,214, 1024
			"Plane", // 105,105,144,7
			"ProximalPhalanxOutlineCorrect", // 600,291,80,2
			"ProximalPhalanxOutlineAgeGroup", // 400,205,80,3
			"ProximalPhalanxTW", // 400,205,80,6
			"RefrigerationDevices", // 375,375,720,3
			"ScreenType", // 375,375,720,3
			"ShapeletSim", // 20,180,500,2
			"ShapesAll", // 600,600,512,60
			"SmallKitchenAppliances", // 375,375,720,3
			"SonyAIBORobotSurface1", // 20,601,70,2
			"SonyAIBORobotSurface2", // 27,953,65,2
			"Strawberry",//370,613,235
                        "SwedishLeaf", // 500,625,128,15
			"Symbols", // 25,995,398,6
			"SyntheticControl", // 300,300,60,6
			"ToeSegmentation1", // 40,228,277,2
			"ToeSegmentation2", // 36,130,343,2
			"Trace", // 100,100,275,4
			"TwoLeadECG", // 23,1139,82,2
			"TwoPatterns", // 1000,4000,128,4
			"UWaveGestureLibraryX", // 896,3582,315,8
			"UWaveGestureLibraryY", // 896,3582,315,8
			"UWaveGestureLibraryZ", // 896,3582,315,8
			"UWaveGestureLibraryAll", // 896,3582,945,8
			"Wafer", // 1000,6164,152,2
			"Wine",//54	57	234
                        "WordSynonyms", // 267,638,270,25
			"Worms", //77, 181,900,5
                        "WormsTwoClass",//77, 181,900,5
                        "Yoga" // 300,3000,426,2
                };   
      //</editor-fold>    

                

//<editor-fold defaultstate="collapsed" desc="five splits of the new 85 UCR datasets">    
		public static String[][] fiveSplits={	
      {			"Adiac",        // 390,391,176,37
			"ArrowHead",    // 36,175,251,3
			"Beef",         // 30,30,470,5
			"BeetleFly",    // 20,20,512,2
			"BirdChicken",  // 20,20,512,2
			"Car",          // 60,60,577,4
			"CBF",                      // 30,900,128,3
			"ChlorineConcentration",    // 467,3840,166,3
			"CinCECGTorso", // 40,1380,1639,4
			"Coffee", // 28,28,286,2
			"Computers", // 250,250,720,2
			"CricketX", // 390,390,300,12
			"CricketY", // 390,390,300,12
			"CricketZ", // 390,390,300,12
			"DiatomSizeReduction", // 16,306,345,4
			"DistalPhalanxOutlineCorrect", // 600,276,80,2
			"DistalPhalanxOutlineAgeGroup", // 400,139,80,3
			"DistalPhalanxTW", // 400,139,80,6
			"Earthquakes" // 322,139,512,2
      },
      {
                        "ECG200",   //100, 100, 96
                        "ECG5000",  //4500, 500,140
			"ECGFiveDays", // 23,861,136,2
			"FaceFour", // 24,88,350,4
			"FacesUCR", // 200,2050,131,14
			"FiftyWords", // 450,455,270,50
			"Fish", // 175,175,463,7
			"GunPoint", // 50,150,150,2
			"Ham",      //105,109,431
			"Haptics", // 155,308,1092,5
			"Herring", // 64,64,512,2
			"ItalyPowerDemand", // 67,1029,24,2
			"LargeKitchenAppliances", // 375,375,720,3
			"Lightning2", // 60,61,637,2
			"Lightning7", // 70,73,319,7
			"Mallat", // 55,2345,1024,8
			"Meat",//60,60,448
                        "MedicalImages", // 381,760,99,10
      },
      {
			"MiddlePhalanxOutlineCorrect", // 600,291,80,2
			"MiddlePhalanxOutlineAgeGroup", // 400,154,80,3
			"MiddlePhalanxTW", // 399,154,80,6
			"MoteStrain", // 20,1252,84,2
			"OliveOil", // 30,30,570,4
			"OSULeaf", // 200,242,427,6
			"Plane", // 105,105,144,7
			"ProximalPhalanxOutlineCorrect", // 600,291,80,2
			"ProximalPhalanxOutlineAgeGroup", // 400,205,80,3
			"ProximalPhalanxTW", // 400,205,80,6
			"RefrigerationDevices", // 375,375,720,3
			"ScreenType", // 375,375,720,3
			"ShapeletSim", // 20,180,500,2
			"SmallKitchenAppliances", // 375,375,720,3
			"SonyAIBORobotSurface1", // 20,601,70,2
			"SonyAIBORobotSurface2", // 27,953,65,2
			"Strawberry",//370,613,235
                        "SwedishLeaf", // 500,625,128,15
			"Symbols", // 25,995,398,6
			"SyntheticControl" // 300,300,60,6
      },
      {
			"ToeSegmentation1", // 40,228,277,2
			"ToeSegmentation2", // 36,130,343,2
			"Trace", // 100,100,275,4
			"TwoLeadECG", // 23,1139,82,2
			"TwoPatterns", // 1000,4000,128,4
			"UWaveGestureLibraryX", // 896,3582,315,8
			"UWaveGestureLibraryY", // 896,3582,315,8
			"UWaveGestureLibraryZ", // 896,3582,315,8
			"Wafer", // 1000,6164,152,2
			"Wine",//54	57	234
                        "WordSynonyms", // 267,638,270,25
			"Worms", //77, 181,900,5
                        "WormsTwoClass",//77, 181,900,5
                        "Yoga", // 300,3000,426,2
                        "InlineSkate", // 100,550,1882,7
                        "InsectWingbeatSound",//1980,220,256
			"FaceAll", // 560,1690,131,14
			"PhalangesOutlinesCorrect", // 1800,858,80,2
                        "Phoneme", //1896,214, 1024
			"ShapesAll", // 600,600,512,60
      },
      {
      			"ElectricDevices", // 8926,7711,96,7
			"FordA", // 3601,1320,500,2
			"FordB", // 3636,810,500,2
                        "HandOutlines", // 1000,370,2709,2
			"NonInvasiveFetalECGThorax1", // 1800,1965,750,42
			"NonInvasiveFetalECGThorax2", // 1800,1965,750,42
			"StarlightCurves", // 1000,8236,1024,3
			"UWaveGestureLibraryAll", // 896,3582,945,8
      }
                };   
      //</editor-fold>    
                
                
//UCR data sets
    //<editor-fold defaultstate="collapsed" desc="tscProblems46: 46 UCR Data sets">    
		public static String[] tscProblems46={	
			"Adiac", // 390,391,176,37
			"Beef", // 30,30,470,5
			"Car", // 60,60,577,4
			"CBF", // 30,900,128,3
			"ChlorineConcentration", // 467,3840,166,3
			"CinCECGTorso", // 40,1380,1639,4
			"Coffee", // 28,28,286,2
			"CricketX", // 390,390,300,12
			"CricketY", // 390,390,300,12
			"CricketZ", // 390,390,300,12
			"DiatomSizeReduction", // 16,306,345,4
			"ECGFiveDays", // 23,861,136,2
			"FaceAll", // 560,1690,131,14
			"FaceFour", // 24,88,350,4
			"FacesUCR", // 200,2050,131,14
			"FiftyWords", // 450,455,270,50
			"Fish", // 175,175,463,7
			"GunPoint", // 50,150,150,2
			"Haptics", // 155,308,1092,5
			"InlineSkate", // 100,550,1882,7
			"ItalyPowerDemand", // 67,1029,24,2
			"Lightning2", // 60,61,637,2
			"Lightning7", // 70,73,319,7
			"Mallat", // 55,2345,1024,8
                        "MedicalImages", // 381,760,99,10
			"MoteStrain", // 20,1252,84,2
			"NonInvasiveFetalECGThorax1", // 1800,1965,750,42
			"NonInvasiveFetalECGThorax2", // 1800,1965,750,42
			"OliveOil", // 30,30,570,4
			"OSULeaf", // 200,242,427,6
			"Plane", // 105,105,144,7
			"SonyAIBORobotSurface1", // 20,601,70,2
			"SonyAIBORobotSurface2", // 27,953,65,2
			"StarLightCurves", // 1000,8236,1024,3
                        "SwedishLeaf", // 500,625,128,15
			"Symbols", // 25,995,398,6
			"SyntheticControl", // 300,300,60,6
			"Trace", // 100,100,275,4
			"TwoLeadECG", // 23,1139,82,2
			"TwoPatterns", // 1000,4000,128,4
			"UWaveGestureLibraryX", // 896,3582,315,8
			"UWaveGestureLibraryY", // 896,3582,315,8
			"UWaveGestureLibraryZ", // 896,3582,315,8
			"Wafer", // 1000,6164,152,2
                        "WordSynonyms", // 267,638,270,25
                        "Yoga" // 300,3000,426,2
                };   
      //</editor-fold>

//Small UCR data sets
    //<editor-fold defaultstate="collapsed" desc="tscProblemsSmall: Small UCR Data sets">    
		public static String[] tscProblemsSmall={	
			"Beef", // 30,30,470,5
			"Car", // 60,60,577,4
			"Coffee", // 28,28,286,2
			"Cricket_X", // 390,390,300,12
			"Cricket_Y", // 390,390,300,12
			"Cricket_Z", // 390,390,300,12
			"DiatomSizeReduction", // 16,306,345,4
			"fish", // 175,175,463,7
			"GunPoint", // 50,150,150,2
			"ItalyPowerDemand", // 67,1029,24,2
			"MoteStrain", // 20,1252,84,2
			"OliveOil", // 30,30,570,4
			"Plane", // 105,105,144,7
			"SonyAIBORobotSurface", // 20,601,70,2
			"SonyAIBORobotSurfaceII", // 27,953,65,2
			"SyntheticControl", // 300,300,60,6
			"Trace", // 100,100,275,4
			"TwoLeadECG", // 23,1139,82,2
                };   
      //</editor-fold>

//<editor-fold defaultstate="collapsed" desc="spectral: Spectral data">    
		public static String[] spectral={	
//Train Size, Test Size, Series Length, Nos Classes
			"Beef", // 30,30,470,5
			"Coffee", // 28,28,286,2
			"Ham",
			"Meat",
			"OliveOil", // 30,30,570,4
			"Strawberry",
			"Wine",
//To add: spirits                        
                };
      //</editor-fold>
                
                
  //Small Files  
    //<editor-fold defaultstate="collapsed" desc="smallTSCProblems:">    
		public static String[] smallTSCProblems={	
                    "Beef","BeetleFly","BirdChicken","FaceFour","Plane","FacesUCR"};

/*//Train Size, Test Size, Series Length, Nos Classes
			"Adiac", // 390,391,176,37
			"ArrowHead", // 36,175,251,3
			"Beef", // 30,30,470,5
			"BeetleFly", // 20,20,512,2
			"BirdChicken", // 20,20,512,2
			"Car", // 60,60,577,4
			"CBF", // 30,900,128,3
			"ChlorineConcentration", // 467,3840,166,3
			"CinC_ECG_torso", // 40,1380,1639,4
			"Computers", // 250,250,720,2
			"Cricket_X", // 390,390,300,12
			"Cricket_Y", // 390,390,300,12
			"Cricket_Z", // 390,390,300,12
			"DiatomSizeReduction", // 16,306,345,4
			"DistalPhalanxOutlineAgeGroup", // 400,139,80,3
			"DistalPhalanxTW", // 400,139,80,6
			"Earthquakes", // 322,139,512,2
			"ECGFiveDays", // 23,861,136,2
			"ElectricDevices", // 8926,7711,96,7
			"FaceAll", // 560,1690,131,14
			"FacesUCR", // 200,2050,131,14
			"fiftywords", // 450,455,270,50
			"fish", // 175,175,463,7
			"FordA", // 3601,1320,500,2
			"FordB", // 3636,810,500,2
			"GunPoint", // 50,150,150,2
			"Ham",
                        "HandOutlines", // 1000,370,2709,2
			"Haptics", // 155,308,1092,5
			"Herring", // 64,64,512,2
			"InlineSkate", // 100,550,1882,7
			"ItalyPowerDemand", // 67,1029,24,2
			"LargeKitchenAppliances", // 375,375,720,3
			"Lightning2", // 60,61,637,2
			"Lightning7", // 70,73,319,7
			"MALLAT", // 55,2345,1024,8
//			"Meat",
			"MiddlePhalanxOutlineAgeGroup", // 400,154,80,3
			"MiddlePhalanxTW", // 399,154,80,6
			"MoteStrain", // 20,1252,84,2
			"NonInvasiveFatalECG_Thorax1", // 1800,1965,750,42
			"NonInvasiveFatalECG_Thorax2", // 1800,1965,750,42
			"OSULeaf", // 200,242,427,6
			"PhalangesOutlinesCorrect", // 1800,858,80,2
			"Plane", // 105,105,144,7
			"ProximalPhalanxOutlineAgeGroup", // 400,205,80,3
			"ProximalPhalanxTW", // 400,205,80,6
			"RefrigerationDevices", // 375,375,720,3
			"ScreenType", // 375,375,720,3
//			"ShapeletSim", // 20,180,500,2
			"ShapesAll", // 600,600,512,60
			"SmallKitchenAppliances", // 375,375,720,3
			"SonyAIBORobotSurfaceII", // 27,953,65,2
			"StarLightCurves", // 1000,8236,1024,3
			"Strawberry",
			"Symbols", // 25,995,398,6
			"TwoLeadECG", // 23,1139,82,2
			"TwoPatterns", // 1000,4000,128,4
			"UWaveGestureLibrary_X", // 896,3582,315,8
			"UWaveGestureLibrary_Y", // 896,3582,315,8
			"UWaveGestureLibrary_Z", // 896,3582,315,8
			"UWaveGestureLibraryAll", // 896,3582,945,8
			"wafer", // 1000,6164,152,2
//			"Wine",
                        "WordSynonyms", // 267,638,270,25
			"Worms",
                        "WormsTwoClass",
                        "yoga" // 300,3000,426,2
                };  */ 
      //</editor-fold>    

 //Large Problems  
    //<editor-fold defaultstate="collapsed" desc="largProblems:">    
    public static String[] largeProblems={	
"HeartbeatBIDMC","MNIST",
//"CambridgeMEG","KaggleMEG",        
    };
                
//Sets used in papers                
                
//<editor-fold defaultstate="collapsed" desc="rakthanmanon13fastshapelets">             
                /* Problem sets used in @article{rakthanmanon2013fast,
  title={Fast Shapelets: A Scalable Algorithm for Discovering Time Series Shapelets},
  author={Rakthanmanon, T. and Keogh, E.},
  journal={Proceedings of the 13th {SIAM} International Conference on Data Mining},
  year={2013}
}
All included except Cricket. There are three criket problems and they are not 
* alligned, the class values in the test set dont match

*/
		public static String[] fastShapeletProblems={	
			"ItalyPowerDemand", // 67,1029,24,2
			"MoteStrain", // 20,1252,84,2
			"SonyAIBORobotSurfaceII", // 27,953,65,2
			"SonyAIBORobotSurface", // 20,601,70,2
			"Beef", // 30,30,470,5
			"GunPoint", // 50,150,150,2
			"TwoLeadECG", // 23,1139,82,2
                        "Adiac", // 390,391,176,37
			"CBF", // 30,900,128,3
			"ChlorineConcentration", // 467,3840,166,3
			"Coffee", // 28,28,286,2
			"DiatomSizeReduction", // 16,306,345,4
			"ECGFiveDays", // 23,861,136,2
			"FaceFour", // 24,88,350,4
			"FacesUCR", // 200,2050,131,14
			"fish", // 175,175,463,7
			"Lighting2", // 60,61,637,2
			"Lighting7", // 70,73,319,7
			"FaceAll", // 560,1690,131,14
			"MALLAT", // 55,2345,1024,8
			"MedicalImages", // 381,760,99,10
			"OliveOil", // 30,30,570,4
			"OSULeaf", // 200,242,427,6
			"SwedishLeaf", // 500,625,128,15
			"Symbols", // 25,995,398,6
			"SyntheticControl", // 300,300,60,6
			"Trace", // 100,100,275,4
			"wafer", // 1000,6164,152,2
                        "yoga",
                        "FaceAll",
                        "TwoPatterns",
        		"CinC_ECG_torso" // 40,1380,1639,4
                };
//</editor-fold>
  
   
    //<editor-fold defaultstate="collapsed" desc="marteau09stiffness: TWED">                 
  		public static String[] marteau09stiffness={
			"SyntheticControl", // 300,300,60,6
			"GunPoint", // 50,150,150,2
			"CBF", // 30,900,128,3
			"FaceAll", // 560,1690,131,14
			"OSULeaf", // 200,242,427,6
			"SwedishLeaf", // 500,625,128,15
			"fiftywords", // 450,455,270,50
			"Trace", // 100,100,275,4
			"TwoPatterns", // 1000,4000,128,4
			"wafer", // 1000,6164,152,2
			"FaceFour", // 24,88,350,4
			"Lightning2", // 60,61,637,2
			"Lightning7", // 70,73,319,7
			"ECG200", // 100,100,96,2
			"Adiac", // 390,391,176,37
			"yoga", // 300,3000,426,2
			"fish", // 175,175,463,7
			"Coffee", // 28,28,286,2
			"OliveOil", // 30,30,570,4
			"Beef" // 30,30,470,5
                };  
                //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="stefan13movesplit: Move-Split-Merge">                 
  		public static String[] stefan13movesplit={
			"Coffee", // 28,28,286,2
			"CBF", // 30,900,128,3
			"ECG200", // 100,100,96,2
			"SyntheticControl", // 300,300,60,6
			"GunPoint", // 50,150,150,2
			"FaceFour", // 24,88,350,4
			"Lightning7", // 70,73,319,7
			"Trace", // 100,100,275,4
			"Adiac", // 390,391,176,37
			"Beef", // 30,30,470,5
			"Lightning2", // 60,61,637,2
			"OliveOil", // 30,30,570,4
                        "OSULeaf", // 200,242,427,6
			"SwedishLeaf", // 500,625,128,15
			"fish", // 175,175,463,7
                        "FaceAll", // 560,1690,131,14
			"fiftywords", // 450,455,270,50
			"TwoPatterns", // 1000,4000,128,4
			"wafer", // 1000,6164,152,2
			"yoga" // 300,3000,426,2
                };  
                //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="lines14elasticensemble">    
    public static String[] datasetsForDAMI2014_Lines={	
                //Train Size, Test Size, Series Length, Nos Classes
        "Adiac", // 390,391,176,37
        "ArrowHead", // 36,175,251,3
        "Beef", // 30,30,470,5
        "BeetleFly", // 20,20,512,2
        "BirdChicken", // 20,20,512,2
        "Car", // 60,60,577,4
        "CBF", // 30,900,128,3
        "ChlorineConcentration", // 467,3840,166,3
        "CinC_ECG_torso", // 40,1380,1639,4
        "Coffee", // 28,28,286,2
        "Computers", // 250,250,720,2
        "Cricket_X", // 390,390,300,12
        "Cricket_Y", // 390,390,300,12
        "Cricket_Z", // 390,390,300,12
        "DiatomSizeReduction", // 16,306,345,4
        "DistalPhalanxOutlineCorrect", // 600,276,80,2
        "DistalPhalanxOutlineAgeGroup", // 400,139,80,3
        "DistalPhalanxTW", // 400,139,80,6
        "Earthquakes", // 322,139,512,2
        "ECGFiveDays", // 23,861,136,2
        "ElectricDevices", // 8926,7711,96,7
        "FaceAll", // 560,1690,131,14
        "FaceFour", // 24,88,350,4
        "FacesUCR", // 200,2050,131,14
        "fiftywords", // 450,455,270,50
        "fish", // 175,175,463,7
        "FordA", // 3601,1320,500,2
        "FordB", // 3636,810,500,2
        "GunPoint", // 50,150,150,2
        "HandOutlines", // 1000,370,2709,2
        "Haptics", // 155,308,1092,5
        "Herring", // 64,64,512,2
        "InlineSkate", // 100,550,1882,7
        "ItalyPowerDemand", // 67,1029,24,2
        "LargeKitchenAppliances", // 375,375,720,3
        "Lightning2", // 60,61,637,2
        "Lightning7", // 70,73,319,7
        "MALLAT", // 55,2345,1024,8
        "MedicalImages", // 381,760,99,10
        "MiddlePhalanxOutlineCorrect", // 600,291,80,2
        "MiddlePhalanxOutlineAgeGroup", // 400,154,80,3
        "MiddlePhalanxTW", // 399,154,80,6
        "MoteStrain", // 20,1252,84,2
        "NonInvasiveFetalECG_Thorax1", // 1800,1965,750,42
        "NonInvasiveFetalECG_Thorax2", // 1800,1965,750,42
        "OliveOil", // 30,30,570,4
        "OSULeaf", // 200,242,427,6
        "PhalangesOutlinesCorrect", // 1800,858,80,2
        "Plane", // 105,105,144,7
        "ProximalPhalanxOutlineCorrect", // 600,291,80,2
        "ProximalPhalanxOutlineAgeGroup", // 400,205,80,3
        "ProximalPhalanxTW", // 400,205,80,6
        "RefrigerationDevices", // 375,375,720,3
        "ScreenType", // 375,375,720,3
        "ShapeletSim", // 20,180,500,2
        "ShapesAll", // 600,600,512,60
        "SmallKitchenAppliances", // 375,375,720,3
        "SonyAIBORobotSurface", // 20,601,70,2
        "SonyAIBORobotSurfaceII", // 27,953,65,2
        "StarLightCurves", // 1000,8236,1024,3
        "SwedishLeaf", // 500,625,128,15
        "Symbols", // 25,995,398,6
        "SyntheticControl", // 300,300,60,6
        "ToeSegmentation1", // 40,228,277,2
        "ToeSegmentation2", // 36,130,343,2
        "Trace", // 100,100,275,4
        "TwoLeadECG", // 23,1139,82,2
        "TwoPatterns", // 1000,4000,128,4
        "UWaveGestureLibrary_X", // 896,3582,315,8
        "UWaveGestureLibrary_Y", // 896,3582,315,8
        "UWaveGestureLibrary_Z", // 896,3582,315,8
        "wafer", // 1000,6164,152,2
        "WordSynonyms", // 267,638,270,25
        "yoga" // 300,3000,426,2
    };   
      //</editor-fold>    
  

static int[] testSizes85={391,175,30,20,20,60,900,3840,1380,28,250,390,390,390,306,276,139,139,139,100,4500,861,7711,1690,88,2050,455,175,1320,810,150,105,370,308,64,550,1980,1029,375,61,73,2345,60,760,291,154,154,1252,1965,1965,30,242,858,1896,105,291,205,205,375,375,180,600,375,601,953,8236,370,625,995,300,228,130,100,1139,4000,3582,3582,3582,3582,6164,54,638,77,77,3000};                
                
//UCI Classification problems: NOTE THESE ARE -train NOT _TRAIN
//<editor-fold defaultstate="collapsed" desc="UCI Classification problems">                 
  public static String[] uciFileNames={             
                "abalone",
    "banana",
    "cancer",
    "clouds",
    "concentric",
    "diabetes",
    "ecoli",
    "german",
    "glass2",
    "glass6",
    "haberman",
    "heart",
    "ionosphere",
    "liver",
    "magic",
    "pendigitis",
    "phoneme",
    "ringnorm",
    "satimage",
     "segment",
     "sonar",
     "thyroid",
     "twonorm",
     "vehicle",
     "vowel",
     "waveform",
     "wdbc",
     "wins",
     "yeast"};
//</editor-fold>

  //Gavin data sets  
/*
banana	       
flare_solar  
splice     
transfusion
breast_cancer  
synthetic  
vertebra
image	    
spambase  
tiianic    
*/
  
    public static String[] UCIContinuousFileNames={"abalone","acute-inflammation","acute-nephritis","adult","annealing","arrhythmia","audiology-std","balance-scale","balloons","bank","blood","breast-cancer","breast-cancer-wisc","breast-cancer-wisc-diag","breast-cancer-wisc-prog","breast-tissue","car","cardiotocography-10clases","cardiotocography-3clases",
        "chess-krvk","chess-krvkp","congressional-voting","conn-bench-sonar-mines-rocks","conn-bench-vowel-deterding",
        "connect-4","contrac","credit-approval","cylinder-bands","dermatology","echocardiogram","ecoli","energy-y1","energy-y2","fertility","flags","glass","haberman-survival","hayes-roth","heart-cleveland","heart-hungarian","heart-switzerland","heart-va","hepatitis","hill-valley","horse-colic","ilpd-indian-liver","image-segmentation","ionosphere","iris","led-display","lenses","letter","libras","low-res-spect","lung-cancer","lymphography","magic","mammographic",
        "miniboone","molec-biol-promoter","molec-biol-splice","monks-1","monks-2","monks-3","mushroom","musk-1","musk-2","nursery","oocytes_merluccius_nucleus_4d","oocytes_merluccius_states_2f","oocytes_trisopterus_nucleus_2f","oocytes_trisopterus_states_5b","optical","ozone","page-blocks","parkinsons","pendigits","pima","pittsburg-bridges-MATERIAL","pittsburg-bridges-REL-L","pittsburg-bridges-SPAN","pittsburg-bridges-T-OR-D","pittsburg-bridges-TYPE","planning","plant-margin","plant-shape","plant-texture","post-operative","primary-tumor","ringnorm","seeds","semeion","soybean","spambase","spect","spectf","statlog-australian-credit","statlog-german-credit","statlog-heart","statlog-image","statlog-landsat","statlog-shuttle","statlog-vehicle","steel-plates","synthetic-control","teaching","thyroid","tic-tac-toe","titanic","trains","twonorm","vertebral-column-2clases","vertebral-column-3clases","wall-following","waveform","waveform-noise","wine","wine-quality-red","wine-quality-white","yeast","zoo"};

    public static String[] UCIContinuousWithoutBigFour={"abalone","acute-inflammation","acute-nephritis","annealing","arrhythmia","audiology-std","balance-scale","balloons","bank","blood","breast-cancer","breast-cancer-wisc","breast-cancer-wisc-diag","breast-cancer-wisc-prog","breast-tissue","car","cardiotocography-10clases","cardiotocography-3clases",
        "chess-krvkp","congressional-voting","conn-bench-sonar-mines-rocks","conn-bench-vowel-deterding",
        "connect-4","contrac","credit-approval","cylinder-bands","dermatology","echocardiogram","ecoli","energy-y1","energy-y2","fertility","flags","glass","haberman-survival","hayes-roth","heart-cleveland","heart-hungarian","heart-switzerland","heart-va","hepatitis","hill-valley","horse-colic","ilpd-indian-liver","image-segmentation","ionosphere","iris","led-display","lenses","letter","libras","low-res-spect","lung-cancer","lymphography","mammographic",
        "molec-biol-promoter","molec-biol-splice","monks-1","monks-2","monks-3","mushroom","musk-1","musk-2","nursery","oocytes_merluccius_nucleus_4d","oocytes_merluccius_states_2f","oocytes_trisopterus_nucleus_2f","oocytes_trisopterus_states_5b","optical","ozone","page-blocks","parkinsons","pendigits","pima","pittsburg-bridges-MATERIAL","pittsburg-bridges-REL-L","pittsburg-bridges-SPAN","pittsburg-bridges-T-OR-D","pittsburg-bridges-TYPE","planning","plant-margin","plant-shape","plant-texture","post-operative","primary-tumor","ringnorm","seeds","semeion","soybean","spambase","spect","spectf","statlog-australian-credit","statlog-german-credit","statlog-heart","statlog-image","statlog-landsat","statlog-shuttle","statlog-vehicle","steel-plates","synthetic-control","teaching","thyroid","tic-tac-toe","titanic","trains","twonorm","vertebral-column-2clases","vertebral-column-3clases","wall-following","waveform","waveform-noise","wine","wine-quality-red","wine-quality-white","yeast","zoo"};

//Refactor when repo back 
    public static String[] ReducedUCI={"bank","blood","breast-cancer-wisc-diag",
        "breast-tissue","cardiotocography-10clases",
        "conn-bench-sonar-mines-rocks","conn-bench-vowel-deterding",
        "ecoli","glass","hill-valley",
        "image-segmentation","ionosphere","iris","libras","magic",
        "miniboone",
        "oocytes_merluccius_nucleus_4d","oocytes_trisopterus_states_5b",
        "optical","ozone","page-blocks","parkinsons","pendigits",
        "planning","post-operative","ringnorm","seeds","spambase",
            "statlog-landsat","statlog-shuttle","statlog-vehicle","steel-plates",
            "synthetic-control","twonorm","vertebral-column-3clases",
            "wall-following","waveform-noise","wine-quality-white","yeast"};
 
    
        public static String[] twoClassProblems2018={"BeetleFly","BirdChicken","Chinatown",
            "Coffee","Computers","DistalPhalanxOutlineCorrect","DodgerLoopGame",
            "DodgerLoopWeekend","Earthquakes","ECG200","ECGFiveDays","FordA","FordB",
            "FreezerRegularTrain","FreezerSmallTrain","GunPoint","GunPointAgeSpan",
            "GunPointMaleVersusFemale","GunPointOldVersusYoung","Ham","HandOutlines",
            "Herring","HouseTwenty","ItalyPowerDemand","Lightning2","MiddlePhalanxOutlineCorrect",
                "MoteStrain","PhalangesOutlinesCorrect","PowerCons","ProximalPhalanxOutlineCorrect",
                "SemgHandGenderCh2","ShapeletSim","SonyAIBORobotSurface1","SonyAIBORobotSurface2",
                "Strawberry","ToeSegmentation1","ToeSegmentation2","TwoLeadECG","Wafer","Wine",
                "WormsTwoClass","Yoga"};

public static String[] notNormalised={"ArrowHead","Beef","BeetleFly","BirdChicken","Coffee","Computers","Cricket_X","Cricket_Y","Cricket_Z","DistalPhalanxOutlineAgeGroup","DistalPhalanxOutlineCorrect","DistalPhalanxTW","ECG200","Earthquakes","ElectricDevices","FordA","FordB","Ham","Herring","LargeKitchenAppliances","Meat","MiddlePhalanxOutlineAgeGroup","MiddlePhalanxOutlineCorrect","MiddlePhalanxTW","OliveOil","PhalangesOutlinesCorrect","Plane","ProximalPhalanxOutlineAgeGroup","ProximalPhalanxOutlineCorrect","ProximalPhalanxTW","RefrigerationDevices","ScreenType","ShapeletSim","ShapesAll","SmallKitchenAppliances","Strawberry","ToeSegmentation1","ToeSegmentation2","UWaveGestureLibraryAll","UWaveGestureLibrary_Z","Wine","Worms","WormsTwoClass","fish"};

  public static void processUCRData(){
      System.out.println(" nos files ="+tscProblems46.length);
      String s;
      for(int str=39;str<43;str++){
          s=tscProblems46[str];
          InFile trainF= new InFile(problemPath+s+"/"+s+"_TRAIN");
          InFile testF= new InFile(problemPath+s+"/"+s+"_TEST");
          Instances train= DatasetLoading.loadDataNullable(problemPath+s+"/"+s+"_TRAIN");
          Instances test= DatasetLoading.loadDataNullable(problemPath+s+"/"+s+"_TEST");
          int trainSize=trainF.countLines();
          int testSize=testF.countLines();
          Attribute a=train.classAttribute();
          String tt=a.value(0);
          int first=Integer.parseInt(tt);
          System.out.println(s+" First value ="+tt+" first ="+first);
          if(trainSize!=train.numInstances() || testSize!=test.numInstances()){
              System.out.println(" ERROR MISMATCH SIZE TRAIN="+trainSize+","+train.numInstances()+" TEST ="+testSize+","+test.numInstances());
              System.exit(0);
          }
          trainF= new InFile(problemPath+s+"/"+s+"_TRAIN");
          testF= new InFile(problemPath+s+"/"+s+"_TEST");
          File dir = new File(problemPath+s);
          if(!dir.exists()){
              dir.mkdir();
          }
          OutFile newTrain = new OutFile(problemPath+s+"/"+s+"_TRAIN.arff");
          OutFile newTest = new OutFile(problemPath+s+"/"+s+"_TEST.arff");
          Instances header = new Instances(train,0);
          newTrain.writeLine(header.toString());
          newTest.writeLine(header.toString());
          for(int i=0;i<trainSize;i++){
                String line=trainF.readLine();
                line=line.trim();
                String[] split=line.split("/s+");
              try{
//                System.out.println(split[0]+"First ="+split[1]+" last ="+split[split.length-1]+" length = "+split.length+" nos atts "+train.numAttributes());
                double c=Double.valueOf(split[0]);
                if((int)(c-1)!=(int)train.instance(i).classValue() && (int)(c)!=(int)train.instance(i).classValue()&&(int)(c+1)!=(int)train.instance(i).classValue()){
                  System.out.println(" ERROR MISMATCH IN CLASS "+s+" from instance "+i+" ucr ="+(int)c+" mine ="+(int)train.instance(i).classValue());
                  System.exit(0);
                }
                for(int j=1;j<train.numAttributes();j++){
                    double v=Double.valueOf(split[j]);
                    newTrain.writeString(v+",");
                }
                if(first<=0)
                    newTrain.writeString((int)train.instance(i).classValue()+"\n");
                else
                    newTrain.writeString((int)(train.instance(i).classValue()+1)+"\n");
                    
              }catch(Exception e){
               System.out.println("Error problem "+s+" instance ="+i+" length ="+split.length+" val ="+split[0]);
               System.exit(0);
                  
              }
          }
          for(int i=0;i<testSize;i++){
                String line=testF.readLine();
                line=line.trim();
                String[] split=line.split("/s+");
              try{
//                System.out.println(split[0]+"First ="+split[1]+" last ="+split[split.length-1]+" length = "+split.length+" nos atts "+train.numAttributes());
                double c=Double.valueOf(split[0]);
                if((int)(c-1)!=(int)test.instance(i).classValue() && (int)(c)!=(int)test.instance(i).classValue()&&(int)(c+1)!=(int)test.instance(i).classValue()){
                  System.out.println(" ERROR MISMATCH IN CLASS "+s+" from instance "+i+" ucr ="+(int)c+" mine ="+(int)test.instance(i).classValue());
                  System.exit(0);
                }
                for(int j=1;j<test.numAttributes();j++){
                    double v=Double.valueOf(split[j]);
                    newTest.writeString(v+",");
                }
                if(first<=0)
                    newTest.writeString((int)test.instance(i).classValue()+"\n");
                else
                    newTest.writeString((int)(test.instance(i).classValue()+1)+"\n");
                    
              }catch(Exception e){
               System.out.println("Error problem "+s+" instance ="+i+" length ="+split.length+" val ="+split[0]);
               System.exit(0);
                  
              }
          }

          
      }
      
      
  }
  
  
  public static void listNotNormalisedList(String[] fileNames) throws Exception{
    TreeSet<String> notNormed=new TreeSet<>();
    DecimalFormat df = new DecimalFormat("###.######");
    for(String s:fileNames){
//Load test train
        Instances train=DatasetLoading.loadDataNullable(problemPath+s+"/"+s+"_TRAIN");
        Instances test=DatasetLoading.loadDataNullable(problemPath+s+"/"+s+"_TEST");
//Find summary 
        SummaryStats ss= new SummaryStats();
        train=ss.process(train);
        test=ss.process(test);
        int i=1;
        for(Instance ins:train){
            double stdev=ins.value(1)*ins.value(1);
//            stdev*=train.numAttributes()-1/(train.numAttributes()-2);
            if(Math.abs(ins.value(0))>0.01 || Math.abs(1-stdev)>0.01){
                System.out.println(" Not normalised train series ="+s+" index "+i+" mean = "+df.format(ins.value(0))+" var ="+df.format(stdev));
                notNormed.add(s);
                break;
            }
        }
        for(Instance ins:test){
            double stdev=ins.value(1)*ins.value(1);
//            stdev*=train.numAttributes()-1/(train.numAttributes()-2);
            if(Math.abs(ins.value(0))>0.01 || Math.abs(1-stdev)>0.01){
                System.out.println(" Not normalised test series ="+s+" index "+i+" mean = "+df.format(ins.value(0))+" var ="+df.format(stdev));
                notNormed.add(s);
                break;
            }
        }
    }
    System.out.print("String[] notNormalised={");
    for(String s:notNormed)
                System.out.print("\""+s+"\",");
    System.out.println("}");
    System.out.println("TOTAL NOT NORMED ="+notNormed.size());

  }

public static void dataDescription(String[] fileNames){
    //Produce summary descriptions
    //dropboxPath=uciPath;
        OutFile f=new OutFile(problemPath+"DataDimensions.csv");
        MetaData[] all=new MetaData[fileNames.length];
        TreeSet<String> nm=new TreeSet<>();
        nm.addAll(Arrays.asList(notNormalised));     
        f.writeLine("Problem,TrainSize,TestSize,SeriesLength,NumClasses,Normalised,ClassCounts");
                
        for(int i=0;i<fileNames.length;i++){
            try{
                Instances test=DatasetLoading.loadDataNullable(problemPath+fileNames[i]+"/"+fileNames[i]+"_TEST");
                Instances train=DatasetLoading.loadDataNullable(problemPath+fileNames[i]+"/"+fileNames[i]+"_TRAIN");
                Instances allData =new Instances(test);
                for(int j=0;j<train.numInstances();j++)
                    allData.add(train.instance(j));
//                allData.randomize(new Random());
//                OutFile combo=new OutFile(problemPath+tscProblems85[i]+"/"+tscProblems85[i]+".arff");    
//                combo.writeString(allData.toString());
                boolean normalised=true;
                if(nm.contains(fileNames[i]))
                    normalised=false;
                int[] classCounts=new int[allData.numClasses()*2];
                for(Instance ins: train)
                    classCounts[(int)(ins.classValue())]++;
                for(Instance ins: test)
                    classCounts[allData.numClasses()+(int)(ins.classValue())]++;
                all[i]=new MetaData(fileNames[i],train.numInstances(),test.numInstances(),test.numAttributes()-1,test.numClasses(),classCounts,normalised);
                f.writeLine(all[i].toString());
                System.out.println(all[i].toString());
                }
            catch(Exception e){
                System.out.println(" ERRROR"+e);
            }
        }
        /*
        Arrays.sort(all);       
        f=new OutFile(problemPath+"DataDimensionsBySeriesLength.csv");
        for(MetaData m: all)
            f.writeLine(m.toString());
        Arrays.sort(all, new MetaData.CompareByTrain());       
        f=new OutFile(problemPath+"DataDimensionsByTrainSize.csv");
        for(MetaData m: all)
            f.writeLine(m.toString());
        Arrays.sort(all, new MetaData.CompareByClasses());       
        f=new OutFile(problemPath+"DataDimensionsByNosClasses.csv");
        for(MetaData m: all)
            f.writeLine(m.toString());
        Arrays.sort(all, new MetaData.CompareByTotalSize());       
        f=new OutFile(problemPath+"DataDimensionsByTotalSize.csv");
        for(MetaData m: all)
            f.writeLine(m.toString());
*/

}



public static void dataDescriptionDataNotSplit(String[] fileNames){
    //Produce summary descriptions
    //dropboxPath=uciPath;
        OutFile f=new OutFile(problemPath+"DataDimensions.csv");
        f.writeLine("problem,numinstances,numAttributes,numClasses,classDistribution");
        try{
            for(int i=0;i<fileNames.length;i++){
                Instances allData=DatasetLoading.loadDataNullable(problemPath+fileNames[i]+"/"+fileNames[i]);
//                allData.randomize(new Random());
//                OutFile combo=new OutFile(problemPath+tscProblems85[i]+"/"+tscProblems85[i]+".arff");    
//                combo.writeString(allData.toString());
                int[] classCounts=new int[allData.numClasses()];
                for(Instance ins: allData)
                    classCounts[(int)(ins.classValue())]++;
                f.writeString(fileNames[i]+","+allData.numInstances()+","+(allData.numAttributes()-1)+","+allData.numClasses());
                for(int c:classCounts)
                     f.writeString(","+(c/(double)allData.numInstances()));
                f.writeString("\n");
            }
        }catch(Exception e){
            System.out.println(" ERRROR"+e);
        }

}



public static void makeTable(String means, String stdDev,String outfile){
    InFile m=new InFile(means);
    InFile sd=new InFile(stdDev);
    int lines=m.countLines();
    m=new InFile(means);
    String s=m.readLine();
    int columns=s.split(",").length;
    m=new InFile(means);
    OutFile out=new OutFile(outfile);
    DecimalFormat meanF=new DecimalFormat(".###");
    DecimalFormat sdF=new DecimalFormat(".##");
    
}
public static void createReadmeFiles(String[] problems){
    String myFilePath="C:\\Users\\ajb\\Dropbox\\TSC Website\\DataDescriptions\\";
    String theirFilePath="Z:\\Data\\NewTSCProblems";
    int count=0;
    for(String str:problems){
//        System.out.println("Processing "+str);
        File header=new File(theirFilePath+str+"\\README.md");
        File txtHeader=new File(theirFilePath+str+"\\"+str+".txt");
 //           System.out.println("No text file ");
        if(header.exists()){//Copy to a text file
//            System.out.println("there is an md file ");
            InFile in=new InFile(theirFilePath+str+"\\README.md");
            OutFile out=new OutFile(theirFilePath+str+"\\"+str+".txt");
            String line=in.readLine();
            while(line!=null){
                out.writeLine(line);
                line=in.readLine();
            }
        }
        else{ //Copy from my files 
            File ff= new File(myFilePath+str+".txt");
            if(ff.exists()){
                InFile in=new InFile(myFilePath+str+".txt");
                OutFile out=new OutFile(theirFilePath+str+"\\"+str+".txt");
                String line=in.readLine();
                while(line!=null){
                    out.writeLine(line);
                    line=in.readLine();
                }
            }
            else
                System.out.println("No description for "+str);
        }
    }
        
}

public static void buildArffs(String[] problems){
    String header;
    InFile trainTxt,testTxt,hdr;
    OutFile trainArff,testArff;
    
    for(String str:problems){
        System.out.println("Making ARFF for "+str);
       trainTxt=new InFile(path+str+"\\"+str+"_TRAIN.txt");
       hdr=new InFile(path+str+"\\"+str+".txt");
       trainArff= new OutFile(path+str+"\\"+str+"_TRAIN.arff");
       testArff= new OutFile(path+str+"\\"+str+"_TEST.arff");
//Write header comments
       String line=hdr.readLine();
       while(line!=null){
           trainArff.writeLine("%"+line);
           testArff.writeLine("%"+line);
           line=hdr.readLine();
       }
//Write arff meta data
        trainArff.writeLine("@Relation "+str);
        testArff.writeLine("@Relation "+str);
//Read in the train data
        ArrayList<Integer> classValues= new ArrayList<>();
        ArrayList<Double[]> attributeValues= new ArrayList<>();
        line=trainTxt.readLine();
            Double[] data=null;
        while(line!=null){
            line=line.trim();
//            String[] aCase=line.split(",");
            String[] aCase=line.split("\\s+");
            for(int i=0;i<aCase.length;i++)
                aCase[i]=aCase[i].trim();
            classValues.add((int)Double.parseDouble(aCase[0]));
            data= new Double[aCase.length-1];
            for(int i=0;i<aCase.length-1;i++)
                data[i]=Double.parseDouble(aCase[i+1]);
            attributeValues.add(data);
            line=trainTxt.readLine();
        }
//Write all data in CSV format
        int numAtts=data.length;
        for(int i=1;i<=numAtts;i++){
            trainArff.writeLine("@attribute att"+i+" numeric");
            testArff.writeLine("@attribute att"+i+" numeric");
        }
        TreeSet<Integer> ts=new TreeSet(classValues);
        trainArff.writeString("@attribute  target {");
        testArff.writeString("@attribute  target {");
        int size=ts.size();
        int c=1;
        for(Integer in:ts){
            trainArff.writeString(in+"");
            testArff.writeString(in+"");
            if(c<size){
                trainArff.writeString(",");
                testArff.writeString(",");
                c++;
            }else{
                trainArff.writeLine("}");
                testArff.writeLine("}");
           }
        }
        trainArff.writeLine("\n@data");
        testArff.writeLine("\n@data");

        for(int i=0;i<attributeValues.size();i++){
            data=attributeValues.get(i);
            int classValue=classValues.get(i);
            for(int j=0;j<data.length;j++){
                if(Double.isNaN(data[j]))
                    trainArff.writeString("?,");
                else    
                    trainArff.writeString(data[j]+",");                
            }
            trainArff.writeLine(classValue+"");
        }
//Read in the test data
       testTxt=new InFile(path+str+"\\"+str+"_TEST.txt");

        System.out.println("Starting test");
        classValues= new ArrayList<>();
        attributeValues= new ArrayList<>();
        line=testTxt.readLine();
        line=line.trim();
        data=null;
        int cc=0;
        while(line!=null){
            line=line.trim();
//            String[] aCase=line.split(",");
            String[] aCase=line.split("\\s+");
            int classValue=(int)Double.parseDouble(aCase[0]);
            data= new Double[aCase.length-1];
            for(int i=0;i<aCase.length-1;i++)
                data[i]=Double.parseDouble(aCase[i+1]);
            for(int j=0;j<data.length;j++)
                if(Double.isNaN(data[j]))
                    testArff.writeString("?,");
                else    
                    testArff.writeString(data[j]+",");
                
            testArff.writeLine(classValue+"");
            line=testTxt.readLine();
        }
        
    }

    
}

public static void testArffs(String[] problems){
    String header;
    Instances train,test;
    
    for(String str:problems){
        System.out.println("Loading ARFF for "+str);
       train=DatasetLoading.loadDataNullable(path+str+"\\"+str+"_TRAIN.arff");
       test=DatasetLoading.loadDataNullable(path+str+"\\"+str+"_TEST.arff");
       Classifier c= new IBk();
       double acc = ClassifierTools.singleTrainTestSplitAccuracy(c, train, test);
        System.out.println(" 1NN acc on "+str +" = "+acc);
       
       
    }

    
}


public static void describeTextFiles(){
    String path="Z:\\Data\\NewTSCProblems\\";
    InFile train;
    InFile test;
    InFile text;
    int count=1;
    File dirList=new File(path);
    String[] fn = dirList.list();
    System.out.println("Number of problems  ="+fn.length);
    OutFile out = new OutFile(path+"DataDescription.csv");
    out.writeLine("Problem,TrainCases,TestCases,NumberClasses,FixedLength,ReadMe");
    for(String str:tscProblems2018){
 //       System.out.println("Testing problem "+str+" number "+count);
        train=new InFile(path+str+"\\"+str+"_TRAIN.txt");
        test=new InFile(path+str+"\\"+str+"_TEST.txt");
            
        int trainCases=train.countLines();
        int testCases=test.countLines();
  //      System.out.println(str+" train cases ="+trainCases+" test cases = "+testCases);
        
        train=new InFile(path+str+"\\"+str+"_TRAIN.txt");
        test=new InFile(path+str+"\\"+str+"_TEST.txt");
            int attributes=0;
        for(int i=0;i<trainCases;i++){
            String[] line = train.readLine().split("\\s+");
 //                   System.out.println(" line "+count+" length in train ="+str+" = "+line.length);
            
            if(i==0){
                attributes=line.length;
//                    System.out.println("First line length in train ="+str+" = "+line.length);
            }
            else{
                if(attributes!=line.length){
                    System.out.println("VARIABLE LENGTH PROBLEM IN TRAIN ="+str+" first line ="+attributes+" current line ="+line.length);
                    }
            }
            
        }
        attributes=0;
        for(int i=0;i<testCases;i++){
            String[] line = test.readLine().split("\\s+");
            
            if(i==0)
                attributes=line.length;
            else{
                if(attributes!=line.length){
                 System.out.println("VARIABLE LENGTH PROBLEM IN TEST ="+str+" first line ="+attributes+" current line ="+line.length);
                }
            }
            
        }

        out.writeString(str+","+trainCases+","+testCases+",,,");
        File f=new File(path+str+"\\README.md");
        if(f.exists())
            out.writeLine("true");
       else
            out.writeLine("false");
            
        count++;
//        if(count==3) 
//            break;

    }
    
}
   public static void pack(String sourceDirPath, String zipFilePath) throws IOException {
    Path p = Files.createFile(Paths.get(zipFilePath));
    try (ZipOutputStream zs = new ZipOutputStream(Files.newOutputStream(p))) {
        Path pp = Paths.get(sourceDirPath);
        Files.walk(pp)
          .filter(path -> !Files.isDirectory(path))
          .forEach(path -> {
              ZipEntry zipEntry = new ZipEntry(pp.relativize(path).toString());
              try {
                  zs.putNextEntry(zipEntry);
                  Files.copy(path, zs);
                  zs.closeEntry();
            } catch (IOException e) {
                System.err.println(e);
            }
          });
    }
 }

   public static void packAll(String sourceDirPath, String zipDirPath, String[] files) throws IOException{
       for(String str:files){
            File src=new File(sourceDirPath+str);
            if(src.exists()){
                File zip=new File(zipDirPath+str+".zip");
                if(!zip.exists()){
                    System.out.println("Packing "+sourceDirPath+str+" to "+zipDirPath+str+".zip");
                    pack(sourceDirPath+str,zipDirPath+str+".zip");
                }
                else{
                    System.out.println(sourceDirPath+str+" to "+zipDirPath+str+".zip already exists");
                    
                }
            }
            else
                System.out.println("Directory "+sourceDirPath+str+" does not exist");
       }
       
       
   }
   public static void makeUploadTable(){
//  Dataset_id,Dataset,Donator1,Donator2,Train_size,Test_size,Length,
//Number_of_classes,Type,Best_algorithm,Best_acc,Original_source,Paper_first_used,Image,Description
//First_link,Second_link,First_used_TSC,Timestamp,Multivariate Flag, Dimension   
       
   }
   public static boolean hasMissing(String file){
       for(String str:variableLength2018Problems)
           if(str.equals(file)) return true;
       return false;
           
   }
   public static void makeUpLoadFile(String dest, String source){
       OutFile of = new OutFile(dest);
       InFile inf=new InFile(source);
       String line=inf.readLine();
       while(line!=null){
           String[] split=line.split(",");
           for(int i=0;i<split.length-1;i++)
               of.writeString("\""+split[i]+"\",");
           of.writeLine("\""+split[split.length-1]+"\"");         
           line=inf.readLine();
       }
   }
   
   
public static void main(String[] args) throws Exception{
    problemPath="E:\\Data\\ConcatenatedMTSC\\";
    dataDescription(mtscProblems2018);
    System.exit(0);
    path="E:\\Data\\TSCProblems2018\\";
    makeUpLoadFile("Z:\\Data\\MultivariateTSCProblems\\formattedUpload.csv","Z:\\Data\\MultivariateTSCProblems\\upload.csv");
    OutFile of = new OutFile("C:\\temp\\TSCNoMissing.txt");
        for(String str:tscProblems2018){
            if(!hasMissing(str))
                of.writeLine(str);
        }
    String zipPath="Z:\\Data\\TSCProblems2018_Zips\\";
    
    
    String[] test={"Adiac"};
//    packAll(path,zipPath,tscProblems2018);
//    testArffs(tscProblems2018);
//    pack("Z:\\Data\\NewTSCProblems\\Car","c:\\temp\\car.zip");
//    path="C:\\New TSC Data\\UCR_archive_2018_to_release\\";
    buildArffs(test);
//    buildArffs(tscProblems2018);
//    createReadmeFiles(tscProblems2018);
//    describeTextFiles();
    
  
//    dataDescription(uciFileNames);
/*    for(String s:uciFileNames){
        Instances train =ClassifierTools.loadDataThrowable(uciPath+s+"\\"+s+"-train");
        Instances test =ClassifierTools.loadDataThrowable(uciPath+s+"\\"+s+"-test");
        System.out.println(s);
    }
 */   
}
   public static class MetaData implements Comparable<MetaData>{
        String fileName;
        int trainSetSize;
        int testSetSize;
        int seriesLength;
        int nosClasses;
        int[] classDistribution;
        boolean normalised=true;
        public MetaData(String n, int t1, int t2, int s, int c, int[] dist,boolean norm){
            fileName=n;
            trainSetSize=t1;
            testSetSize=t2;
            seriesLength=s;
            nosClasses=c;
            classDistribution=dist;
            normalised=norm;
        }
        @Override
        public String toString(){
            String str= fileName+","+trainSetSize+","+testSetSize+","+seriesLength+","+nosClasses+","+normalised;
            for(int i:classDistribution)
                str+=","+i;
            return str;
        }
    @Override
        public int compareTo(MetaData o) {
                return seriesLength-o.seriesLength;
    }
    public static class CompareByTrain implements Comparator<MetaData>{
        @Override
        public int compare(MetaData a, MetaData b) {
            return a.trainSetSize-b.trainSetSize;
        }
    }
    public static class CompareByTrainSetSize implements Comparator<MetaData>{
        @Override
        public int compare(MetaData a, MetaData b) {
            return a.trainSetSize-b.trainSetSize;
        }
    }
    public static class CompareByClasses implements Comparator<MetaData>{
        @Override
        public int compare(MetaData a, MetaData b) {
            return a.nosClasses-b.nosClasses;
        }
    }
    public static class CompareByTotalSize implements Comparator<MetaData>{
        @Override
        public int compare(MetaData a, MetaData b) {
            return a.seriesLength*a.trainSetSize-b.seriesLength*b.trainSetSize;
        }
    }
}

}

