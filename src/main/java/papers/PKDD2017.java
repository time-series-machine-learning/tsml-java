/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package papers;

import timeseriesweka.classifiers.BOSS;
import timeseriesweka.classifiers.boss.BoTSWEnsemble;
import timeseriesweka.classifiers.boss.BoTSWEnsemble.BoTSW;
import timeseriesweka.classifiers.boss.BOSSSpatialPyramids;
import timeseriesweka.classifiers.boss.BOSSSpatialPyramids_BD;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import weka.core.Instances;


/**
 *
 * This code notionally reproduces the experimental results presented in 
 * 'From BOP to BOSS and Beyond: Time Series Classification with Dictionary Based Classifiers'
 * 
 * Note that is intended to give working examples, in reality the experiments were performed
 * in parallel over multiple months on a medium-sized cluster at UEA. 
 *      
 * @author James Large
 */
public class PKDD2017 {
    
    public static final boolean verbose = false; //extra printouts
    public static final int numResamples = 25;
    
    //<editor-fold defaultstate="collapsed" desc="allDatasets: The 77 datasets used">   
    public static String[] allDatasets={	
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
            "CinCECGtorso", // 40,1380,1639,4
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
            "Wafer", // 1000,6164,152,2
            "Wine",//54	57	234
            "WordSynonyms", // 267,638,270,25
            "Worms", //77, 181,900,5
            "WormsTwoClass",//77, 181,900,5
            "Yoga" // 300,3000,426,2
    }; 
    //</editor-fold>   

    public static String[] quickTestDataset = new String[] {  "ItalyPowerDemand" }; 
    
    public static String[] datasetsToRun; 
    
    public static void main(String [] args) throws Exception {
        
        datasetsToRun = quickTestDataset;
        
        reproduceBOSSResults();
        reproduceResultsBOTSW_HI();
        reproduceResultsBOTSW_BD();
        reproduceResultsBOSSSP_HI();
        reproduceResultsBOSSSP_BD();
    }
    
    public static void reproduceBOSSResults() throws Exception {
        
        for (String dset : datasetsToRun) {
            Instances train = ClassifierTools.loadData("C:\\TSC Problems\\"+dset+"\\"+dset+"_TRAIN.arff");
            Instances test = ClassifierTools.loadData("C:\\TSC Problems\\"+dset+"\\"+dset+"_TEST.arff");

            System.out.println("Starting BOSS on " + dset);

            BOSS boss = new BOSS();

            double [] accs = new double[numResamples];

            for(int i=0;i<numResamples;i++){
                Instances[] data=InstanceTools.resampleTrainAndTestInstances(train, test, i);

                boss.buildClassifier(data[0]);
                accs[i]= ClassifierTools.accuracy(data[1], boss);

                if (verbose) 
                    if (i==0)
                        System.out.print(accs[i]);
                    else 
                        System.out.print("," + accs[i]);
            }

            if (verbose)
                System.out.println("");
            
            double mean = 0;
            for(int i=0;i<numResamples;i++)
                mean += accs[i];
            mean/=numResamples;                
                
            System.out.println("BOSS on " + dset + " over " + numResamples + " folds: " + mean +"\n");
        }
    }
    
    public static void reproduceResultsBOTSW_HI() throws Exception {
        
        for (String dset : datasetsToRun) {
            Instances train = ClassifierTools.loadData("C:\\TSC Problems\\"+dset+"\\"+dset+"_TRAIN.arff");
            Instances test = ClassifierTools.loadData("C:\\TSC Problems\\"+dset+"\\"+dset+"_TEST.arff");

            System.out.println("Starting BOTSW_HI on " + dset);

            BoTSWEnsemble botsw = new BoTSWEnsemble();
            botsw.setDistanceFunction(BoTSW.DistFunction.HISTOGRAM_INTERSECTION);

            double [] accs = new double[numResamples];

            for(int i=0;i<numResamples;i++){
                Instances[] data=InstanceTools.resampleTrainAndTestInstances(train, test, i);

                botsw.buildClassifier(data[0]);
                accs[i]= ClassifierTools.accuracy(data[1], botsw);

                if (verbose) 
                    if (i==0)
                        System.out.print(accs[i]);
                    else 
                        System.out.print("," + accs[i]);
            }

            if (verbose)
                System.out.println("");

            double mean = 0;
            for(int i=0;i<numResamples;i++)
                mean += accs[i];
            mean/=numResamples;                

            System.out.println("BOTSW_HI on " + dset + " over " + numResamples + " folds: " + mean +"\n");
        }
    }
    
    public static void reproduceResultsBOTSW_BD() throws Exception {
        
        for (String dset : datasetsToRun) {
            Instances train = ClassifierTools.loadData("C:\\TSC Problems\\"+dset+"\\"+dset+"_TRAIN.arff");
            Instances test = ClassifierTools.loadData("C:\\TSC Problems\\"+dset+"\\"+dset+"_TEST.arff");

            System.out.println("Starting BOTSW_BD on " + dset);

            BoTSWEnsemble botsw = new BoTSWEnsemble();
            botsw.setDistanceFunction(BoTSW.DistFunction.BOSS_DISTANCE);

            double [] accs = new double[numResamples];

            for(int i=0;i<numResamples;i++){
                Instances[] data=InstanceTools.resampleTrainAndTestInstances(train, test, i);

                botsw.buildClassifier(data[0]);
                accs[i]= ClassifierTools.accuracy(data[1], botsw);

                if (verbose) 
                    if (i==0)
                        System.out.print(accs[i]);
                    else 
                        System.out.print("," + accs[i]);
            }

            if (verbose)
                System.out.println("");

            double mean = 0;
            for(int i=0;i<numResamples;i++)
                mean += accs[i];
            mean/=numResamples;                

            System.out.println("BOTSW_BD on " + dset + " over " + numResamples + " folds: " + mean +"\n");
        }
    }
    
    public static void reproduceResultsBOSSSP_HI() throws Exception {
               
        for (String dset : datasetsToRun) {
            Instances train = ClassifierTools.loadData("C:\\TSC Problems\\"+dset+"\\"+dset+"_TRAIN.arff");
            Instances test = ClassifierTools.loadData("C:\\TSC Problems\\"+dset+"\\"+dset+"_TEST.arff");

            System.out.println("Starting BOSSSP_HI on " + dset);

            BOSSSpatialPyramids bosssp = new BOSSSpatialPyramids();

            double [] accs = new double[numResamples];

            for(int i=0;i<numResamples;i++){
                Instances[] data=InstanceTools.resampleTrainAndTestInstances(train, test, i);

                bosssp.buildClassifier(data[0]);
                accs[i]= ClassifierTools.accuracy(data[1], bosssp);

                if (verbose) 
                    if (i==0)
                        System.out.print(accs[i]);
                    else 
                        System.out.print("," + accs[i]);
            }

            if (verbose)
                System.out.println("");

            double mean = 0;
            for(int i=0;i<numResamples;i++)
                mean += accs[i];
            mean/=numResamples;                

            System.out.println("BOSSSP_HI on " + dset + " over " + numResamples + " folds: " + mean +"\n");
        }
    }
    
    public static void reproduceResultsBOSSSP_BD() throws Exception {
               
        for (String dset : datasetsToRun) {
            Instances train = ClassifierTools.loadData("C:\\TSC Problems\\"+dset+"\\"+dset+"_TRAIN.arff");
            Instances test = ClassifierTools.loadData("C:\\TSC Problems\\"+dset+"\\"+dset+"_TEST.arff");

            System.out.println("Starting BOSSSP_BD on " + dset);

            BOSSSpatialPyramids_BD bosssp = new BOSSSpatialPyramids_BD();

            double [] accs = new double[numResamples];

            for(int i=0;i<numResamples;i++){
                Instances[] data=InstanceTools.resampleTrainAndTestInstances(train, test, i);

                bosssp.buildClassifier(data[0]);
                accs[i]= ClassifierTools.accuracy(data[1], bosssp);

                if (verbose) 
                    if (i==0)
                        System.out.print(accs[i]);
                    else 
                        System.out.print("," + accs[i]);
            }

            if (verbose)
                System.out.println("");

            double mean = 0;
            for(int i=0;i<numResamples;i++)
                mean += accs[i];
            mean/=numResamples;                

            System.out.println("BOSSSP_BD on " + dset + " over " + numResamples + " folds: " + mean +"\n");
        }
    }
}
