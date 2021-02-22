package local;

import experiments.Experiments;
import experiments.data.DatasetLists;

import java.util.ArrayList;

public class LocalExperiments {
    public static void main(String[] args) throws Exception {
        //even if all else fails, print the args as a sanity check for cluster.

        if (args.length > 0) {
            Experiments.ExperimentalArguments expSettings = new Experiments.ExperimentalArguments(args);
            Experiments.setupAndRunExperiment(expSettings);
        }
        else {//Manually set args
            int folds=1;

            /*
             * Change these settings for your experiment:
             */

            //Experiment Parameters, see
            String classifier="1NN-WDTW";//Classifier name: See ClassifierLists for valid options
            ArrayList<String> parameters= new ArrayList<>();
            parameters.add("-dp=Z:\\ArchiveData\\Univariate_arff\\"); //Where to get datasets
            parameters.add("-rp=Z:\\Results Working Area\\DistanceBased\\tsml\\"); //Where to write results
//            parameters.add("-dp=Z:\\temp\\"); //Where to get datasets
//            parameters.add("-rp=Z:\\temp\\"); //Where to write results
            parameters.add("-gtf=false"); //Whether to generate train files or not
            parameters.add("-cn="+classifier); //Classifier name
            parameters.add("-dn="); //Problem name, don't change here as it is overwritten by probFiles
            parameters.add("-f=1"); //Fold number (fold number 1 is stored as testFold0.csv, its a cluster thing)
            parameters.add("-d=true"); //Debugging
            parameters.add("--force=true"); //Overwrites existing results if true, otherwise set to false

            String[] settings=new String[parameters.size()];
            int count=0;
            for(String str:parameters)
                settings[count++]=str;


//            String[] probFiles= univariate; //Problem name(s)
            String[] probFiles= {"ArrowHead"}; //Problem name(s)
//            String[] probFiles= DatasetLists.fixedLengthMultivariate;
            /*
             * END OF SETTINGS
             */
            System.out.println("Manually set args:");
            for (String str : settings)
                System.out.println("\t"+str);
            System.out.println("");

            boolean threaded=true;
            if (threaded) {
                Experiments.ExperimentalArguments expSettings = new Experiments.ExperimentalArguments(settings);
                System.out.println("Threaded experiment with "+expSettings);
//              setupAndRunMultipleExperimentsThreaded(expSettings, classifiers,probFiles,0,folds);
//                Experiments.setupAndRunMultipleExperimentsThreaded(expSettings, new String[]{classifier},probFiles,0,folds);
            }
            else {//Local run without args, mainly for debugging
                for (String prob:probFiles) {
                    settings[4]="-dn="+prob;

                    for(int i=1;i<=folds;i++) {
                        settings[5]="-f="+i;
                        Experiments.ExperimentalArguments expSettings = new Experiments.ExperimentalArguments(settings);
//                      System.out.println("Sequential experiment with "+expSettings);
                        Experiments.setupAndRunExperiment(expSettings);
                    }
                }
            }
        }
    }

    static String[] univariate = {
            "ACSF1",
            "Adiac",
            "ArrowHead",
            "Beef",
            "BeetleFly",
            "BirdChicken",
            "BME",
            "Car",
            "CBF",
            "ChlorineConcentration",
            "CinCECGTorso",
            "Coffee",
            "Computers",
            "CricketX",
            "CricketY",
            "CricketZ",
            "DiatomSizeReduction",
            "DistalPhalanxOutlineCorrect",
            "DistalPhalanxOutlineAgeGroup",
            "DistalPhalanxTW",
            "Earthquakes",
            "ECG200",
            "ECG5000",
            "ECGFiveDays",
            "FaceAll",
            "FaceFour",
            "FacesUCR",
            "FiftyWords",
            "Fish",
            "FreezerRegularTrain",
            "FreezerSmallTrain",
            "Fungi",
            "GunPoint",
            "GunPointAgeSpan",
            "GunPointMaleVersusFemale",
            "GunPointOldVersusYoung",
            "Ham",
            "Haptics",
            "Herring",
            "HouseTwenty",
            "InlineSkate",
            "InsectEPGRegularTrain",
            "InsectEPGSmallTrain",
            "InsectWingbeatSound",
            "ItalyPowerDemand",
            "LargeKitchenAppliances",
            "Lightning2",
            "Lightning7",
            "Mallat",
            "Meat",
            "MedicalImages",
            "MiddlePhalanxOutlineCorrect",
            "MiddlePhalanxOutlineAgeGroup",
            "MiddlePhalanxTW",
            "MixedShapesRegularTrain",
            "MixedShapesSmallTrain",
            "MoteStrain",
            "OliveOil",
            "OSULeaf",
            "PhalangesOutlinesCorrect",
            "Phoneme",
            "Plane",
            "PowerCons",
            "ProximalPhalanxOutlineCorrect",
            "ProximalPhalanxOutlineAgeGroup",
            "ProximalPhalanxTW",
            "RefrigerationDevices",
            "Rock",
            "ShapeletSim",
            "ShapesAll",
            "SmallKitchenAppliances",
            "SmoothSubspace",
            "SonyAIBORobotSurface1",
            "SonyAIBORobotSurface2",
            "Strawberry",
            "SwedishLeaf",
            "Symbols",
            "SyntheticControl",
            "ToeSegmentation1",
            "ToeSegmentation2",
            "Trace",
            "TwoLeadECG",
            "TwoPatterns",
            "UMD",
            "UWaveGestureLibraryX",
            "UWaveGestureLibraryY",
            "UWaveGestureLibraryZ",
            "Wafer",
            "Wine",
            "WordSynonyms",
            "Worms",
            "WormsTwoClass",
            "Yoga"};


}
