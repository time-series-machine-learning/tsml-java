package examples;

import experiments.Experiments;
import experiments.data.DatasetLoading;
import tsml.classifiers.hybrids.HIVE_COTE;
import utilities.ClassifierTools;
import weka.core.Instances;

import java.io.IOException;
import java.util.concurrent.TimeUnit;

public class HiveCote2Examples {

    public static void simpleBuild() throws Exception {
        System.out.println("Basic Usage: \n" +
                           "   - load a dataset\n" +
                           "   - build a default classifier\n" +
                           "   - make predictions");
        System.out.println("Current default for HIVE COTE is to use version 2.0:\n" +
                           "   STC, DrCIF, Arsenal and TDE");
        String problem="Chinatown";
        Instances train= DatasetLoading.loadData("src/main/java/experiments/data/tsc/" +
                problem + "/" + problem + "_TRAIN"); //load train data of Chinatown dataset
        Instances test= DatasetLoading.loadData("src/main/java/experiments/data/tsc/" +
                problem + "/" + problem + "_TEST"); //load test data of chinatown dataset
        HIVE_COTE hc2 = new HIVE_COTE();//Defaults to HC V2.0
        hc2.setDebug(true); //Verbose output
        hc2.buildClassifier(train);
        double acc = ClassifierTools.accuracy(test,hc2);
        System.out.println(" HC2 test accuracy on "+problem+" = "+acc);
    }


    public static void experimentClassBuild() throws Exception {
        System.out.println("The Experiments.java class located in src/main/java/experiments/Experiments.java is used to " +
                "standardise the output of classifiers to make post processing easier");
        System.out.println("\nExperiments.java can be configured through the use of command line arguments" +
                           "\n\tDocumentation for these arguments can be seen at the top of the ExperimentalArguments class in Experiments.java");
        System.out.println("Experiments.java creates a classifier based on a switch statement in src/main/java/experiments/ClassifierLists.java");
        System.out.println("\nThe classifier cannot be configured directly, as the purpose of this class is to run large scale comparative experiments");
        System.out.println("However bespoke options can be added to ClassifierLists.java.");
        System.out.println("\tThe option need to be added to the setBespokeClassifiers method and the option name needs " +
                "to be added to the 'bespoke' string array\n");
        //Setting up arguments
        String[] arguments = new String[9];
        arguments[0] = "-dp=src/main/java/experiments/data/tsc/"; //Data location
        arguments[1] = "-rp=C:/Temp/"; //Location of where to write results
        arguments[2] = "-gtf=false"; //Generate train files or not
        arguments[3] = "-cn=HC2"; //Classifier name. A list of valid classifier names can be found in ClassifierLists.java
        arguments[4] = "-dn=Chinatown"; //Dataset name
        arguments[5] = "-f=1"; //Fold number
        arguments[6] = "--force=true"; //Overwrites existing results if set to true, otherwise does not
        arguments[7] = "-ctr=0"; //No time contract
        arguments[8] = "-cp=0"; //No checkpointing

        Experiments.debug = true;
        System.out.println("Manually set args: ");
        for (String string : arguments){
            System.out.println("\t" + string);
        }
        System.out.println();

        Experiments.ExperimentalArguments exp = new Experiments.ExperimentalArguments(arguments);
        Experiments.setupAndRunExperiment(exp);

        System.out.println("The output of this will be stored in C:/Temp/HC2/Predictions/Chinatown/testFold0.csv");
    }
    public static void fromComponentBuild() {
//Build some components with train files using experiments

//Rebuild from file

    }


    public static void configuration() {
// Set up to debug print

// Switch version versions

// Set up to checkpoint

//Set up as threaded

//Build from any old classifiers

    }



    /**
     * Contracting restricts the build time. It should be noted
     * 1. Contracts are approximate. It is not possible with all classifier to exactly control the build time.
     * Points to note
     *      TDE: TDE always builds a bagged model, and if train estimates are required, uses out of bag estimates. This
     *      second stage can actually take a long time, since transform is needed for each out of bag case. We can do no
     *      more than esitmate how long this takes, and it may well go over
     *      STC: Contract time is split three ways: transform search, build final rotation forest model and build bagged
     *      model for estimates. The division is decided at the beginning of the build, and so is approximate
     *      CIF: If producing train estimates, the time is evenly split between full build and OOB build.
     *      Arsenal:
     * 2.The contract relates only to the training of the classifier. The test time can also be quite a large
     *   overhead, especially for TDE, and so the overall train/test build time may take considerably longer than the
     *   contracT
     * 3. Train estimate: HIVE_COTE can produce its own train estimates, which it does through a form of bagging
     * (CLARIFY WITH JAMES)
     * 4. Threaded. The HC2.0 version has limited threading: if threaded, each classifier is built in its own thread.
     * If threaded and contracted, the assumption is the contract time is available for each thread, so that the overall
     * build time will be approximately the contract time.
     *
     *
      */
    public static void contracting() throws Exception {
        //Example with a 1 hour sequential contract, each classifier gets approximately 15 mins
        Instances train = DatasetLoading.loadData("src/main/java/experiments/data/tsc/ArrowHead/ArrowHead_TRAIN");

        HIVE_COTE hc2 = new HIVE_COTE();

        System.out.println("HIVE COTE 2.0 is contractable");
        System.out.println("Contract time can be set via the use Contractable methods");

        //Set by minute, hour and day
        hc2.setMinuteLimit(1);
        hc2.setHourLimit(2);
        hc2.setDayLimit(3);
        //Set single unit limits
        hc2.setOneMinuteLimit();
        hc2.setOneHourLimit();
        hc2.setOneDayLimit();
        //Set by specifying TimeUnit
        hc2.setTrainTimeLimit(42, TimeUnit.MINUTES);
        hc2.setTrainTimeLimit(10,TimeUnit.SECONDS);

        System.out.println("The first example is a 1 hour sequential contract");
        System.out.println("Each classifier gets approximately 15 minutes train time");
        hc2.setHourLimit(1);
        hc2.setDebug(true);

        long time = System.nanoTime();
        hc2.buildClassifier(train);
        time = System.nanoTime() - time;
        System.out.println("\t\t Time elapsed = "+time/1000000000+" seconds");



        //Example with a 1 hour threaded contract, each classifier gets approximately 1 hour.
        HIVE_COTE hc2Threaded = new HIVE_COTE();
        System.out.println("HIVE COTE 2.0 can be threaded");
        System.out.println("The amount of threads can be specified");
        // set by number
        // hc2.enableMultiThreading(2);

        //with no argument it will allocate the number of available processors minus 1
        hc2Threaded.enableMultiThreading();

        System.out.println("The second example is a 1 hour threaded contract");
        System.out.println("Each classifier gets approximately 1 hour train time");
        hc2Threaded.setHourLimit(1);
        hc2Threaded.setDebug(true);

        time = System.nanoTime();
        hc2Threaded.buildClassifier(train);
        time = System.nanoTime() - time;
        System.out.println("\t\t Time elapsed = "+time/1000000000+" seconds");
    }


    public static void main(String[] args) throws Exception {
        //simpleBuild();
        experimentClassBuild();
        //contracting();
    }


}
