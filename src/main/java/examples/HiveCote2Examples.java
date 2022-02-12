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
        String problem="Chinatown";
        Instances train= DatasetLoading.loadData("");
        Instances test= DatasetLoading.loadData("");
        HIVE_COTE hc2 = new HIVE_COTE();//Defaults to HC V2.0
        hc2.buildClassifier(train);
        double acc = ClassifierTools.accuracy(test,hc2);
        System.out.println(" HC2 test accuracy on "+problem+" = "+acc);
    }


    public static void experimentClassBuild() {
        Experiments.ExperimentalArguments exp;


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

        System.out.println("First example is a 1 hour sequential contract");
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

        System.out.println("Second example is a 1 hour threaded contract");
        System.out.println("Each classifier gets approximately 1 hour train time");
        hc2Threaded.setHourLimit(1);
        hc2Threaded.setDebug(true);

        time = System.nanoTime();
        hc2Threaded.buildClassifier(train);
        time = System.nanoTime() - time;
        System.out.println("\t\t Time elapsed = "+time/1000000000+" seconds");
    }


    public static void main(String[] args) throws Exception {
        contracting();
    }


}
