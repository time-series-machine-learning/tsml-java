package tests;

import evaluation.storage.ClassifierResults;
import experiments.data.DatasetLoading;
import machine_learning.classifiers.ensembles.ContractRotationForest;
import machine_learning.classifiers.ensembles.EnhancedRotationForest;
import tsml.classifiers.Checkpointable;
import tsml.classifiers.EnhancedAbstractClassifier;
import tsml.classifiers.TrainTimeContractable;
import tsml.classifiers.dictionary_based.TDE;
import tsml.classifiers.distance_based.proximity.ProximityForest;
import tsml.classifiers.hybrids.Arsenal;
import tsml.classifiers.hybrids.HIVE_COTE;
import tsml.classifiers.interval_based.DrCIF;
import tsml.classifiers.shapelet_based.ShapeletTransformClassifier;
import utilities.ClassifierTools;
import weka.classifiers.meta.RotationForest;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Random;

/**
 * Simple development class to road test the classifiers in HiveCote 2, and HC2 itself
 *  HC2.0 is too slow on ArrowHead, must be STC taking an hour
 */
public class ClassifierSanityChecks {

    static String[] classifiers={//"DrCIF","TDE","PF","Arsenal",
            "STC"
//            ,"HiveCote"
            };
    static int[] arrowHeadCorrect={140,155,150,148,144,134,153};
    static EnhancedAbstractClassifier setClassifier(String str){
        //MNissing Arsenal and HC2

        EnhancedAbstractClassifier c = null;
        switch(str){
            case "DrCIF":
                c= new DrCIF();
                break;
            case "TDE":
                c= new TDE();
            break;
            case "PF":
                ProximityForest pf=new ProximityForest();
                pf.setSeed(new Random().nextInt());
                c=pf;
                break;
            case "STC":
                ShapeletTransformClassifier stc= new ShapeletTransformClassifier();
                stc.setMinuteLimit(1);
                stc.setDebug(true);
                c=stc;
                break;
            case "Arsenal":
                c = new Arsenal();
                break;
            case "HiveCote":
                HIVE_COTE hc = new HIVE_COTE();
                hc.setupHIVE_COTE_2_0();
                hc.enableMultiThreading(5);
                c=hc;
                break;
        }
        return c;
    }

    /**
     * To use before finalising tests
     */

    public static void basicUsageTest()throws Exception {
            String path="src/main/java/experiments/data/tsc/";
            String problem="ArrowHead";
            Instances train= DatasetLoading.loadData(path+problem+"/"+problem+"_TRAIN.arff");
            Instances test= DatasetLoading.loadData(path+problem+"/"+problem+"_TEST.arff");
            for(String str:classifiers) {
                EnhancedAbstractClassifier c = setClassifier(str);
                System.out.println(" running "+str);
                if(c!=null) {
                try {
                    long t1= System.nanoTime();
                    c.buildClassifier(train);
                    long t2= System.nanoTime();
                    long trainTime = (t2-t1)/1000000000;
                    int correct=0;
                    for(Instance ins:test){
                        double pred=c.classifyInstance(ins);
                        double[] d = c.distributionForInstance(ins);
                        if(pred==ins.classValue())
                            correct++;
                    }
                    System.out.println(str + " on " + problem + " train time = "+trainTime+" (secs), correct count = "+correct+" test acc = " + correct/(double)test.numInstances());
                }catch(Exception e){
                    System.out.println(" Error building classifier "+str+" exception = "+e);
                }
            } else{
                    System.out.println(" null classifier "+str);
                }
        }



    }


    public static void contractRotationForestTest()throws Exception
    {
 //       String path="src/main/java/experiments/data/tsc/";
//        String problem="";
        String path="Z:\\ArchiveData\\Univariate_arff\\";
        String problem="ChinaTown";

        Instances train= DatasetLoading.loadData(path+problem+"/"+problem+"_TRAIN.arff");
        Instances test= DatasetLoading.loadData(path+problem+"/"+problem+"_TEST.arff");
        long t1,t2, trainTime;
        int correct=0;

        EnhancedAbstractClassifier c = new ContractRotationForest();
        TrainTimeContractable x=( TrainTimeContractable) c;
        ((TrainTimeContractable) c).setMinuteLimit(3);
        t1= System.nanoTime();
        c.setDebug(true);
        c.setEstimateOwnPerformance(false);
        int count=0;
        ClassifierResults trainRes;
/*
        c.buildClassifier(train);
        t2= System.nanoTime();
        trainTime = (t2-t1)/1000000000;
        trainRes = c.getTrainResults();
        System.out.println(" CONTRACT Train Acc = "+trainRes.getAcc()+" results = "+trainRes);
        for(Instance ins:test){
            double pred=c.classifyInstance(ins);
            double[] d = c.distributionForInstance(ins);
            if(pred==ins.classValue())
                correct++;
            if(count<2) {
                for (double dd : d)
                    System.out.print(dd + ", ");
                System.out.println(" PREDICTION = " + pred+ " actual = "+ins.classValue());
            }
            count++;

        }
        System.out.println("\n CRF finished in "+trainTime+" secs, test num correct = "+correct+" acc = "+correct/(double)test.numInstances());
*/
        EnhancedAbstractClassifier c2 = new EnhancedRotationForest();
        t1= System.nanoTime();
        ((TrainTimeContractable) c2).setMinuteLimit(3);
        c2.setDebug(true);

        ((EnhancedRotationForest)c2).setRemovedPercentage(0);
        ((EnhancedRotationForest)c2).setBagging(false);
        c2.setEstimateOwnPerformance(true);
        c2.setTrainEstimateMethod("TRAIN");
        c2.buildClassifier(train);
        c2.setDebug(true);
        t2= System.nanoTime();
        trainTime = (t2-t1)/1000000000;
        correct=0;
       trainRes = c2.getTrainResults();
      System.out.println(" ENHANCED: Train Acc = "+trainRes.getAcc()+" results = "+trainRes);
/*
        trainPred=trainRes.getPredClassValsAsArray();
        trainProbs=trainRes.getProbabilityDistributionsAsArray();

        for(int i=0;i<trainPred.length;i++){
            System.out.print("\n actual = "+train.instance(i).classValue()+" predicted  = "+trainPred[i]+" probs = ");
            for(double d: trainProbs[i])
                System.out.print(d+",");
        }
*/
        count=0;
        correct=0;
        for(Instance ins:test){
            double pred=c2.classifyInstance(ins);
            double[] d = c2.distributionForInstance(ins);
            if(count<2) {
                for (double dd : d)
                    System.out.print(dd + ", ");
                System.out.println(" PREDICTION = " + pred+ " actual = "+ins.classValue());
            }
            count++;
            if(pred==ins.classValue())
                correct++;
        }
        System.out.println("\n ERF finished in "+trainTime+" secs, test num correct = "+correct+" acc = "+correct/(double)test.numInstances());
        RotationForest rotf1= new RotationForest();
        t1= System.nanoTime();
        rotf1.setNumIterations(200);
        rotf1.buildClassifier(train);
        t2= System.nanoTime();
        trainTime = (t2-t1)/1000000000;
        correct=count=0;
        for(Instance ins:test){
            double pred=rotf1.classifyInstance(ins);
            double[] d = rotf1.distributionForInstance(ins);
            if(pred==ins.classValue())
                correct++;
            if(count<2) {
                for (double dd : d)
                    System.out.print(dd + ", ");
                System.out.println(" PREDICTION = " + pred);
            }
            count++;
        }
        System.out.println("\n Default RotF finished in "+trainTime+" secs, test num correct = "+correct+" acc = "+correct/(double)test.numInstances());


    }



    public static void shortContractTest() throws Exception {
        String path = "src/main/java/experiments/data/tsc/";
        String problem = "Beef";

        Instances train = DatasetLoading.loadData(path + problem + "/" + problem + "_TRAIN.arff");
        Instances test = DatasetLoading.loadData(path + problem + "/" + problem + "_TEST.arff");
        for(String str:classifiers) {
            EnhancedAbstractClassifier c = setClassifier(str);
            System.out.println(" running "+str);
            if(c instanceof TrainTimeContractable) {
                ((TrainTimeContractable) c).setMinuteLimit(1);
                System.out.println("Set timer to 1 minute ");
            }
            else{
                System.out.println(" Classifier "+str+" is not TrainTimeContractable, skipping ");
                continue;
            }

            if(c!=null) {
                try {
                    long t1= System.nanoTime();
                    c.buildClassifier(train);
                    long t2= System.nanoTime();
                    long trainTime = (t2-t1)/1000000000;
                    int correct=0;
                    for(Instance ins:test){
                        double pred=c.classifyInstance(ins);
                        double[] d = c.distributionForInstance(ins);
                        if(pred==ins.classValue())
                            correct++;
                    }
                    System.out.println(str + " on " + problem + " train time = "+trainTime+" (secs), correct count = "+correct+" test acc = " + correct/(double)test.numInstances());
                }catch(Exception e){
                    System.out.println(" Error building classifier "+str+" exception = "+e);
                }
            } else{
                System.out.println(" null classifier "+str);
            }
        }


    }


    public static void mediumContractTest() throws Exception {
        String beastPath = "Z:/ArchiveData/Univariate_arff/";
        String problem = "ElectricDevices";

        Instances train = DatasetLoading.loadData(beastPath + problem + "/" + problem + "_TRAIN.arff");
        Instances test = DatasetLoading.loadData(beastPath + problem + "/" + problem + "_TEST.arff");
        System.out.println(" Problem  = "+problem);
        for(String str:classifiers) {
            EnhancedAbstractClassifier c = setClassifier(str);
            System.out.println(" running "+str);
            if(c instanceof TrainTimeContractable) {
                ((TrainTimeContractable) c).setHourLimit(1);
                System.out.println("Set timer to 1 hour ");
            }
            else{
                System.out.println(" Classifier "+str+" is not TrainTimeContractable, skipping ");
                continue;
            }

            if(c!=null) {
                try {
                    long t1= System.nanoTime();
                    c.buildClassifier(train);
                    long t2= System.nanoTime();
                    long trainTime = (t2-t1)/1000000000;
                    System.out.print(str + " on " + problem + " train time = "+trainTime+" (secs),");
                    int correct=0;
                    for(Instance ins:test){
                        double pred=c.classifyInstance(ins);
                        double[] d = c.distributionForInstance(ins);
                        if(pred==ins.classValue())
                            correct++;
                    }
                    System.out.println(" correct count = "+correct+" test acc = " + correct/(double)test.numInstances());
                }catch(Exception e){
                    System.out.println(" Error building classifier "+str+" exception = "+e);
                }
            } else{
                System.out.println(" null classifier "+str);
            }
        }


    }
//Genera


    public static void checkPointTest() throws Exception {
        String path = "src/main/java/experiments/data/tsc/";
        String problem = "Beef";

        Instances train = DatasetLoading.loadData(path + problem + "/" + problem + "_TRAIN.arff");
        Instances test = DatasetLoading.loadData(path + problem + "/" + problem + "_TEST.arff");
        for(String str:classifiers) {
            EnhancedAbstractClassifier c = setClassifier(str);
            System.out.println(" running "+str);
            if(c instanceof Checkpointable) {
                ((Checkpointable) c).setCheckpointPath("");
                ((Checkpointable) c).setCheckpointTimeHours(1);
                System.out.println("Set timer to 1 minute ");
            }
            else{
                System.out.println(" Classifier "+str+" is not TrainTimeContractable, skipping ");
                continue;
            }

            if(c!=null) {
                try {
                    long t1= System.nanoTime();
                    c.buildClassifier(train);
                    long t2= System.nanoTime();
                    long trainTime = (t2-t1)/1000000000;
                    int correct=0;
                    for(Instance ins:test){
                        double pred=c.classifyInstance(ins);
                        double[] d = c.distributionForInstance(ins);
                        if(pred==ins.classValue())
                            correct++;
                    }
                    System.out.println(str + " on " + problem + " train time = "+trainTime+" (secs), correct count = "+correct+" test acc = " + correct/(double)test.numInstances());
                }catch(Exception e){
                    System.out.println(" Error building classifier "+str+" exception = "+e);
                }
            } else{
                System.out.println(" null classifier "+str);
            }
        }


    }


    public static void hackFile() throws Exception {
        String path="src/main/java/experiments/data/tsc/";
        String problem="Beef";

        Instances train= DatasetLoading.loadData(path+problem+"/"+problem+"_TRAIN.arff");
        Instances test= DatasetLoading.loadData(path+problem+"/"+problem+"_TEST.arff");
        EnhancedAbstractClassifier c = new TDE();
        if(c instanceof TrainTimeContractable) {
            ((TrainTimeContractable) c).setMinuteLimit(1);
            System.out.println("Set timer to 1 minute ");
        }
        long t1= System.nanoTime();
        c.buildClassifier(train);
        long t2= System.nanoTime();
        long secs = (t2-t1)/1000000000;
        double acc = ClassifierTools.accuracy(test, c);
        System.out.println("TDE on 1 minute timer " + problem + "  took  = "+secs+" seconds, test acc = " + acc);

        String locaPath="src/main/java/experiments/data/tsc/";
        problem="Beef";


        //1. Test basic load and build with defaults on ArrowHead: print acc and number correct

        //2. Test contract on 1 minute with Beef

        //2. Test contract on 1 hour on Elect

        // Test check pointing.


    }

    public static void main(String[] args) throws Exception {
//        basicUsageTest();
//        shortContractTest();
//        mediumContractTest();
        contractRotationForestTest();
    }

}
