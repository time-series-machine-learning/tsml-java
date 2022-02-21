package ml6002b2022.week4_demo;

import evaluation.MultipleClassifierEvaluation;
import evaluation.storage.ClassifierResults;
import evaluation.tuning.ParameterSpace;
import experiments.Experiments;
import experiments.data.DatasetLists;
import experiments.data.DatasetLoading;
import fileIO.InFile;
import fileIO.OutFile;
import machine_learning.classifiers.tuned.TunedClassifier;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.*;
import weka.core.Instance;
import weka.core.Instances;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.Random;

public class ClassifierEvaluation {
        public static String basePath="C:\\Temp\\UCI\\";
        public static Classifier setClassifier(String c){
            switch(c){
                case "J48":
                    return new J48();
                case"SimpleCart":
                    return new SimpleCart();
                case "FT":
                    return new FT();
                case "HoeffdingTree":
                    return new HoeffdingTree();
                case "LADTree":
                    return new LADTree();
                case "NBTree":
                    return new NBTree();
                case "REPTree":
                    return new REPTree();
                case "DecisionStump":
                    return new DecisionStump();
                case "J48graft":
                    return new J48graft();
            }
            return null;

        }//,"trains"

        static String[] allProblems ={"blood", "hayes-roth","bank","balance-scale","monks-1","breast-cancer-wisc-diag"};


        static String[] allClassifiers={"J48","SimpleCart","FT","HoeffdingTree","LADTree","REPTree","DecisionStump","J48graft"};


        public static boolean isPig(String str){
            switch(str){
                case "miniboone":
                case "connect-4":
                case "statlog-shuttle":
                case "adult":
                case "chess-krvk":
                case "letter":
                case "magic":
                case "nursery":
                case "pendigits":
                case "mushroom":
                case "ringnorm":
                case "twonorm":
                case "thyroid":
                case "musk-2":
                case "statlog-landsat":
                case "optical":
                case "page-blocks":
                case "wall-following":
                case "waveform":
                case "waveform-noise":
                    return true;
            }
            return false;

        }

        public static void runExperimentManually() throws Exception {
            String problem = "optical";
            Instances data = DatasetLoading.loadData(basePath+problem+"/"+problem+".arff");
            System.out.println(data.numClasses());
            Instances[] split = InstanceTools.resampleInstances(data,0,0.5);
            split = DatasetLoading.sampleHayesRoth(0);
            Classifier c = new J48();
            c.buildClassifier(split[0]);
            OutFile out = new OutFile("C:/temp/"+problem+"Resample0.csv");
            out.writeLine(c.getClass().getSimpleName()+","+problem);
            out.writeLine("No parameter info");
            out.writeLine("Blank");
            for(Instance ins:split[1]){
                //Inefficient to call twice
                int pred = (int)c.classifyInstance(ins);
                double[] probs = c.distributionForInstance(ins);
                out.writeString((int)ins.classValue()+","+pred+",");
                System.out.print((int)ins.classValue()+","+pred+",");
                for(double d:probs) {
                    System.out.print("," + d);
                    out.writeString("," + d);
                }
                System.out.print("\n");
                out.writeString("\n");
            }
        }

        public static void runExperimentAutomatically() throws Exception {
            Experiments.ExperimentalArguments expSettings = new Experiments.ExperimentalArguments();
            expSettings.classifier = new J48();
            expSettings.dataReadLocation = basePath;
            expSettings.resultsWriteLocation = "c:/temp/";
            expSettings.classifierName = "J48";
            expSettings.datasetName = "bank";
            expSettings.forceEvaluation = true; // Overwrite existing results?
            expSettings.foldId = 10;  // note that since we're now setting the fold directly, we can resume zero-indexing
            expSettings.debug = true;
            //If splits are not defined, can set here, the default is 50/50 splits
            DatasetLoading.setProportionKeptForTraining(0.75);
//            Experiments.setupAndRunExperiment(expSettings);
            expSettings.run();

        }
        public static void evaluateInCode() throws Exception {
            String problem = "bank";
            Instances data = DatasetLoading.loadData(basePath+problem+"/"+problem+".arff");
            Instances[] split = InstanceTools.resampleInstances(data,0,0.5);
            Evaluation eval = new Evaluation(split[0]);
            Classifier c = new J48();
            c.buildClassifier(split[0]);
            eval.evaluateModel(c,split[1]);
            double acc =1-eval.errorRate();
            double weightedAuroc = eval.weightedAreaUnderROC();
            System.out.println(" Acc = "+acc+" auroc = "+weightedAuroc);
            eval.crossValidateModel(c,data,10,new Random());
            acc =1-eval.errorRate();
            weightedAuroc = eval.weightedAreaUnderROC();
//            weightedAuroc = eval.areaUnderROC(split[0].classIndex());
            System.out.println(" Acc = "+acc+" auroc = "+weightedAuroc);
        }

    public static void runMultipleExperiments() throws Exception {
        Experiments.ExperimentalArguments expSettings = new Experiments.ExperimentalArguments();
        Classifier[] cls= new Classifier[4];
        String[] names = {"C45","RandF","RandF500","TunedC45"};
        J48 c45 = new J48();
        RandomForest randfDefault = new RandomForest();
        RandomForest randf = new RandomForest();
        randf.setNumTrees(500);
        //Tuned C45
        TunedClassifier tunedC45 = new TunedClassifier();
        ParameterSpace ps = new ParameterSpace();
        //Tune on minimum number of instances, for example
        int[] range = new int[10];
        for(int i=0;i<10;i++)
            range[i] = 2+i*2;

        ps.addParameter("M",range);
        tunedC45.setClassifier(new J48());
        tunedC45.setParameterSpace(ps);
        cls[0]=c45;
        cls[1]=randfDefault;
        cls[2]=randf;
        cls[3]=tunedC45;

        expSettings.dataReadLocation = basePath;
        expSettings.resultsWriteLocation = "c:/temp/";
        expSettings.forceEvaluation = true; // Overwrite existing results?
        expSettings.debug = true;
        //If splits are not defined, can set here, the default is 50/50 splits
        DatasetLoading.setProportionKeptForTraining(0.5);
        for(int i=0;i<cls.length;i++) {
            expSettings.classifierName = names[i];
            expSettings.classifier = cls[i];
            for (String str : allProblems) {
                for (int j = 0; j < 5; j++) {
                    expSettings.datasetName = str;
                    expSettings.foldId = j;  // note that since we're now setting the fold directly, we can resume zero-indexing
                    Experiments.setupAndRunExperiment(expSettings);
//                    expSettings.run();
                }
            }
        }
    }


        public static void multipleClassifierEvaluation() throws Exception {
            System.out.println("Classifier evaluation begins");
            MultipleClassifierEvaluation mce = new MultipleClassifierEvaluation("C:/Temp/","MyExperiment", 5);
            mce.setDatasets(allProblems);
            mce.readInClassifiers(new String[] {"C45","RandF","RandF500","TunedC45"},"C:/Temp/");
            mce.setIgnoreMissingResults(true);
            mce.setBuildMatlabDiagrams(false);
            mce.setTestResultsOnly(true);
            mce.setDebugPrinting(true);
            mce.runComparison();
        }
        public static void main(String[] args) throws Exception {
/** PLAN
 * Part 1: Understanding assessment measures
 *  Generate results file
 *  Open in excel
 *  Work out accuracy, TPR etc
 *  Work out NLL
 *  Work out AUROC
 */
//            runExperimentManually();
//            runExperimentAutomatically();
//            runMultipleExperiments();

            /*  Part 2: Generating performance measures in code */
 //           evaluateInCode();
/*  Part 3: Generating performance measures from results files
             */
            //Create results files
            multipleClassifierEvaluation();
            //Compare
//            compareClassifiers();

//            generateResultsExample(30);
//            collateStatsExample(30);
//            createSingleResultsFile(30);
        }
    }
