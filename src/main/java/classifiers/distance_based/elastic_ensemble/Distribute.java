package classifiers.distance_based.elastic_ensemble;

import classifiers.distance_based.knn.Knn;
import evaluation.storage.ClassifierResults;
import evaluation.tuning.ParameterSet;
import evaluation.tuning.ParameterSpace;
import utilities.Utilities;
import weka.core.Instance;
import weka.core.Instances;

import javax.rmi.CORBA.Util;
import java.io.*;
import java.nio.channels.FileChannel;
import java.nio.channels.FileLock;
import java.nio.channels.OverlappingFileLockException;
import java.util.List;
import java.util.concurrent.TimeUnit;

import static experiments.Experiments.sampleDataset;

public class Distribute {

    public static void main(String[] args) throws
                                       Exception {
    System.out.print("args:");
    for(String arg : args) {
        System.out.print(" ");
        System.out.print(arg);
    }
    System.out.println();
    String datasetDirPath = new File(args[0]).getPath();
    System.out.println(datasetDirPath);
    String datasetName = args[1];
    System.out.println(datasetName);
    String resultsDirPath = new File(args[2]).getPath();
    System.out.println(resultsDirPath);
    int seed = Integer.parseInt(args[3]);
    System.out.println(seed);
    Instances[] data = sampleDataset(datasetDirPath, datasetName, seed);
    Instances train = data[0];
    Instances test = data[1];
//    Knn cls = new Knn();
    ElasticEnsemble cls = new ElasticEnsemble();
    cls.setSeed(seed);
    double pp = Double.parseDouble(args[4]);
    System.out.println(pp);
    cls.setNumParametersLimitPercentage(pp);
    for(int n = 10; n <= 100; n+=10) {
        double np = (double) n / 100;
        System.out.println(np + ", " + pp);
        String classifierName = "ee_np=" + np + "_pp=" + pp;
//        String classifierName = "knn";
        String experimentResultsDirPath = resultsDirPath + "/" + classifierName + "/Predictions/" + datasetName;
        String trainResultsFilePath = experimentResultsDirPath + "/trainFold" + seed + ".csv";
        String testResultsFilePath = experimentResultsDirPath + "/testFold" + seed + ".csv";
        cls.setNeighbourhoodSizeLimitPercentage(np);
        boolean trainMissing = !exists(trainResultsFilePath);
        boolean testMissing = !exists(testResultsFilePath);
        if(trainMissing || testMissing) {
            System.out.println("training");
            cls.buildClassifier(train);
        }
        if(trainMissing) {
            System.out.println("getting train results");
            ClassifierResults trainResults = cls.getTrainResults();
            trainResults.setDatasetName(datasetName);
            trainResults.setFoldID(seed);
            trainResults.setClassifierName(classifierName);
            writeToFile(trainResults, trainResultsFilePath);
        } else {
            System.out.println("train exists");
        }
        if(testMissing) {
            System.out.println("getting test results");
            ClassifierResults testResults = cls.getTestResults(test);
            testResults.setDatasetName(datasetName);
            testResults.setFoldID(seed);
            testResults.setClassifierName(classifierName);
            writeToFile(testResults, testResultsFilePath);
            cls.resetTestRandom();
        } else {
            System.out.println("test exists");
        }
    }
}

//    public static void main(String[] args) throws
//                                           Exception {
//        System.out.print("args:");
//        for(String arg : args) {
//            System.out.print(" ");
//            System.out.print(arg);
//        }
//        System.out.println();
//        String datasetDirPath = new File(args[0]).getPath();
//        System.out.println(datasetDirPath);
//        String datasetName = args[1];
//        System.out.println(datasetName);
//        String resultsDirPath = new File(args[2]).getPath();
//        System.out.println(resultsDirPath);
//        int seed = Integer.parseInt(args[3]);
//        System.out.println(seed);
//        Instances[] data = sampleDataset(datasetDirPath, datasetName, seed);
//        Instances train = data[0];
//        Instances test = data[1];
//        ElasticEnsemble ee = new ElasticEnsemble();
//        ee.setSeed(seed);
//        boolean stop = false;
//        int hours = 1;
//        while(!stop) {
//            long contractNanos = TimeUnit.NANOSECONDS.convert(hours, TimeUnit.HOURS);
//            ee.setTimeLimit(contractNanos);
//            String classifierName = "ee_" + hours + "hrs";
//            String experimentResultsDirPath = resultsDirPath + "/" + classifierName + "/Predictions/" + datasetName;
//            String trainResultsFilePath = experimentResultsDirPath + "/trainFold" + seed + ".csv";
//            String testResultsFilePath = experimentResultsDirPath + "/testFold" + seed + ".csv";
//            boolean trainMissing = !exists(trainResultsFilePath);
//            boolean testMissing = !exists(testResultsFilePath);
//            System.out.println("training");
//            ee.buildClassifier(train);
//            if(ee.getTrainResults().getBuildTimeInNanos() > contractNanos) {
//                stop = true;
//            }
//            if(trainMissing) {
//                System.out.println("getting train results");
//                ClassifierResults trainResults = ee.getTrainResults();
//                trainResults.setDatasetName(datasetName);
//                trainResults.setFoldID(seed);
//                trainResults.setClassifierName(classifierName);
//                writeToFile(trainResults, trainResultsFilePath);
//            } else {
//                System.out.println("train exists");
//            }
//            if(testMissing) {
//                System.out.println("getting test results");
//                ClassifierResults testResults = ee.getTestResults(test);
//                testResults.setDatasetName(datasetName);
//                testResults.setFoldID(seed);
//                testResults.setClassifierName(classifierName);
//                writeToFile(testResults, testResultsFilePath);
//                ee.resetTestRandom();
//            } else {
//                System.out.println("test exists");
//            }
//        }
//    }

//    public static void main(String[] args) throws
//                                           Exception {
//        System.out.print("args:");
//        for(String arg : args) {
//            System.out.print(" ");
//            System.out.print(arg);
//        }
//        System.out.println();
//        String datasetDirPath = new File(args[0]).getPath();
//        System.out.println(datasetDirPath);
//        String datasetName = args[1];
//        System.out.println(datasetName);
//        String resultsDirPath = new File(args[2]).getPath();
//        System.out.println(resultsDirPath);
//        int seed = Integer.parseInt(args[3]);
//        System.out.println(seed);
//        Instances[] data = sampleDataset(datasetDirPath, datasetName, seed);
//        Instances train = data[0];
//        Instances test = data[1];
//        ElasticEnsemble ee = new ElasticEnsemble();
//        ee.setNumParametersLimitPercentage(0.1);
//        ee.setSeed(seed);
//        ee.setNeighbourSearchStrategy(Knn.NeighbourSearchStrategy.RANDOM);
//        int n = Integer.parseInt(args[4]);
////        n *= train.numClasses();
////        double pp = Double.parseDouble(args[4]);
////        System.out.println(pp);
////        ee.setNumParametersLimitPercentage(pp);
////        for(int n = 5; n <= 30; n+=5) {
////            double np = (double) n / 100;
////            System.out.println(np + ", " + pp);
//            System.out.println("n=" + n);
//            String classifierName = "ee_n=" + n;
////            String classifierName = "ee_np=" + np + "_pp=" + pp;
//            String experimentResultsDirPath = resultsDirPath + "/" + classifierName + "/Predictions/" + datasetName;
//            String trainResultsFilePath = experimentResultsDirPath + "/trainFold" + seed + ".csv";
//            String testResultsFilePath = experimentResultsDirPath + "/testFold" + seed + ".csv";
//            ee.setTrainNeighbourhoodSizeLimit(n);
////            ee.setTrainNeighbourhoodSizeLimitPercentage(np);
//            boolean trainMissing = !exists(trainResultsFilePath);
//            boolean testMissing = !exists(testResultsFilePath);
//            if(trainMissing || testMissing) {
//                System.out.println("training");
//                ee.buildClassifier(train);
//            }
//            if(trainMissing) {
//                System.out.println("getting train results");
//                ClassifierResults trainResults = ee.getTrainResults();
//                trainResults.setDatasetName(datasetName);
//                trainResults.setFoldID(seed);
//                trainResults.setClassifierName(classifierName);
//                writeToFile(trainResults, trainResultsFilePath);
//            } else {
//                System.out.println("train exists");
//            }
//            if(testMissing) {
//                System.out.println("getting test results");
//                ClassifierResults testResults = ee.getTestResults(test);
//                testResults.setDatasetName(datasetName);
//                testResults.setFoldID(seed);
//                testResults.setClassifierName(classifierName);
//                writeToFile(testResults, testResultsFilePath);
//                ee.resetTestRandom();
//            } else {
//                System.out.println("test exists");
//            }
////        }
//    }

    private static boolean exists(String path) {
        File file = new File(path);
        return file.exists() && file.length() > 0;
    }

    private static FileLock lockedExists(String path) throws
                                                     IOException {
        // Get a file channel for the file
        File file = new File(path);
        Utilities.mkdirParent(file);
        FileChannel channel = new RandomAccessFile(file, "rw").getChannel();

        // Use the file channel to create a lock on the file.
        // This method blocks until it can retrieve the lock.
        FileLock lock = channel.lock();
        boolean exists = file.length() > 0;
        if(exists) {
            lock.close();
            // Close the file
            lock.channel().close();
            lock = null;
        }
        return lock;
    }

    private static void writeToFile(ClassifierResults results, String path, FileLock lock) throws Exception {
        writeToFile(results, path);
        lock.close();
        lock.channel().close();
    }

    private static void writeToFile(ClassifierResults results, String path) throws Exception {
        Utilities.mkdirParent(path);
        BufferedWriter output = new BufferedWriter(new FileWriter(path));
        output.write(results.writeFullResultsToString());
        output.close();
    }
}
