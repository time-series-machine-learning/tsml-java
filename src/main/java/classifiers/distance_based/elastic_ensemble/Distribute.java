package classifiers.distance_based.elastic_ensemble;

import classifiers.distance_based.knn.Knn;
import evaluation.storage.ClassifierResults;
import evaluation.tuning.ParameterSet;
import evaluation.tuning.ParameterSpace;
import utilities.Utilities;
import weka.core.Instances;

import java.io.File;
import java.io.FileOutputStream;
import java.io.ObjectOutputStream;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.zip.GZIPOutputStream;

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
        boolean wait = false;
        Thread.currentThread().setPriority(Thread.MIN_PRIORITY);
        String datasetDirPath = new File(args[0]).getPath();
        String datasetName = args[1];
        String classifierName = args[2];
        String resultsDirPath = new File(args[3]).getPath();
        int seed = Integer.parseInt(args[4]);
        int parameterIndex = Integer.parseInt(args[5]);
        boolean overwrite = Boolean.parseBoolean(args[6]);
        Instances[] data = sampleDataset(datasetDirPath, datasetName, seed);
        if(wait) Thread.sleep(500);
        Instances train = data[0];
        Instances test = data[1];
        Knn knn = new Knn();
        List<ParameterSpace> parameterSpaces = ElasticEnsemble.getParameterSpaces(train, ElasticEnsemble.getDefaultParameterSpaceGetters());
        int sum = 0;
        for(ParameterSpace parameterSpace : parameterSpaces) {
            parameterSpace.removeDuplicateValues();
            sum += parameterSpace.size();
        }
        System.out.println("n_param_vals: " + sum);
        ParameterSet parameterSet = getParameterPermutation(parameterIndex, parameterSpaces);
        knn.setOptions(parameterSet.getOptions());
        knn.setSeed(seed);
        resultsDirPath = resultsDirPath + '/' + classifierName + '/' + datasetName;
        String trainResultsFilePath = resultsDirPath + "/auxFold" + seed + "/trainParam" + parameterIndex + ".csv.gzip";
        String testResultsFilePath = resultsDirPath + "/auxFold" + seed + "/testParam" + parameterIndex + ".csv.gzip";
        boolean trainExists = !Utilities.mkfile(trainResultsFilePath);
        boolean testExists = !Utilities.mkfile(testResultsFilePath);
        if(!overwrite && trainExists && testExists) {
            System.out.println("train and test exists");
            System.exit(0);
        }
        // todo work out if this overwrites files
        ObjectOutputStream trainOutput = new ObjectOutputStream(new GZIPOutputStream(new FileOutputStream(trainResultsFilePath)));
        ObjectOutputStream testOutput = new ObjectOutputStream(new GZIPOutputStream(new FileOutputStream(testResultsFilePath)));
        for(int i = 0; i <= train.size(); i++) {
            long startTime = System.nanoTime();
            System.out.println("train size: " + i);
            knn.setSampleSize(i);
            knn.buildClassifier(train);
            if(!trainExists || overwrite) {
                ClassifierResults trainResults = knn.getTrainResults();
                trainResults.setDatasetName(datasetName);
                trainResults.setFoldID(seed);
                trainResults.setClassifierName(classifierName);
                trainOutput.writeObject(trainResults.writeFullResultsToString());
                trainOutput.flush();
            }
            if(!trainExists || overwrite) {
                knn.resetTestRandom();
                ClassifierResults testResults = knn.getTestResults(test);
                testResults.setDatasetName(datasetName);
                testResults.setFoldID(seed);
                testResults.setClassifierName(classifierName);
                testOutput.writeObject(testResults.writeFullResultsToString());
                testOutput.flush();
            }
            long endTime = System.nanoTime();
            long diff = endTime - startTime;
            long diffMillis = TimeUnit.MILLISECONDS.convert(diff, TimeUnit.NANOSECONDS);
            if(diffMillis < 500 && wait) {
                Thread.sleep(500 - diffMillis);
            }
        }
        trainOutput.close();
        testOutput.close();
    }

    private static ParameterSet getParameterPermutation(int parameterIndex, List<ParameterSpace> parameterSpaces) {
        int index = parameterIndex;
        boolean stop = false;
        ParameterSpace parameterSpace;
        int parameterSpaceIndex = 0;
        do {
            parameterSpace = parameterSpaces.get(parameterSpaceIndex);
            int size = parameterSpace.size();
            if(index < size) {
                stop = true;
            } else {
                index -= size;
                parameterSpaceIndex++;
            }
        }
        while (!stop);
        return parameterSpace.get(index);
    }
}
