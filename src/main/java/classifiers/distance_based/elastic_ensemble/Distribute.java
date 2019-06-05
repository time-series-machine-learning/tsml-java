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
import java.util.zip.GZIPOutputStream;

import static experiments.Experiments.sampleDataset;

public class Distribute {

    public static void main(String[] args) throws
                                           Exception {
        String datasetDirPath = new File(args[0]).getPath();
        String datasetName = args[1];
        String classifierName = args[2];
        String resultsDirPath = new File(args[3]).getPath();
        int seed = Integer.parseInt(args[4]);
        int parameterIndex = Integer.parseInt(args[5]);
        Instances[] data = sampleDataset(datasetDirPath, datasetName, seed);
        Instances train = data[0];
        Instances test = data[1];
        Knn knn = new Knn();
        List<ParameterSpace> parameterSpaces = ElasticEnsemble.getParameterSpaces(train, ElasticEnsemble.getDefaultParameterSpaceGetters());
        ParameterSet parameterSet = getParameterPermutation(parameterIndex, parameterSpaces);
        knn.setOptions(parameterSet.getOptions());
        knn.setSeed(seed);
        resultsDirPath = resultsDirPath + '/' + classifierName + '/' + datasetName;
        String trainResultsFilePath = resultsDirPath + '/' + "trainFold" + seed + "_" + parameterIndex + ".csv.gzip";
        String testResultsFilePath = resultsDirPath + '/' + "testFold" + seed + "_" + parameterIndex + ".csv.gzip";
        Utilities.mkfile(trainResultsFilePath);
        Utilities.mkfile(testResultsFilePath);
        ObjectOutputStream trainOutput = new ObjectOutputStream(new GZIPOutputStream(new FileOutputStream(trainResultsFilePath)));
        ObjectOutputStream testOutput = new ObjectOutputStream(new GZIPOutputStream(new FileOutputStream(testResultsFilePath)));
        for(int i = 0; i <= train.size(); i++) {
            System.out.println(i);
            knn.setSampleSize(i);
            knn.buildClassifier(train);
            ClassifierResults trainResults = knn.getTrainResults();
            knn.resetTestRandom();
            ClassifierResults testResults = knn.getTestResults(test);
            trainResults.setDatasetName(datasetName);
            trainResults.setFoldID(seed);
            trainResults.setClassifierName(classifierName);
            testResults.setDatasetName(datasetName);
            testResults.setFoldID(seed);
            testResults.setClassifierName(classifierName);
            trainOutput.writeObject(trainResults.writeFullResultsToString());
            testOutput.writeObject(testResults.writeFullResultsToString());
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
