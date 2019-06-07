package classifiers.distance_based.elastic_ensemble;

import classifiers.distance_based.knn.Knn;
import evaluation.storage.ClassifierResults;
import evaluation.tuning.ParameterSet;
import evaluation.tuning.ParameterSpace;
import utilities.Utilities;
import weka.core.Instances;

import java.io.*;
import java.util.List;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

import static experiments.Experiments.sampleDataset;

public class Reader {
    public static void main(String[] args) throws
                                           IOException {
        String type = args[0];
        String datasetName = args[1];
        String classifierName = args[2];
        String resultsDirPath = new File(args[3]).getPath();
        int seed = Integer.parseInt(args[4]);
        int parameterIndex = Integer.parseInt(args[5]);
        resultsDirPath = resultsDirPath + '/' + classifierName + '/' + datasetName;
        String trainResultsFilePath = resultsDirPath + "/auxFold" + seed + '/' + type + "Param" + parameterIndex + ".csv.gzip";
        ObjectInputStream trainInput = new ObjectInputStream(new GZIPInputStream(new FileInputStream(trainResultsFilePath)));
        int i = 0;
        try {
            while(true) {
                System.out.println(i++);
                System.out.println(trainInput.readObject());
                System.out.println();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        trainInput.close();
    }
}
