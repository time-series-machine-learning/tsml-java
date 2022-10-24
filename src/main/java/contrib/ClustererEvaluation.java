package contrib;

import evaluation.MultipleEstimatorEvaluation;
import evaluation.storage.EstimatorResultsCollection;
import experiments.data.DatasetLists;

import java.io.FileNotFoundException;

public class ClustererEvaluation {

    public static void tslearnCompare() throws Exception {
        MultipleEstimatorEvaluation mee;
        evaluation.MultipleEstimatorEvaluation x = new evaluation.MultipleEstimatorEvaluation("C:\\Results Working Area\\Clustering\\Analysis\\", "tslearn", 1);
        String[] datasets = DatasetLists.tscProblems112;
        x.setDatasets(datasets);
        x.setIgnoreMissingResults(true);
        x.setBuildMatlabDiagrams(true);
        x.setTestResultsOnly(false);
        //x.setDebugPrinting(true);
        x.setUseClusteringStatistics();
        x.setResultsType(EstimatorResultsCollection.ResultsType.CLUSTERING);
        String[] clst1 = {"tslearn-dba"};
        String loc1 = "C:\\Results Working Area\\Clustering\\tslearn\\";
        x.readInEstimators(clst1, loc1);
        String[] clst2 = {"sktime-dba"};
        String loc2 = "C:\\Results Working Area\\Clustering\\kmeans-dba\\";
        x.readInEstimators(clst2, loc2);
        String[] clst3 = {"kmedoids-msm"};
        String loc3 = "C:\\Results Working Area\\Clustering\\kmedoids\\";
        x.readInEstimators(clst3, loc3);
        String[] clst4 = {"kmeans-msm"};
        String loc4 = "C:\\Results Working Area\\Clustering\\kmeans\\";
        x.readInEstimators(clst4, loc4);
        x.runComparison();


    }
    public static void medoidsCompare() throws Exception {
        evaluation.MultipleEstimatorEvaluation x = new evaluation.MultipleEstimatorEvaluation("C:\\Results Working Area\\Clustering\\Analysis\\", "medoids", 1);
        String[] datasets = DatasetLists.tscProblems112;
        x.setDatasets(datasets);
        x.setIgnoreMissingResults(true);
        x.setBuildMatlabDiagrams(true);
        x.setTestResultsOnly(false);
        //x.setDebugPrinting(true);
        x.setUseClusteringStatistics();
        x.setResultsType(EstimatorResultsCollection.ResultsType.CLUSTERING);
        String[] clst3 = {"kmedoids-msm", "kmedoids-twe"};
        String loc3 = "C:\\Results Working Area\\Clustering\\kmedoids\\";
        x.readInEstimators(clst3, loc3);
        x.runComparison();


    }


    public static void main(String[] args) throws Exception {
        medoidsCompare();
    }

}
