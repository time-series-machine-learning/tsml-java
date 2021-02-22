package local;

import evaluation.MultipleClassifierEvaluation;
import experiments.data.DatasetLists;

public class localCollateResults {


    public static void MCEFromOneDirectory(String readPath, String[] ts,String resultsName, int folds, boolean testOnly) throws Exception {
        MultipleClassifierEvaluation m=new MultipleClassifierEvaluation(readPath,
                resultsName,folds);
        m.setIgnoreMissingResults(true);
        m.setBuildMatlabDiagrams(true);
        m.setDebugPrinting(true);
        m.setUseAllStatistics();
        String[] allProbs= DatasetLists.tscProblems112;
        m.setDatasets(allProbs);
        m.setTestResultsOnly(testOnly);
        m.readInClassifiers(ts,readPath);
        m.runComparison();
    }
    

    public static void distanceCompare() throws Exception {
        String[] classifiers={"sktime-MSM"};
        String resultsName="MSM";
        int folds=1;
        String sktimePath="Z:\\Results Working Area\\DistanceBased\\sktime\\";
        String tsmlPath="Z:\\Results Working Area\\DistanceBased\\tsml\\";
        MultipleClassifierEvaluation m=new MultipleClassifierEvaluation("Z:\\Results Working Area\\DistanceBased\\",
                resultsName,folds);
        m.setIgnoreMissingResults(true);
        m.setBuildMatlabDiagrams(true);
        m.setDebugPrinting(true);
        m.setUseAllStatistics();
        String[] allProbs= DatasetLists.tscProblems112;
        m.setDatasets(allProbs);
        m.setTestResultsOnly(true);
        m.readInClassifiers(classifiers,sktimePath);

        String[] tsmlClassifiers={"tsml-MSM"};

        m.readInClassifiers(tsmlClassifiers,tsmlPath);
        m.runComparison();
    }

    public static void main(String[] args) throws Exception {
        distanceCompare();
    }



}
