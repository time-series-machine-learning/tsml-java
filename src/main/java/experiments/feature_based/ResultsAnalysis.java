package experiments.feature_based;

import evaluation.MultipleClassifierEvaluation;
import experiments.data.DatasetLists;

import java.util.ArrayList;

public class ResultsAnalysis {

    public static String[] transforms = {"Catch22","PCA","RandIntervals","RandIntC22","Signature","Summary","TSFresh"};
    public static String[] transformsShort = {"Catch22","PCA","Signature","Summary","TSFresh"};//"RandomIntervals","RandomIntC22",
    public static String[] classifiers = {"RidgeCV","RotF","XGBoost"};
    public static String[] selCls = {"Ridge-RIC22","RotF-RIC22","XG-RIC22","Ridge-TSFr","RotF-TSFr","XG-TSFr"};//"RandomIntervals","RandomIntC22",
    public static String resultsLocation = "C:\\Results Working Area\\FeatureBased\\";
    public static String analysisLocation = "C:\\Results Working Area\\FeatureBased\\Analysis\\";

    public static void allClassifiers() throws Exception {
        MultipleClassifierEvaluation m=new MultipleClassifierEvaluation(analysisLocation, "bestTransforms", 30);
        m.setBuildMatlabDiagrams(true);
        m.setDebugPrinting(false);
        m.setUseAllStatistics();
        m.setDatasets(DatasetLists.tscProblems112);
        m.setIgnoreMissingResults(true);
        m.readInClassifiers(selCls,  resultsLocation + "\\ByTransform\\");
//        m.readInClassifiers(new String[]{"RotF","DTWCV"},resultsLocation);
        m.runComparison();

    }

    public static void fixedClassifier(String classifier) throws Exception {
        MultipleClassifierEvaluation m=new MultipleClassifierEvaluation(analysisLocation, classifier, 30);
        m.setBuildMatlabDiagrams(true);
        m.setDebugPrinting(false);
        m.setUseAllStatistics();
        m.setDatasets(DatasetLists.tscProblems112);
        m.setIgnoreMissingResults(true);
        String[] classifiers = new String[transforms.length];
        for(int i=0;i<transforms.length;i++)
            classifiers[i]=transforms[i];
        m.readInClassifiers(classifiers,resultsLocation+classifier+"\\");
        m.readInClassifiers(new String[]{"RotF","DTWCV"},resultsLocation);
        m.runComparison();

    }
    public static void againstBest() throws Exception {
        MultipleClassifierEvaluation m=new MultipleClassifierEvaluation(analysisLocation, "vsSOTA", 30);
        m.setBuildMatlabDiagrams(true);
        m.setDebugPrinting(false);
        m.setUseAllStatistics();
        m.setDatasets(DatasetLists.tscProblems112);
        m.setIgnoreMissingResults(true);
        m.readInClassifiers(new String[]{"RotF-TSFr"},  resultsLocation + "\\ByTransform\\");
        m.readInClassifiers(new String[]{"DTWCV","HC2","TS-CHIEF","ROCKET","InceptionTime"},  resultsLocation);
        m.runComparison();

    }

    public static void main(String[] args) throws Exception {
        againstBest();
//        allClassifiers();
//        fixedClassifier(classifiers[1]);
//        System.out.printf("Comparing classifier "+classifiers[0]);
//        fixedClassifier(classifiers[0]);
//        System.out.printf("Comparing classifier "+classifiers[2]);
//        fixedClassifier(classifiers[2]);

    }

}
