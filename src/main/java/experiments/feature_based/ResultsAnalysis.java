package experiments.feature_based;

import evaluation.MultipleClassifierEvaluation;
import experiments.data.DatasetLists;

public class ResultsAnalysis {

    public static String[] transforms = {"Catch22","PCATransformer","RandomIntervals","SignatureTransformer","SummaryTransformer","TSFreshFeatureExtractor"};
    public static String[] classifiers = {"RidgeCV","RotF","XGBoost"};
    public static String resultsLocation = "Z:\\Results Working Area\\FeatureBased\\";
    public static String analysisLocation = "Z:\\Results Working Area\\FeatureBased\\Analysis\\";

    public static void distanceDebug() throws Exception {
        MultipleClassifierEvaluation m=new MultipleClassifierEvaluation("E://Temp//", "distances", 1);
        m.setBuildMatlabDiagrams(true);
        m.setDebugPrinting(false);
        m.setUseAllStatistics();
        m.setDatasets(DatasetLists.tscProblems112);
        m.setIgnoreMissingResults(true);
        m.readInClassifiers(new String[] {"1nn-dtw"},
                "Z:\\Results Working Area\\Debug\\DistancesOld\\");
        m.readInClassifiers(new String[] {"1nn-dtw"},
                "Z:\\Results Working Area\\Debug\\DistancesNew\\");
        m.runComparison();

    }



}
