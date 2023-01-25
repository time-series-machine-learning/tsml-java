package contrib;

import evaluation.MultipleEstimatorEvaluation;
import evaluation.storage.EstimatorResultsCollection;
import experiments.data.DatasetLists;
import fileIO.InFile;
import fileIO.OutFile;

import java.io.File;
import java.io.FileNotFoundException;

import static contrib.ClassifierEvaluation.summariseResultsPresent;

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

    static String[] clst = {"dtw","ddtw", "edr","erp","euclidean","lcss","msm", "twe","wdtw","wddtw"};

    public static void normaliseCompare(String n, String name, int r) throws Exception {
        evaluation.MultipleEstimatorEvaluation x = new evaluation.MultipleEstimatorEvaluation("C:\\Results Working Area\\Clustering\\dami_paper\\analysis\\", name, r);
        String[] datasets = DatasetLists.tscProblems112;
        x.setDatasets(datasets);
        x.setIgnoreMissingResults(true);
        x.setBuildMatlabDiagrams(true);
        x.setTestResultsOnly(false);
        //x.setDebugPrinting(true);
        x.setUseClusteringStatistics();
        x.setResultsType(EstimatorResultsCollection.ResultsType.CLUSTERING);
        x.readInEstimators(clst, n);
        x.runComparison();


    }
    public static void tuneCompare() throws Exception {
        String name = "tunedDistances";
        int r = 1;
        evaluation.MultipleEstimatorEvaluation x = new evaluation.MultipleEstimatorEvaluation("C:\\Results Working Area\\Clustering\\dami_paper\\analysis\\", name, r);
        String[] datasets = DatasetLists.tscProblems112;
        x.setDatasets(datasets);
        x.setIgnoreMissingResults(true);
        x.setBuildMatlabDiagrams(true);
        x.setTestResultsOnly(true);
        //x.setDebugPrinting(true);
        x.setUseClusteringStatistics();
        x.setResultsType(EstimatorResultsCollection.ResultsType.CLUSTERING);
        String c1 = "C:\\Results Working Area\\Clustering\\dami_paper\\section5_4_distance_tuned\\";
        String[] cls = {"dtw","twe","msm","wdtw","erp"};
        x.readInEstimators(cls, c1);
        x.runComparison();
    }
    public static void tuneVsUntuned() throws Exception {
        String name = "Section5_5TunedVsUntuned";
        int r = 1;
        evaluation.MultipleEstimatorEvaluation x = new evaluation.MultipleEstimatorEvaluation("C:\\Results Working Area\\Clustering\\dami_paper\\analysis\\", name, r);
        String[] datasets = DatasetLists.tscProblems112;
        x.setDatasets(datasets);
        x.setIgnoreMissingResults(true);
        x.setBuildMatlabDiagrams(true);
        x.setTestResultsOnly(true);
        //x.setDebugPrinting(true);
        x.setUseClusteringStatistics();
        x.setResultsType(EstimatorResultsCollection.ResultsType.CLUSTERING);
        String c1 = "C:\\Results Working Area\\Clustering\\dami_paper\\section5_5_distance_tuned\\";
        String[] cls = {"dtw-t","msm-t","wdtw-t","erp-t"};
        x.readInEstimators(cls, c1);
        String c2 = "C:\\Results Working Area\\Clustering\\dami_paper\\first_submission\\kmeans\\";
        String[] cls2 = {"dtw","msm","wdtw","erp"};
        x.readInEstimators(cls2, c2);
        x.runComparison();
    }

    public static void windowCompare(String n, String name, int r) throws Exception {
        evaluation.MultipleEstimatorEvaluation x = new evaluation.MultipleEstimatorEvaluation("C:\\Results Working Area\\Clustering\\sktime\\Analysis\\", name, r);
        String[] datasets = DatasetLists.tscProblems112;
        x.setDatasets(datasets);
        x.setIgnoreMissingResults(true);
        x.setBuildMatlabDiagrams(true);
        x.setTestResultsOnly(false);
        //x.setDebugPrinting(true);
        x.setUseClusteringStatistics();
        x.setResultsType(EstimatorResultsCollection.ResultsType.CLUSTERING);
        String [] clst_w={"dtw20","dtw5","ed","msm"};
        x.readInEstimators(clst_w, n);
        x.runComparison();
    }

    public static void dtwCompare(String n, String name, int r) throws Exception {
        evaluation.MultipleEstimatorEvaluation x = new evaluation.MultipleEstimatorEvaluation("C:\\Results Working Area\\Clustering\\sktime\\Analysis\\", name, r);
        String[] datasets = DatasetLists.tscProblems112;
        x.setDatasets(datasets);
        x.setIgnoreMissingResults(true);
        x.setBuildMatlabDiagrams(true);
        x.setTestResultsOnly(true);
        //x.setDebugPrinting(true);
        x.setUseClusteringStatistics();
        x.setResultsType(EstimatorResultsCollection.ResultsType.CLUSTERING);
        String [] clst_w={"dtw-t","dtw-ba","dtw5","dtw20","ed","msm","t-dtw20"};
        x.readInEstimators(clst_w, n);
        x.runComparison();
    }
    static String[] distances = {"dtw","msm","twe","wdtw","edr","erp"};
    static String loc = "C:\\Results Working Area\\Clustering\\";


    public static void collateInertia(String name){
        OutFile of = new OutFile("C:\\Results Working Area\\Clustering\\dami_paper\\section5_1_inertia\\Inertia"+name+".csv");
        OutFile of2 = new OutFile("C:\\Results Working Area\\Clustering\\dami_paper\\section5_1_inertia\\Iterations"+name+".csv");
        String dir1="C:\\Results Working Area\\Clustering\\dami_paper\\section5_1_inertia\\"+name+"\\";
        for(String c:clst)
            of.writeString(","+c);
        of.writeString("\n");
        for(String c:clst)
            of2.writeString(","+c);
        of2.writeString("\n");
        for(String a: DatasetLists.tscProblems112){
            of.writeString(a);
            of2.writeString(a);
            for(String c:clst) {
                String nm = dir1+c+"\\Predictions\\"+a+"\\testResample0.csv";
                System.out.println(" Checking "+nm);
                File f= new File(nm);
                if(f.exists()) {
                    InFile inf = new InFile(nm);
                    inf.readLine();
                    String second = inf.readLine();
                    String[] split = second.split(",");
                    of.writeString("," + split[split.length - 3]);
                    of2.writeString("," + split[split.length - 1]);
                }
                else{
                    of.writeString(", ");
                    of2.writeString(", ");
                }
            }
            of.writeString("\n");
            of2.writeString("\n");
        }
    }

    public static void initCompare() throws Exception {
        String name = "Section5_3InitAlgoskmeansppkmedoids";
        int r = 1;
        evaluation.MultipleEstimatorEvaluation x = new evaluation.MultipleEstimatorEvaluation("C:\\Results Working Area\\Clustering\\dami_paper\\analysis\\Section 5.3 Init\\", name, r);
        String[] datasets = DatasetLists.tscProblems112;
        x.setDatasets(datasets);
        x.setIgnoreMissingResults(true);
        x.setBuildMatlabDiagrams(true);
        x.setTestResultsOnly(true);
        //x.setDebugPrinting(true);
        x.setUseClusteringStatistics();
        x.setResultsType(EstimatorResultsCollection.ResultsType.CLUSTERING);
        String c1 = "C:\\Results Working Area\\Clustering\\dami_paper\\section5_3_init_algo\\kmeanspp\\kmedoids\\";
        String[] cls = {"dtw","msm","euclidean"};
        x.readInEstimators(cls, c1);
        x.runComparison();
    }


    public static void main(String[] args) throws Exception {
//        collateInertia("kmeans");
//        collateInertia("kmedoids");
        tuneVsUntuned();
        //initCompare();

        //        medoidsCompare();
 //       normaliseCompare("C:\\Results Working Area\\Clustering\\first_submission\\kmeans\\","normedMeans",1);
 //       normaliseCompare("C:\\Results Working Area\\Clustering\\first_submission\\kmedoids\\","normedMedoids",1);

//        summariseResultsPresent(clst,DatasetLists.tscProblems112, loc+"normalised\\kmeans\\mean\\", "completeNormed.csv");
//        normaliseCompare(loc+"normalised\\kmeans\\mean\\","normalised",1);
//        summariseResultsPresent(clst,DatasetLists.tscProblems112, loc+"raw\\kmeans\\mean\\", "completeRaw.csv");
//        normaliseCompare(loc+"raw\\kmeans\\mean\\","raw",1);
//        tuneCompare();
//       tuneVsUntuned();
//        windowCompare(loc+"dtw_section\\","differentWindows",1);
//        dtwCompare(loc+"dtw_section\\","dtwVariantsTune20",1);

//        String path="C:\\Results Working Area\\Clustering\\dami_paper\\section5_4_distance_tuned\\";
 //       summariseResultsPresent(distances, DatasetLists.tscProblems112, path, "TunedCounts.csv");

//        summariseResultsPresent(clst,DatasetLists.tscProblems112, loc+"raw\\kmeans\\mean\\", "completeRaw.csv");
//        summariseResultsPresent(clst,DatasetLists.tscProblems112, loc+"normalised\\kmeans\\mean\\", "completeNormed.csv");
//        summariseResultsPresent(clst,DatasetLists.tscProblems112, loc+"raw\\kmedoids\\mean\\", "completeRaw.csv");
//        summariseResultsPresent(clst,DatasetLists.tscProblems112, loc+"normalised\\kmedoids\\mean\\", "completeNormed.csv");

    }

}
