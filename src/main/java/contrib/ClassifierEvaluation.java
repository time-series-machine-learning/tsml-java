package contrib;

import evaluation.storage.ClassifierResults;
import evaluation.storage.EstimatorResultsCollection;
import experiments.data.DatasetLists;
import fileIO.InFile;
import fileIO.OutFile;
import org.checkerframework.checker.units.qual.A;
import tsml.classifiers.hybrids.HIVE_COTE;
import experiments.CollateResults;
import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.stream.Stream;

public class ClassifierEvaluation {

    public static String[] highDimension = {
            "ArticularyWordRecognition", //Index 0
            "DuckDuckGeese",
            "EMO",
            "FingerMovements",
            "HAR",
            "HandMovementDirection",
            "Heartbeat",
            "JapaneseVowelsEq",
            "MotorImagery",
            "MindReading",
            "NATOPS",
            "PEMS-SF",
            "PhonemeSpectra",
            "Siemens",
            "SpokenArabicDigitsEq",
    };
    //</editor-fold>


    public static void dimensionSelection() throws Exception {
        evaluation.MultipleEstimatorEvaluation x = new evaluation.MultipleEstimatorEvaluation("C:\\Results Working Area\\DimensionSelection\\Analysis\\", "All", 30);
        String[] datasets = highDimension;
        x.setDatasets(datasets);
        x.setIgnoreMissingResults(true);
        x.setBuildMatlabDiagrams(true);
        x.setTestResultsOnly(true);
        //x.setDebugPrinting(true);
        //x.setUseClusteringStatistics();
        x.setResultsType(EstimatorResultsCollection.ResultsType.CLASSIFICATION);
        String[] clst3 = {"HC2","ROCKET", "HC2-Rand20", "HC2-Rand60", "HC2-CLST","HC2-ECP","HC2-ECS","HC2-KMEANS","HC2-MSTS","HC2-R_A", "HC2-R_M","HC2-R_S"};
        String loc3 = "C:\\Results Working Area\\DimensionSelection\\";
        x.readInEstimators(clst3, loc3);
        x.runComparison();




    }
    public static void referenceResults() throws Exception {
        evaluation.MultipleEstimatorEvaluation x = new evaluation.MultipleEstimatorEvaluation("C:\\Temp\\", "AllResults", 30);

//        System.arraycopy(d1,0,datasets,0,d1.length);
//        System.arraycopy(d2,0,datasets,d1.length,d2.length);
        ClassifierResults.printOnFailureToLoad = false;
        x.setIgnoreMissingResults(true);
        x.setBuildMatlabDiagrams(true);
        x.setTestResultsOnly(true);
        x.setResultsType(EstimatorResultsCollection.ResultsType.CLASSIFICATION);
        String loc = "X:\\MultivariateReferenceResults\\tsml\\";
        String[] d = DatasetLists.mtscProblems2018;
        String[] clst = {"Arsenal","cBOSS","CIF","DrCIF","DTW-A","DTW-D","DTW-I","gRSF","HC1","HC2","InceptionTime","MrSEQL","MUSE","ResNet","RISE","ROCKET","STC","TDE","TSF"}; //,"TDE", "DrCIF", "CIF","HC1","TDE","HC2","STSF"};
        x.setDatasets(d);
        x.readInEstimators(clst, loc);
        //x.setDebugPrinting(true);
        //x.setUseClusteringStatistics();
        x.runComparison();
    }

    public static void localResults(String[] classifiers, String[] d, String location, String name, int numFolds) throws Exception {
//        String location = "C:\\Results Working Area\\MultivariateReferenceResults\\sktime\\";
        evaluation.MultipleEstimatorEvaluation x = new evaluation.MultipleEstimatorEvaluation(location, name, numFolds);

        ClassifierResults.printOnFailureToLoad = false;
        x.setIgnoreMissingResults(true);
        x.setBuildMatlabDiagrams(true);
        x.setTestResultsOnly(true);
        x.setResultsType(EstimatorResultsCollection.ResultsType.CLASSIFICATION);
        String loc = location;
        System.out.println(" number problems = "+d.length);
        x.setDatasets(d);
        x.readInEstimators(classifiers, loc);
        //x.setDebugPrinting(true);
        //x.setUseClusteringStatistics();
        x.runComparison();
    }
/*
PhonemeSpectra
PenDigits
InsectWingbeatEq
FaceDetection
EigenWorms
SpokenArabicDigits
Seimens
EMOPain
Tiselac


 */
    public static void listNewProblems(){
        HashSet<String> all = new HashSet<>();  //128
        HashSet<String> equalLength = new HashSet<>();//113, inc mushroom
        HashSet<String> newArff = new HashSet<>();
        HashSet<String> newTS = new HashSet<>();
        for(String str:DatasetLists.tscProblems2018)
            all.add(str);
        for(String str:DatasetLists.tscProblems112)
            equalLength.add(str);






    }

    public static void summariseAccuracies(String[] classifiers, String[] problems, String path, int maxResamples) {
        double[][][] accTest = new double[problems.length][classifiers.length][maxResamples];
        double[][][] accTrain = new double[problems.length][classifiers.length][maxResamples];;
        for(int i=0;i< classifiers.length;i++){
            for(int j=0;j< problems.length;j++){
                for(int k=0;k<maxResamples;k++){
                    String fileName=path+classifiers[i]+"\\Predictions\\"+problems[j]+"\\";
                    File f = new File(fileName+"testResample"+k+".csv");
                    if(f.exists()){
                        InFile inf = new InFile(fileName+"testResample"+k+".csv");
                        inf.readLine();
                        inf.readLine();
                        accTest[j][i][k] = inf.readDouble();
                    }
                    else
                        accTest[j][i][k] = Double.POSITIVE_INFINITY;
                }
            }
        }
        OutFile all = new OutFile(path+"DefaultSplitAccuracies.csv");
        for(int i=0;i< classifiers.length;i++)
            all.writeString(","+classifiers[i]);
        all.writeString("\n");
        for(int j=0;j< problems.length;j++) {
            all.writeString(problems[j]);
            for (int i = 0; i < classifiers.length; i++)
                all.writeString("," + accTest[j][i][0]);
            all.writeString(",\n");
        }
        all = new OutFile(path+"AverageAccuracies.csv");
        for(int i=0;i< classifiers.length;i++)
            all.writeString(","+classifiers[i]);
        all.writeString("\n");
        for(int j=0;j< problems.length;j++) {
            all.writeString(problems[j]);
            for (int i = 0; i < classifiers.length; i++){
                double sum=0;
                for(int k=0;k<maxResamples; k++){
                    if(accTest[j][i][k]!=Double.POSITIVE_INFINITY)
                        sum+=accTest[j][i][k];
                    else {
                        sum = Double.POSITIVE_INFINITY;
                        break;
                    }
                }
                if(sum!=Double.POSITIVE_INFINITY)
                    all.writeString("," + sum/maxResamples);
                else
                    all.writeString("," + sum);
            }
            all.writeString(",\n");
        }
    }
    public static void summariseResultsPresent(String[] classifiers, String[] problems, String path, String outfile){
        int maxResamples=30;
        int[][] countTest = new int[problems.length][classifiers.length];
        int[][] countTrain = new int[problems.length][classifiers.length];
        for(int i=0;i< classifiers.length;i++){
            for(int j=0;j< problems.length;j++){
                for(int k=0;k<maxResamples;k++){
                   String fileName=path+classifiers[i]+"\\Predictions\\"+problems[j]+"\\";
                    File f = new File(fileName+"testResample"+k+".csv");
                    if(f.exists())
                        countTest[j][i]++;
                    f = new File(fileName+"trainResample"+k+".csv");
                    if(f.exists())
                        countTrain[j][i]++;
                }
            }
        }
        OutFile out= new OutFile(path+outfile);
        System.out.println("************** TEST COUNTS **************************");
        out.writeString(",,Test Counts");
        for (int i = 0; i < classifiers.length+2; i++)
            out.writeString(",");
        out.writeString("Train Counts");
        out.writeString("\n");

        for (int i = 0; i < classifiers.length; i++)
            out.writeString(","+classifiers[i]);
        out.writeString(",,");
        for (int i = 0; i < classifiers.length; i++)
            out.writeString(","+classifiers[i]);
        out.writeLine("");

        for(int j=0;j< problems.length;j++) {
            System.out.print(problems[j]);
            out.writeString(problems[j]);
            for (int i = 0; i < classifiers.length; i++) {
                System.out.print("," + countTest[j][i]);
                out.writeString(","+countTest[j][i]);
            }
            out.writeString(",,");
            for (int i = 0; i < classifiers.length; i++) {
                out.writeString(","+countTrain[j][i]);
            }
            System.out.print("\n");
            out.writeString("\n");
        }
        System.out.println("************** TRAIN COUNTS **************************");
//        for (int i = 0; i < classifiers.length; i++)
//            out.writeString(","+classifiers[i]);
//        out.writeLine("");
        for(int j=0;j< problems.length;j++) {
            System.out.print(problems[j]);
            for (int i = 0; i < classifiers.length; i++){
                System.out.print("," + countTrain[j][i]);
            }
            System.out.print("\n");

        }
    }
    //<editor-fold defaultstate="collapsed" desc="defaultCapableSktimeClassifiers: sktime classifiers that can be built with default parameters"
    static String[] defaultCapableSktimeClassifiers = {
            "Arsenal",
            "BOSSEnsemble",
            "CNNClassifier",
            "CanonicalIntervalForest",
            "Catch22Classifier",
            "ComposableTimeSeriesForestClassifier",
            "ContractableBOSS",
            "DrCIF",
            "DummyClassifier",
            "ElasticEnsemble",
            "FreshPRINCE",
            "HIVECOTEV1",
            "HIVECOTEV2",
            "IndividualBOSS",
            "IndividualTDE",
            "KNeighborsTimeSeriesClassifier",
            "MatrixProfileClassifier",
            "ProximityForest",
            "ProximityStump",
            "ProximityTree",
            "RandomIntervalClassifier",
            "RandomIntervalSpectralEnsemble",
            "RocketClassifier",
            "ShapeDTW",
            "ShapeletTransformClassifier",
            "SignatureClassifier",
            "SummaryClassifier",
            "SupervisedTimeSeriesForest",
            "TSFreshClassifier",
            "TemporalDictionaryEnsemble",
            "TimeSeriesForestClassifier",
            "WEASEL"
    };
//</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="allSktimeClassifiers: All classifiers in sktime toolkit"
    static String[] allSktimeClassifiers = {
            "Arsenal",
            "BOSSEnsemble",
            "CNNClassifier",
            "CanonicalIntervalForest",
            "Catch22Classifier",
            "ClassifierPipeline",
            "ColumnEnsembleClassifier",
            "ComposableTimeSeriesForestClassifier",
            "ContractableBOSS",
            "DrCIF",
            "DummyClassifier",
            "ElasticEnsemble",
            "FreshPRINCE",
            "HIVECOTEV1",
            "HIVECOTEV2",
            "IndividualBOSS",
            "IndividualTDE",
            "KNeighborsTimeSeriesClassifier",
            "MUSE",
            "MatrixProfileClassifier",
            "ProbabilityThresholdEarlyClassifier",
            "ProximityForest",
            "ProximityStump",
            "ProximityTree",
            "RandomIntervalClassifier",
            "RandomIntervalSpectralEnsemble",
            "RocketClassifier",
            "ShapeDTW",
            "ShapeletTransformClassifier",
            "SignatureClassifier",
            "SklearnClassifierPipeline",
            "SummaryClassifier",
            "SupervisedTimeSeriesForest",
            "TSFreshClassifier",
            "TemporalDictionaryEnsemble",
            "TimeSeriesForestClassifier",
            "WEASEL"
    };
//</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="mtscSktimeClassifiers: Multivariate capable sktime classifiers"


    static String[] mtscSktimeClassifiers = {
            "Arsenal",
            "CanonicalIntervalForest",
            "Catch22Classifier",
            "CNNClassifier",
            "DrCIF",
            "FreshPRINCE",
            "HIVECOTEV2",
            "IndividualTDE",
            "KNeighborsTimeSeriesClassifier",
            "MLPClassifier",
            "MUSE",
            "RandomIntervalClassifier",
            "RocketClassifier",
            "ShapeletTransformClassifier",
            "SignatureClassifier",
            "SummaryClassifier",
            "TemporalDictionaryEnsemble",
            "TSFreshClassifier",
    };
//</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="utscSktimeClassifiers: Univariate classifiers set in estimator-evaluation through set_classifier()"
    static String[] utscSktimeClassifiers = {
            "Arsenal",
            "BOSSEnsemble",
            "CIF",  //Not Default
            "Catch22",  //Not Default
            "ComposableTimeSeriesForestClassifier",
            "ContractableBOSS",
            "CNNClassifier",
            "DrCIF", //Not Default
            "DummyClassifier",
            "ElasticEnsemble",
            "FreshPRINCE",
            "HIVECOTEV1",
            "HIVECOTEV2",
            "IndividualBOSS",
            "IndividualTDE",
            "KNeighborsTimeSeriesClassifier",
            "MatrixProfileClassifier",
            "Mini-ROCKET",
            "MLPClassifier",
            "Multi-ROCKET",
            "ProximityForest",
            "ProximityStump",
            "ProximityTree",
            "MUSE",
            "RandomInterval", //Not Default
            "RocketClassifier",
            "RISE",//Not Default
            "ShapeDTW",
            "SignatureClassifier",
            "STC", //Not Default
            "Summary", //Not Default
            "STSF",
            "TSF",
            "TSFresh",
            "TemporalDictionaryEnsemble",
            "WEASEL"
    };
//</editor-fold>


    //<editor-fold defaultstate="collapsed" desc="selectedClassifiers: specific subset of classifiers"
    static String[] selectedClassifiers = {
            "1NN-DTW",
            "BOSS",
            "Arsenal",
            "Catch22",
            "CNN",
            "cBOSS",
            "DrCIF",
            "FreshPRINCE",
            "HC1",
            "HC2",
            "Hydra",
            "ROCKET",
            "Mini-ROCKET",
            "Multi-ROCKET",
            "RDST",
            "STC",
            "STSF",
            "TDE",
            "TSF",
            "TSFresh",
            "WEASEL-Dilation"
    };
//</editor-fold>


    //Change name CNNClassifier -> CNN when finished
    static String[] clst = {"Arsenal","Catch22","CIF","CNNClassifier","DrCIF","FreshPRINCE","HC2","Mini-ROCKET\n" +
        "//        dimensionSelection();","Multi-ROCKET", "ROCKET", "STC","TDE"};//Fresh Prince, "CIF","MUSE",


    public static void collateResults(String resultsPath,String[] classifiers, String[] problems, int folds) throws Exception {
        for(int i=0;i<classifiers.length;i++) {
            String cls = classifiers[i];
            System.out.println("Processing classifier =" + cls);
            File f = new File(resultsPath + cls);
            if (f.isDirectory()) { //Check classifier directory exists.
                System.out.println("Base path " + resultsPath + cls + " exists");
                File stats = new File(resultsPath + cls + "/SummaryStats");
                //Delete any stats files there already
                String[] entries = stats.list();
                for (String s : entries) {
                    File currentFile = new File(stats.getPath(), s);
                    currentFile.delete();
                }
                stats.delete();
            }
            ClassifierResults res = new ClassifierResults("ResultsPath");

        }

    }

    /**
     * Collate individual results files to create accuracy by classifier files
     * @param path
     * @param classifiers
     * @param problems
     */
    public static void accuracyByClassifier(String path, String[] classifiers, String[] problems){
        File f = new File(path+"ByClassifier\\");
        f.mkdirs();
        for(String cls:classifiers){
           OutFile of = new OutFile(path+"ByClassifier\\"+cls+".csv");
           for(String str:problems){



           }
        }

    }
    String[] sota = {
            "HC2", "ROCKET", "Hydra", "WEASEL-dilation"};
    public static void main(String[] args) throws Exception {
//        referenceResults();
//        System.out.println(" Number of classifiers = "+ allSktimeClassifiers.length);
//          String path="X:\\Results Working Area\\MultivariateReferenceResults\\sktime\\";
          String path="C:\\Results Working Area\\DefaultClassifiers\\sktime\\";
          String[] allProblems = Stream.concat(Arrays.stream(DatasetLists.tscProblems112), Arrays.stream(DatasetLists.mtscProblems2022))
                .toArray(String[]::new);
//        collateResults(path, defaultCapableSktimeClassifiers, allProblems, 1);
        path="X:\\Results Working Area\\ReduxBakeoff\\sktime\\";
///DatasetLists.mtscProblems2022
          summariseResultsPresent(selectedClassifiers, DatasetLists.tscProblems112, path, "UnivariateCounts.csv");
//        summariseResultsPresent (defaultCapableSktimeClassifiers,allProblems,path);
 //       localResults(selectedClassifiers, DatasetLists.tscProblems112,path,"UnivariateResults",30);
//        summariseResultsPresent (selectedClassifiers,DatasetLists.eegProblems,path);
//        summariseAccuracies(selectedClassifiers,DatasetLists.eegProblems,path,30);

//        System.out.println(" Number of UTSC classifiers = "+ utscSktimeClassifiers.length);
//        path="X:\\Results Working Area\\UnivariateReferenceResults\\sktime\\";
//        summariseResultsPresent (utscSktimeClassifiers,DatasetLists.tscProblems112,path);
    }
    public static void buildHC2(){
        HIVE_COTE hc2 = new HIVE_COTE();
        hc2.setBuildIndividualsFromResultsFiles(true);
//        hc2.setResultsFileLocationParameters("C:/Temp/",problem,0);
//        hc2.setClassifiersNamesForFileRead(components);

    }

}
