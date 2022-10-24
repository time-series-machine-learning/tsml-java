package contrib;

import evaluation.storage.ClassifierResults;
import evaluation.storage.EstimatorResultsCollection;
import experiments.data.DatasetLists;
import fileIO.OutFile;
import tsml.classifiers.hybrids.HIVE_COTE;

import java.io.File;
import java.util.HashSet;

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

    public static void localResults(String[] classifiers) throws Exception {
        String location = "C:\\Results Working Area\\MultivariateReferenceResults\\sktime\\";
        evaluation.MultipleEstimatorEvaluation x = new evaluation.MultipleEstimatorEvaluation(location, "AllResults", 10);

//        System.arraycopy(d1,0,datasets,0,d1.length);
//        System.arraycopy(d2,0,datasets,d1.length,d2.length);
        ClassifierResults.printOnFailureToLoad = false;
        x.setIgnoreMissingResults(true);
        x.setBuildMatlabDiagrams(true);
        x.setTestResultsOnly(true);
        x.setResultsType(EstimatorResultsCollection.ResultsType.CLASSIFICATION);
        String loc = location;
        String[] d = DatasetLists.mtscProblems2022;
        System.out.println(" number problems = "+d.length);
 //       String[] clst = {"TDE","DrCIF","Arsenal","STC","Mini-ROCKET","Multi-ROCKET","ROCKET"};//,HC2"."FreshPRINCE","MUSE"}; //,,"",,"TDE","CIF","HC1","TDE","HC2","STSF"};

//        String[] clst = {"Arsenal","cBOSS","CIF","DrCIF","DTW-A","DTW-D","DTW-I","gRSF","HC1","HC2","InceptionTime","MrSEQL","MUSE","ResNet","RISE","ROCKET","STC","TDE","TSF"}; //,"TDE", "DrCIF", "CIF","HC1","TDE","HC2","STSF"};
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

    public static void summariseResultsPresent(String[] classifiers, String[] problems, String path){
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
        OutFile out= new OutFile(path+"completeCounts.csv");
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
    static String[] allMultivariate = {
            "Arsenal",
            "CNN",
            "CIF",
            "Catch22",
            "DrCIF",
            "FreshPRINCE",
            "HC2",
            "IndividualTDE",
            "KNeighborsTimeSeriesClassifier",
            "MUSE",
            "ProbabilityThresholdEarlyClassifier",
            "RandomInterval",
            "ROCKET",
            "STC",
            "Signature",
            "Summary",
            "TSFresh",
            "TDE",
            "WeightedEnsemble",
};//Fresh Prince, "CIF",
    static String[] mtscSktimeClassifiers = {
            "Arsenal",
            "CIF",  //Not Default
            "Catch22",  //Not Default
            "CIF",  //Not Default
            "CNNClassifier",
            "DrCIF", //Not Default
            "FreshPRINCE",
            "HIVECOTEV2",
            "IndividualTDE",
            "KNeighborsTimeSeriesClassifier",
            "Mini-ROCKET",
            "MLPClassifier",
            "Multi-ROCKET",
            "MUSE",
            "RandomInterval", //Not Default
            "RocketClassifier",
            "STC", //Not Default
            "SignatureClassifier",
            "Summary", //Not Default
            "TemporalDictionaryEnsemble",
            "TSFresh",
    };
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
    

    
//Change name CNNClassifier -> CNN when finished
    static String[] clst = {"Arsenal","Catch22","CIF","CNNClassifier","DrCIF","FreshPRINCE","HC2","Mini-ROCKET\n" +
        "//        dimensionSelection();","Multi-ROCKET", "ROCKET", "STC","TDE"};//Fresh Prince, "CIF","MUSE",
    public static void main(String[] args) throws Exception {
//        referenceResults();
 //       localResults(referenceClassifiers);
        System.out.println(" Number of MTSC classifiers = "+ mtscSktimeClassifiers.length);
        String path="X:\\Results Working Area\\MultivariateReferenceResults\\sktime\\";
        summariseResultsPresent (mtscSktimeClassifiers,DatasetLists.mtscProblems2022,path);
        System.out.println(" Number of UTSC classifiers = "+ utscSktimeClassifiers.length);
        path="X:\\Results Working Area\\UnivariateReferenceResults\\sktime\\";
        summariseResultsPresent (utscSktimeClassifiers,DatasetLists.tscProblems112,path);
    }
    public static void buildHC2(){
        HIVE_COTE hc2 = new HIVE_COTE();
        hc2.setBuildIndividualsFromResultsFiles(true);
//        hc2.setResultsFileLocationParameters("C:/Temp/",problem,0);
//        hc2.setClassifiersNamesForFileRead(components);

    }

}
