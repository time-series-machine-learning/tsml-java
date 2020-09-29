package machine_learning.calibration;

import evaluation.MultipleClassifierEvaluation;
import evaluation.storage.ClassifierResults;
import evaluation.storage.ClassifierResultsCollection;
import experiments.Experiments;
import experiments.data.DatasetLists;
import tsml.classifiers.hybrids.HIVE_COTE;
import utilities.FileHandlingTools;
import weka.classifiers.Classifier;

import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;

public class CalibrationExperiments {

    public static ClassifierResults[] calibrateResults(ClassifierResults origTrainRes, ClassifierResults origTestRes) throws Exception {
        Calibrator calibrator = new DirichletCalibrator();

        long startTime = System.nanoTime();
        calibrator.buildCalibrator(origTrainRes);
        long calibbuildtime = System.nanoTime() - startTime;


        //train
        startTime = System.nanoTime();
        double[][] trainCalProbs = calibrator.calibrateInstances(origTrainRes);
        long trainCalibInstTime = (long)((System.nanoTime() - startTime) / (double)origTrainRes.numInstances());
        ClassifierResults calibTrainRes = CalibratedTSClassifier.buildCalibratorTrainResults(
                origTrainRes, "DirichletCalibrator_Unreg", trainCalProbs, calibbuildtime, trainCalibInstTime
        );


        //test
        startTime = System.nanoTime();
        double[][] testCalProbs = calibrator.calibrateInstances(origTestRes);
        long testCalibInstTime = (long)((System.nanoTime() - startTime) / (double)origTrainRes.numInstances());
        ClassifierResults calibTestRes = CalibratedTSClassifier.buildCalibratorTrainResults(
                origTestRes, "DirichletCalibrator_Unreg", testCalProbs, calibbuildtime, testCalibInstTime
        );

        return new ClassifierResults[] { calibTrainRes, calibTestRes };
    }

    public static void calibrateAllResults(String writeDir, ClassifierResultsCollection col) throws Exception {
        if (col.getNumMissingResults() > 0)
            throw new UnsupportedOperationException("currently not looking for mossing results");

        ClassifierResults[/*split*/][/*classifier*/][/*dset*/][/*fold*/] res = col.retrieveResults();


        for (int cid = 0; cid < col.getNumClassifiers(); cid++) {
            for (int did = 0; did < col.getNumDatasets(); did++) {
                for (int fid = 0; fid < col.getNumFolds(); fid++) {

                    ClassifierResults[] traintest = calibrateResults(res[0][cid][did][fid], res[1][cid][did][fid]);

                    String classifierName = col.getClassifierNamesInOutput()[cid] + "_calib";
                    String dsetName = col.getDatasetNamesInOutput()[did];
                    String fold = ""+col.getFolds()[fid];

                    File localWriteLoc = Paths.get(writeDir, classifierName, "Predictions", dsetName).toFile();
                    localWriteLoc.mkdirs();

                    traintest[0].writeFullResultsToFile(localWriteLoc.getAbsolutePath() + "/trainFold" + fold + ".csv");
                    traintest[1].writeFullResultsToFile(localWriteLoc.getAbsolutePath() + "/testFold" + fold + ".csv");

                }
            }
        }
    }


    public static void main(String[] args) throws Exception {

//        String[] classifiers = new String[] { "SVML", "C45" };
//        String[] datasets = Arrays.copyOfRange(DatasetLists.smallTSCProblems, 0, 3);
//        int numFolds = 3;

//        String[] classifiers = {"TSF","RISE","STC","cBOSS"};
//        String[] datasets = DatasetLists.tscProblems112;
//        int numFolds = 30;

        String[] classifiers = {"HIVE-COTE1.0", "HIVE-COTE1.0_innercalib"};
        String[] datasets = DatasetLists.tscProblems112;
        int numFolds = 30;

        String dataPath = "E:\\MyDocumentsE\\DATA\\Univariate_arff\\";
        String clsResPath = "E:\\MyDocumentsE\\RESULTS\\Calibration\\BaseResults\\";
        String calibResPath = "E:\\MyDocumentsE\\RESULTS\\Calibration\\CalibResults\\";
        String anaPath = "E:\\MyDocumentsE\\ANALYSIS\\Calibration\\";





        //uncalib results gen
        Experiments.ExperimentalArguments exp = new Experiments.ExperimentalArguments();
        exp.dataReadLocation = dataPath;
        exp.resultsWriteLocation = clsResPath;
        exp.generateErrorEstimateOnTrainSet = true;
        exp.forceEvaluation = false;
        Experiments.setupAndRunMultipleExperimentsThreaded(exp, classifiers, datasets, 0, numFolds);





        //calib results gen
//        ClassifierResultsCollection col = new ClassifierResultsCollection();
//        col.setCleanResults(false);
//
//        col.addClassifiers(classifiers, clsResPath);
//        col.setDatasets(datasets);
//        col.setFolds(numFolds);
//        col.setSplit_TrainTest();
//        col.load();
//
//        calibrateAllResults(calibResPath, col);









        //calib vs uncalib comparison
        for (String classifier : classifiers) {
            String expName = classifier + " Calib_" + datasets.length + "dsets";

            MultipleClassifierEvaluation mce = new MultipleClassifierEvaluation(anaPath, expName, numFolds);
            mce.setBuildMatlabDiagrams(false);
            mce.setTestResultsOnly(true);
            mce.setDatasets(datasets);
            mce.readInClassifier(classifier, classifier, clsResPath);
            mce.readInClassifier(classifier, classifier+"_calib", calibResPath);
            mce.runComparison();
        }
    }

}
