/*
 * Copyright (C) 2019 xmw13bzu
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

package experiments.data;

import experiments.ClassifierLists;
import experiments.Experiments;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Map;
import java.util.TreeMap;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.regex.Pattern;

import tsml.classifiers.distance_based.utils.strings.StrUtils;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import utilities.multivariate_tools.MultivariateInstanceTools;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ArffSaver;

/**
 * Class for handling the loading of datasets, from disk and the baked-in example datasets
 *
 * @author James Large (james.large@uea.ac.uk)
 */
public class DatasetLoading {

    private final static Logger LOGGER = Logger.getLogger(Experiments.class.getName());

    private static final String BAKED_IN_DATA_MASTERPATH = "src/main/java/experiments/data/";

    public static final String BAKED_IN_UCI_DATA_PATH = BAKED_IN_DATA_MASTERPATH + "uci/";
    public static final String BAKED_IN_TSC_DATA_PATH = BAKED_IN_DATA_MASTERPATH + "tsc/";
    public static final String BAKED_IN_MTSC_DATA_PATH = BAKED_IN_DATA_MASTERPATH + "mtsc/";

    public static final String[] BAKED_IN_UCI_DATASETS = { "iris", "hayes-roth", "teaching" };
    public static final String[] BAKED_IN_TSC_DATASETS = { "ItalyPowerDemand", "Beef" };
    public static final String[] BAKED_IN_MTSC_DATASETS = { "BasicMotions" };

    private static String LOXO_ATT_ID = "experimentsSplitAttribute";
    private static double proportionKeptForTraining = 0.5;

    private static int MAX_DECIMAL_PLACES = Integer.MAX_VALUE;

    private static boolean debug = false;

    public static String getLeaveOneXOutAttributeID() {
        return LOXO_ATT_ID;
    }

    public static void setLeaveOneXOutAttributeID(String LOXO_ATT_ID) {
        DatasetLoading.LOXO_ATT_ID = LOXO_ATT_ID;
    }

    public static double getProportionKeptForTraining() {
        return proportionKeptForTraining;
    }

    public static void setProportionKeptForTraining(double proportionKeptForTraining) {
        DatasetLoading.proportionKeptForTraining = proportionKeptForTraining;
    }



    public static void setDebug(boolean d) {
        debug = d;

        if (debug)
            LOGGER.setLevel(Level.FINEST);
        else
            LOGGER.setLevel(Level.INFO);
    }
    public static boolean getDebug() {
        return debug;
    }


    /**
     * Helper function for loading the baked-in ItalyPowerDemand dataset, one of the
     * UCR datasets for TSC
     *
     * http://timeseriesclassification.com/description.php?Dataset=ItalyPowerDemand
     *
     * UCR data comes with predefined fold 0 splits. If a seed of 0 is given, that exact split is returned.
     * Train/test distributions are maintained between resamples.
     *
     * @param seed the seed for resampling the data.
     * @return new Instances[] { trainSet, testSet };
     * @throws Exception if data loading or sampling failed
     */
    public static Instances[] sampleItalyPowerDemand(int seed) throws Exception {
        return sampleDataset(BAKED_IN_TSC_DATA_PATH, "ItalyPowerDemand", seed);
    }

    public static Instances loadItalyPowerDemand() throws Exception {
        final Instances[] instances = sampleItalyPowerDemand(0);
        instances[0].addAll(instances[1]);
        return instances[0];
    }

    public static Instances[] sampleGunPoint(int seed) throws Exception {
        return sampleDataset(BAKED_IN_TSC_DATA_PATH, "GunPoint", seed);
    }

    public static Instances loadGunPoint() throws Exception {
        final Instances[] instances = sampleGunPoint(0);
        instances[0].addAll(instances[1]);
        return instances[0];
    }

    /**
     * Helper function for loading the baked-in Beef dataset, one of the
     * UCR datasets for TSC
     *
     * http://timeseriesclassification.com/description.php?Dataset=Beef
     *
     * UCR data comes with predefined fold 0 splits. If a seed of 0 is given, that exact split is returned.
     * Train/test distributions are maintained between resamples.
     *
     * @param seed the seed for resampling the data.
     * @return new Instances[] { trainSet, testSet };
     * @throws Exception if data loading or sampling failed
     */
    public static Instances[] sampleBeef(int seed) throws Exception {
        return sampleDataset(BAKED_IN_TSC_DATA_PATH, "Beef", seed);
    }

    public static Instances loadBeef() throws Exception {
        final Instances[] instances = sampleBeef(0);
        instances[0].addAll(instances[1]);
        return instances[0];
    }

    /**
     * Helper function for loading the baked-in BasicMotions dataset, one of the
     * UEA datasets for MTSC
     *
     * http://timeseriesclassification.com/description.php?Dataset=BasicMotions
     *
     * UEA-MTSC data comes with predefined fold 0 splits. If a seed of 0 is given, that exact split is returned.
     * Train/test distributions are maintained between resamples.
     *
     * @param seed the seed for resampling the data
     * @return new Instances[] { trainSet, testSet };
     * @throws Exception if data loading or sampling failed
     */
    public static Instances[] sampleBasicMotions(int seed) throws Exception {
        return sampleDataset(BAKED_IN_MTSC_DATA_PATH, "BasicMotions", seed);
    }

    /**
     * Helper function for loading the baked-in Iris dataset, one of the classical
     * UCI datasets for general classification
     *
     * https://archive.ics.uci.edu/ml/datasets/iris
     *
     * UCI data comes in a single file. The proportion of data kept for training is
     * defined by the static proportionKeptForTraining, default = 0.5
     *
     * @param seed the seed for resampling the data.
     * @return new Instances[] { trainSet, testSet };
     * @throws Exception if data loading or sampling failed
     */
    public static Instances[] sampleIris(int seed) throws Exception {
        return sampleDataset(BAKED_IN_UCI_DATA_PATH, "iris", seed);
    }

    /**
     * Helper function for loading the baked-in Hayes-Roth dataset, one of the classical
     * UCI datasets for general classification
     *
     * https://archive.ics.uci.edu/ml/datasets/Hayes-Roth
     *
     * UCI data comes in a single file. The proportion of data kept for training is
     * defined by the static proportionKeptForTraining, default = 0.5
     *
     * @param seed the seed for resampling the data.
     * @return new Instances[] { trainSet, testSet };
     * @throws Exception if data loading or sampling failed
     */
    public static Instances[] sampleHayesRoth(int seed) throws Exception {
        return sampleDataset(BAKED_IN_UCI_DATA_PATH, "hayes-roth", seed);
    }



    /**
     * This method will return a train/test split of the problem, resampled with the fold ID given.
     *
     * Currently, there are four ways to load datasets. These will be attempted from
     * top to bottom, in an order designed to make the fewest assumptions
     * possible about the nature of the split, in terms of potential differences in class distributions,
     * train and test set sizes, etc.
     *
     * 1) if predefined splits are found at the specified location, in the form dataLocation/dsetName/dsetName0_TRAIN and TEST,
     *      these will be loaded and used as they are, OTHERWISE...
     * 2) if a predefined fold0 split is given as in the UCR archive, and fold0 is being experimented on, the split exactly as it is defined will be used.
     *      For fold != 0, the fold0 split is combined and resampled, maintaining the original train and test distributions. OTHERWISE...
     * 3) if only a single file is found containing all the data, this dataset is  stratified randomly resampled with proportionKeptForTraining (default=0.5)
     *      instances reserved for the _TRAIN_ set. OTHERWISE...
     * 4) if the dataset loaded has a first attribute whose name _contains_ the string "experimentsSplitAttribute".toLowerCase()
     *      then it will be assumed that we want to perform a leave out one X cross validation. Instances are sampled such that fold N is comprised of
     *      a test set with all instances with first-attribute equal to the Nth unique value in a sorted list of first-attributes. The train
     *      set would be all other instances. The first attribute would then be removed from all instances, so that they are not given
     *      to the classifier to potentially learn from. It is up to the user to ensure the the foldID requested is within the range of possible
     *      values 1 to numUniqueFirstAttValues OTHERWISE...
     * 5) error
     *
     * @return new Instances[] { trainSet, testSet };
     */
    public static Instances[] sampleDataset(String parentFolder, String problem, int fold) throws Exception {
        parentFolder = StrUtils.asDirPath(parentFolder);
        Instances[] data = new Instances[2];
        File trainFile = new File(parentFolder + problem + "/" + problem + fold + "_TRAIN.arff");
        File testFile = new File(parentFolder + problem + "/" + problem + fold + "_TEST.arff");
        boolean predefinedSplitsExist = trainFile.exists() && testFile.exists();
        if (predefinedSplitsExist) {
            // CASE 1)
            data[0] = loadDataThrowable(trainFile);
            data[1] = loadDataThrowable(testFile);
            LOGGER.log(Level.FINE, problem + " loaded from predefined folds.");
        } else {
            trainFile = new File(parentFolder + problem + "/" + problem + "_TRAIN.arff");
            testFile = new File(parentFolder + problem + "/" + problem + "_TEST.arff");
            boolean predefinedFold0Exists = trainFile.exists() && testFile.exists();
            if (predefinedFold0Exists) {
                // CASE 2)
                data[0] = loadDataThrowable(trainFile);
                data[1] = loadDataThrowable(testFile);
                if(fold!=0)
//                    data = InstanceTools.resampleTrainAndTestInstances(data[0], data[1], fold);
//                data = InstanceTools.resampleTrainAndTestInstances(data[0], data[1], fold);
                if (data[0].checkForAttributeType(Attribute.RELATIONAL)) {
//                    data = MultivariateInstanceTools.resampleMultivariateTrainAndTestInstances(data[0], data[1], fold);
                    data = MultivariateInstanceTools.resampleMultivariateTrainAndTestInstances_old(data[0], data[1], fold);

                } else {
                    data = InstanceTools.resampleTrainAndTestInstances(data[0], data[1], fold);
                }

                LOGGER.log(Level.FINE, problem + " resampled from predfined fold0 split.");
            } else {
                // We only have a single file with all the data
                Instances all = null;
                try {
                    all = DatasetLoading.loadDataThrowable(parentFolder + problem + "/" + problem);
                } catch (IOException io) {
                    String msg = "Could not find the dataset \"" + problem + "\" in any form at the path\n" + parentFolder + "\n" + "The IOException: " + io;
                    LOGGER.log(Level.SEVERE, msg, io);
                }
                boolean needToDefineLeaveOutOneXFold = all.attribute(0).name().toLowerCase().contains(LOXO_ATT_ID.toLowerCase());
                if (needToDefineLeaveOutOneXFold) {
                    // CASE 4)
                    data = splitDatasetByFirstAttribute(all, fold);
                    LOGGER.log(Level.FINE, problem + " resampled from full data file.");
                } else {
                    // CASE 3)
                    if (all.checkForAttributeType(Attribute.RELATIONAL)) {
                        data = MultivariateInstanceTools.resampleMultivariateInstances(all, fold, proportionKeptForTraining);
                    } else {
                        data = InstanceTools.resampleInstances(all, fold, proportionKeptForTraining);
                    }
                    LOGGER.log(Level.FINE, problem + " resampled from full data file.");
                }
            }
        }
        return data;
    }

    /**
     * If the dataset loaded has a first attribute whose name _contains_ the string "experimentsSplitAttribute".toLowerCase()
     * then it will be assumed that we want to perform a leave out one X cross validation. Instances are sampled such that fold N is comprised of
     * a test set with all instances with first-attribute equal to the Nth unique value in a sorted list of first-attributes. The train
     * set would be all other instances. The first attribute would then be removed from all instances, so that they are not given
     * to the classifier to potentially learn from. It is up to the user to ensure the the foldID requested is within the range of possible
     * values 1 to numUniqueFirstAttValues
     *
     * @return new Instances[] { trainSet, testSet };
     */
    public static Instances[] splitDatasetByFirstAttribute(Instances all, int foldId) {
        TreeMap<Double, Integer> splitVariables = new TreeMap<>();
        for (int i = 0; i < all.numInstances(); i++) {
            //even if it's a string attribute, this val corresponds to the index into the array of possible strings for this att
            double key= all.instance(i).value(0);
            Integer val = splitVariables.get(key);
            if (val == null)
                val = 0;
            splitVariables.put(key, ++val);
        }

        //find the split attribute value to keep for testing this fold
        double idToReserveForTestSet = -1;
        int testSize = -1;
        int c = 0;
        for (Map.Entry<Double, Integer> splitVariable : splitVariables.entrySet()) {
            if (c++ == foldId) {
                idToReserveForTestSet = splitVariable.getKey();
                testSize = splitVariable.getValue();
            }
        }

        //make the split
        Instances train = new Instances(all, all.size() - testSize);
        Instances test  = new Instances(all, testSize);
        for (int i = 0; i < all.numInstances(); i++)
            if (all.instance(i).value(0) == idToReserveForTestSet)
                test.add(all.instance(i));
        train.addAll(all);

        //delete the split attribute
        train.deleteAttributeAt(0);
        test.deleteAttributeAt(0);

        return new Instances[] { train, test };
    }


    /**
     * Loads the arff file at the target location and sets the last attribute to be the class value,
     * or throws IOException on any error.
     *
     * @param fullPath path to the file to try and load
     * @return Instances from file.
     * @throws java.io.IOException if cannot find the file, or file is malformed
     */
    public static Instances loadDataThrowable(String fullPath) throws IOException {
        return loadDataThrowable(new File(fullPath));
    }

    /**
     * Loads the arff file at the target location and sets the last attribute to be the class value,
     * or throws IOException on any error.
     *
     * @param targetFile the file to try and load
     * @return Instances from file.
     * @throws java.io.IOException if cannot find the file, or file is malformed
     */
    public static Instances loadDataThrowable(File targetFile) throws IOException {
        String[] parts = targetFile.getName().split(Pattern.quote("."));
        String extension = "";
        final String ARFF = ".arff", TS = ".ts";

        if (parts.length == 2) {
            extension = "." + parts[1]; //split will remove the .
        }
        else {
            //have not been given a specific extension
            //look for arff or ts formats
            //arbitrarily looking for arff first
            File newtarget = new File(targetFile.getAbsolutePath() + ARFF);
            if (newtarget.exists()) {
                targetFile = newtarget;
                extension = ARFF;
            }
            else {
                newtarget = new File(targetFile.getAbsolutePath() + TS);
                if (newtarget.exists()) {
                    targetFile = newtarget;
                    extension = TS;
                }
                else
                    throw new IOException("Cannot find file " + targetFile.getAbsolutePath() + " with either .arff or .ts extensions.");
            }
        }

        Instances inst = null;
        FileReader reader = new FileReader(targetFile);

        if (extension.toLowerCase().equals(ARFF)) {
            inst = new Instances(reader);
        }
        else if (extension.toLowerCase().equals(TS)) {
            TSReader tsreader = new TSReader(reader);
            inst = tsreader.GetInstances();
        }

        inst.setClassIndex(inst.numAttributes() - 1);
        reader.close();

        return inst;
    }


    /**
     * Loads the arff file at the target location and sets the last attribute to be the class value,
     * or returns null on any error, such as not finding the file or it being malformed
     *
     * @param fullPath path to the file to try and load
     * @return Instances from file.
     */
    public static Instances loadData(String fullPath) {
        return loadDataNullable(new File(fullPath));
    }



    /**
     * Loads the arff file at the target location and sets the last attribute to be the class value,
     * or returns null on any error, such as not finding the file or it being malformed
     *
     * @param fullPath path to the file to try and load
     * @return Instances from file.
     */
    public static Instances loadDataNullable(String fullPath) {
        return loadDataNullable(new File(fullPath));
    }

    /**
     * Loads the arff file at the target location and sets the last attribute to be the class value,
     * or returns null on any error, such as not finding the file or it being malformed
     *
     * @param targetFile the file to try and load
     * @return Instances from file.
     */
    public static Instances loadDataNullable(File targetFile) {
        try {
            return loadDataThrowable(targetFile);
        } catch (IOException e) {
            System.out.println("Unable to load data on path " + targetFile.getAbsolutePath() + " Exception thrown =" + e);
            return null;
        }
    }

    /**
     *  Simple util to saveDatasets out. Useful for shapelet transform.
     *
     * @param dataSet
     * @param fileName
     */
    public static void saveDataset(Instances dataSet, String fileName) {
        try {
            ArffSaver saver = new ArffSaver();
            saver.setMaxDecimalPlaces(MAX_DECIMAL_PLACES);
            saver.setInstances(dataSet);
            if (fileName.endsWith(".arff")) {
                saver.setFile(new File(fileName));
            } else {
                saver.setFile(new File(fileName + ".arff"));
            }
            saver.writeBatch();
        } catch (IOException ex) {
            System.out.println("Error saving transformed dataset" + ex);
        }
    }






















    public static void main(String[] args) throws Exception {
//        tests();

        DatasetLoading.sampleItalyPowerDemand(0);
    }

    private static boolean quickEval(Instances insts) throws Exception {
        Classifier ed = ClassifierLists.setClassifierClassic("ED", 0);
        ed.buildClassifier(insts);
        return ClassifierTools.accuracy(insts, ed) == 1.0;
    }

    private static void assert_t(boolean result) throws Exception {
        if (!result) //todo reassess how the proper assert works...
            throw new Exception("Hacky assert failed");
    }

    /**
     * Obvious candidate for moving over to proper unit tests when codebase updates 
     * to incorporate them properly
     */
    private static void tests() throws Exception {

        //should handle both with/without extension
        String dataPath = BAKED_IN_UCI_DATA_PATH + "iris/iris";
        System.out.println("Testing: testARFFLoad("+dataPath+")");
        if (testARFFLoad(dataPath))
            System.out.println("Passed: testARFFLoad("+dataPath+")\n");

        dataPath += ".arff";
        System.out.println("Testing: testARFFLoad("+dataPath+")");
        if (testARFFLoad(dataPath))
            System.out.println("Passed: testARFFLoad("+dataPath+")\n");

        System.out.println("Testing: testUCILoad()");
        if (testUCILoad())
            System.out.println("Passed: testUCILoad()\n");

        // 0 loads default split, 1 will resample
        System.out.println("Testing: testTSCLoad("+0+")");
        if (testTSCLoad(0))
            System.out.println("Passed: testTSCLoad("+0+")\n");

        System.out.println("Testing: testTSCLoad("+1+")");
        if (testTSCLoad(1))
            System.out.println("Passed: testTSCLoad("+1+")\n");

        System.out.println("Testing: testMTSCLoad("+0+")");
        if (testMTSCLoad(0))
            System.out.println("Passed: testMTSCLoad("+0+")\n");

        System.out.println("Testing: testMTSCLoad("+1+")");
        if (testMTSCLoad(1))
            System.out.println("Passed: testMTSCLoad("+1+")\n");
    }

    private static boolean testARFFLoad(String dataPath) throws Exception {

        Instances data = DatasetLoading.loadDataThrowable(dataPath);

        assert_t(data != null);
        assert_t(data.relationName().equals("iris"));
        assert_t(data.numInstances() == 150);
        assert_t(data.numAttributes() == 5);
        assert_t(data.numClasses() == 3);
        assert_t(data.classIndex() == data.numAttributes()-1);

        assert_t(quickEval(data));

        return true;
    }

    private static boolean testUCILoad() throws Exception {
        proportionKeptForTraining = 0.5;
        Instances[] data = sampleIris(0);

        assert_t(data != null);

        assert_t(data[0] != null);
        assert_t(data[0].relationName().equals("iris"));
        assert_t(data[0].numInstances() == 75);
        assert_t(data[0].numAttributes() == 5);
        assert_t(data[0].numClasses() == 3);
        assert_t(data[0].classIndex() == data[0].numAttributes()-1);

        assert_t(data[1] != null);
        assert_t(data[1].relationName().equals("iris"));
        assert_t(data[1].numInstances() == 75);
        assert_t(data[1].numAttributes() == 5);
        assert_t(data[1].numClasses() == 3);
        assert_t(data[1].classIndex() == data[1].numAttributes()-1);

        assert_t(quickEval(data[0]));
        assert_t(quickEval(data[1]));

        return true;
    }

    private static boolean testTSCLoad(int seed) throws Exception {
        //2 class
        Instances[] data = sampleItalyPowerDemand(seed);

        assert_t(data != null);

        assert_t(data[0] != null);
        assert_t(data[0].relationName().equals("ItalyPowerDemand"));
        assert_t(data[0].numInstances() == 67);
        assert_t(data[0].numAttributes() == 25);
        assert_t(data[0].numClasses() == 2);
        assert_t(data[0].classIndex() == data[0].numAttributes()-1);

        assert_t(data[1] != null);
        assert_t(data[1].relationName().equals("ItalyPowerDemand"));
        assert_t(data[1].numInstances() == 1029);
        assert_t(data[1].numAttributes() == 25);
        assert_t(data[1].numClasses() == 2);
        assert_t(data[1].classIndex() == data[1].numAttributes()-1);

        assert_t(quickEval(data[0]));
        assert_t(quickEval(data[1]));

        return true;
    }

    private static boolean testMTSCLoad(int seed) throws Exception {
        Instances[] data = sampleBasicMotions(seed);

        assert_t(data != null);

        assert_t(data[0] != null);
        assert_t(data[0].relationName().equals("BasicMotions"));
        assert_t(data[0].numInstances() == 40);
        assert_t(data[0].attribute(0).isRelationValued());
        assert_t(data[0].attribute(0).relation().numAttributes() == 100);
        assert_t(data[0].numClasses() == 4);
        assert_t(data[0].classIndex() == data[0].numAttributes()-1);

        assert_t(data[1] != null);
        assert_t(data[1].relationName().equals("BasicMotions"));
        assert_t(data[1].numInstances() == 40);
        assert_t(data[1].attribute(0).isRelationValued());
        assert_t(data[1].attribute(0).relation().numAttributes() == 100);
        assert_t(data[1].numClasses() == 4);
        assert_t(data[1].classIndex() == data[1].numAttributes()-1);

        assert_t(quickEval(data[0]));
        assert_t(quickEval(data[1]));

        return true;
    }


}
