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

import experiments.Experiments;
import tsml.classifiers.distance_based.utils.strings.StrUtils;
import tsml.data_containers.TimeSeriesInstances;
import tsml.data_containers.ts_fileIO.TSReader;

import java.io.File;
import java.io.FileReader;
import java.util.logging.Level;
import java.util.logging.Logger;


public class DatasetLoadingTS {

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
        DatasetLoadingTS.LOXO_ATT_ID = LOXO_ATT_ID;
    }

    public static double getProportionKeptForTraining() {
        return proportionKeptForTraining;
    }

    public static void setProportionKeptForTraining(double proportionKeptForTraining) {
        DatasetLoadingTS.proportionKeptForTraining = proportionKeptForTraining;
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
    public static TimeSeriesInstances[] sampleDataset(String parentFolder, String problem, int fold) throws Exception {
        parentFolder = StrUtils.asDirPath(parentFolder + "/" + problem);
        TimeSeriesInstances[] data = new TimeSeriesInstances[2];
        TSReader ts_reader_train = new TSReader(new FileReader(new File(parentFolder + "/" + problem  + "_TRAIN" + ".ts")));
        TSReader ts_reader_test = new TSReader(new FileReader(new File(parentFolder + "/" + problem  + "_TEST" + ".ts")));
        data[0] = ts_reader_train.GetInstances();
        data[1] = ts_reader_test.GetInstances();


        return data;
    }














}
