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
import java.io.File;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import utilities.multivariate_tools.MultivariateInstanceTools;
import weka.core.Attribute;
import weka.core.Instances;

/**
 * Class for handling the loading of datasets, from disk and the baked-in example datasets
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public class DataLoading {

    private final static Logger LOGGER = Logger.getLogger(Experiments.class.getName());

    public static boolean debug = false;
    
    
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
    public static Instances[] sampleDataset(String parentFolder, String problem, int fold) throws Exception {
        Instances[] data = new Instances[2];
        File trainFile = new File(parentFolder + problem + "/" + problem + fold + "_TRAIN.arff");
        File testFile = new File(parentFolder + problem + "/" + problem + fold + "_TEST.arff");
        boolean predefinedSplitsExist = trainFile.exists() && testFile.exists();
        if (predefinedSplitsExist) {
            // CASE 1)
            data[0] = ClassifierTools.loadData(trainFile);
            data[1] = ClassifierTools.loadData(testFile);
            LOGGER.log(Level.FINE, problem + " loaded from predfined folds.");
        } else {
            trainFile = new File(parentFolder + problem + "/" + problem + "_TRAIN.arff");
            testFile = new File(parentFolder + problem + "/" + problem + "_TEST.arff");
            boolean predefinedFold0Exists = trainFile.exists() && testFile.exists();
            if (predefinedFold0Exists) {
                // CASE 2)
                data[0] = ClassifierTools.loadData(trainFile);
                data[1] = ClassifierTools.loadData(testFile);
                if (data[0].checkForAttributeType(Attribute.RELATIONAL)) {
                    data = MultivariateInstanceTools.resampleMultivariateTrainAndTestInstances(data[0], data[1], fold);
                } else {
                    data = InstanceTools.resampleTrainAndTestInstances(data[0], data[1], fold);
                }
                LOGGER.log(Level.FINE, problem + " resampled from predfined fold0 split.");
            } else {
                // We only have a single file with all the data
                Instances all = null;
                try {
                    all = ClassifierTools.loadDataThrowable(parentFolder + problem + "/" + problem);
                } catch (IOException io) {
                    String msg = "Could not find the dataset \"" + problem + "\" in any form at the path\n" + parentFolder + "\n" + "The IOException: " + io;
                    LOGGER.log(Level.SEVERE, msg, io);
                }
                boolean needToDefineLeaveOutOneXFold = all.attribute(0).name().toLowerCase().contains(Experiments.LOXO_ATT_ID.toLowerCase());
                if (needToDefineLeaveOutOneXFold) {
                    // CASE 4)
                    data = Experiments.splitDatasetByFirstAttribute(all, fold);
                    LOGGER.log(Level.FINE, problem + " resampled from full data file.");
                } else {
                    // CASE 3)
                    if (all.checkForAttributeType(Attribute.RELATIONAL)) {
                        data = MultivariateInstanceTools.resampleMultivariateInstances(all, fold, Experiments.proportionKeptForTraining);
                    } else {
                        data = InstanceTools.resampleInstances(all, fold, Experiments.proportionKeptForTraining);
                    }
                    LOGGER.log(Level.FINE, problem + " resampled from full data file.");
                }
            }
        }
        return data;
    }

}
