/*
 * This file is part of the UEA Time Series Machine Learning (TSML) toolbox.
 *
 * The UEA TSML toolbox is free software: you can redistribute it and/or 
 * modify it under the terms of the GNU General Public License as published 
 * by the Free Software Foundation, either version 3 of the License, or 
 * (at your option) any later version.
 *
 * The UEA TSML toolbox is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with the UEA TSML toolbox. If not, see <https://www.gnu.org/licenses/>.
 */

package experiments;

import tsml.classifiers.EnhancedAbstractClassifier;
import weka.core.Instances;

/**
 * Class for conducting experiments on a single problem
 * Wish list
 * 1. Cross validation: create a single output file with the cross validation predictions
 * 2. AUROC: take cross validation results and form the data for a AUROC plot
 * 3. Tuning: tune classifier on a train split
 * 4. Sensitivity: plot parameter space
 * 5. Robustness: performance with changing train set size (including acc estimates)
 */
public class SingleProblemExperiments {
    /**
     * Input, classifier, train set, test set, number of intervals (k)
     *
     * Train set will be resampled for k different train sizes at equally spaced intervals

     * Output to file results: TrainSize, TestAccActual, (TestAccEstimated, optional)
      */
    public static void increasingTrainSetSize(EnhancedAbstractClassifier c, Instances train, Instances test, int nIntervals, String results){
    // Work out intervals
        int fullLength=train.numInstances();
        int interval = fullLength/(nIntervals-1);
    //

    }

}
