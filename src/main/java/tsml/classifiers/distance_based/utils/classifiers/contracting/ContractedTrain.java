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
 
package tsml.classifiers.distance_based.utils.classifiers.contracting;

import tsml.classifiers.TrainTimeContractable;
import tsml.classifiers.TrainTimeable;

public interface ContractedTrain extends TrainTimeContractable, TimedTrain, ProgressiveBuild {

    /**
     * Is the classifier fully built? This is irrelevant of contract timings and is instead a reflection of whether
     * work remains and further time could be allocated to the classifier to build the model further.
     * @return
     */
    @Override boolean isFullyBuilt();

    long getTrainTimeLimit();

    default boolean hasTrainTimeLimit() {
        return getTrainTimeLimit() > 0;
    }

    /**
     * 
     * @param nanos the amount of time currently taken (or the expectation of how long something will take and thus
     *              whether there is enough time to complete it). E.g. this could be the current run time of a clsf plus
     *              the predicted time to do some unit of work to improve the classifier. The result would indicate if
     *              there's enough time to get this unit of work done within the contract and can therefore be used to
     *              decide whether to do it in the first place.
     * @return
     */
    default boolean insideTrainTimeLimit(long nanos) {
        return !hasTrainTimeLimit() || nanos < getTrainTimeLimit();
    }

    default boolean withinTrainContract(long time) {
        return insideTrainTimeLimit(time);
    }
    
    default long findRemainingTrainTime(long trainTime) {
        if(!hasTrainTimeLimit()) {
            return Long.MAX_VALUE;
        }
        final long trainTimeLimit = getTrainTimeLimit();
        return trainTimeLimit - trainTime;
    }
    
    default long findRemainingTrainTime() {
        return findRemainingTrainTime(getRunTime());
    }
}
