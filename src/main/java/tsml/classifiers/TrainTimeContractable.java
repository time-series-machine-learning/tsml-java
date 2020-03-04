/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package tsml.classifiers;

import tsml.classifiers.distance_based.utils.params.ParamHandler;
import tsml.classifiers.distance_based.utils.params.ParamSet;

import java.util.concurrent.TimeUnit;

/**
 * Interface that allows the user to impose a train time contract of a classifier that
    implements this interface

known classifiers: ShapeletTransformClassifier, RISE (not tested) HIVE_COTE (partial),
* BOSS, TSF , ContractRotationForest
 *
 * ********************************NOTES********************************
 * 1) contract time of <=0 means no contract!
 */
public interface TrainTimeContractable
    extends ParamHandler, TrainTimeable {
    default void setOneDayLimit(){ setTrainTimeLimit(TimeUnit.DAYS, 1); }
    
    default void setOneHourLimit(){ setTrainTimeLimit(TimeUnit.HOURS, 1); }

    default void setOneMinuteLimit(){ setTrainTimeLimit(TimeUnit.MINUTES, 1); }
    
    default void setDayLimit(int t){ setTrainTimeLimit(TimeUnit.DAYS, t); }

    default void setHourLimit(int t){ setTrainTimeLimit(TimeUnit.HOURS, t); }
    
    default void setMinuteLimit(int t){ setTrainTimeLimit(TimeUnit.MINUTES, t); }

    //set any value in nanoseconds you like.
    // this is the method which should be implemented in sub classes. Other time setting methods wrap around this
    default void setTrainTimeLimit(long time) { throw new UnsupportedOperationException(); }

    //pass in an value from the TimeUnit enum and the amount of said values.
    default void setTrainTimeLimit(TimeUnit time, long amount) {
        setTrainTimeLimitNanos(TimeUnit.NANOSECONDS.convert(amount, time));
    }

    default void setTrainTimeLimit(long amount, TimeUnit time) {
        setTrainTimeLimit(time, amount);
    }

    default long getTrainTimeLimitNanos() {
        throw new UnsupportedOperationException();
    }

    default void setTrainTimeLimitNanos(long nanos) {
        setTrainTimeLimit(nanos);
    }

    /**
     * whether there's a train time limit set. Train time limit must be >0, otherwise <=0 is assumed to be no train
     * time limit.
     * @return
     */
    default boolean hasTrainTimeLimit() {
        return getTrainTimeLimitNanos() > 0;
    }

    default void removeTrainTimeLimit() {
        setTrainTimeLimit(-1);
    }

    /**
     * get the remaining train time in nanoseconds
     * @return
     */
    default long getRemainingTrainTimeNanos() {
        long result = getTrainTimeLimitNanos() - getTrainTimeNanos();
        return result;
    }

    /**
     * checks whether there is remaining train time. If there is no train time limit then there is always remaining
     * time.
     * @return
     */
    default boolean hasRemainingTrainTime() {
        if(!hasTrainTimeLimit()) {
            return true; // if there's no train time limit then there's always remaining train time
        }
        long prediction = predictNextTrainTimeNanos();
        return getRemainingTrainTimeNanos() > prediction;
    }

    /**
     * predicts the time required for the next stage of training. This is used in calculating whether there's
     * remaining train time. Defaults to return 0 as we do not know how long the next step will take, therefore we
     * assume we can train further as long as there is >0 nanoseconds remaining.
     * @return
     */
    default long predictNextTrainTimeNanos() {
        return 0;
    }

    /**
     * Method of knowing whether the classifier is fully built and will do no more work. This is required as setting
     * a 5 min contract may lead to the classifier only using 4.5mins, say. We need to know when it is done so we can
     * use the remaining .5 mins appropriately for other work rather than spinning on this classifier until 5 mins is
     * reached. The earlier the classifier abandons the more of a problem this becomes.
     * @return
     */
    default boolean isBuilt() {
        throw new UnsupportedOperationException();
    }

    /**
     * Whether there is remaining training work to be done. This comes down to two things: a) whether the classifier
     * has work to do and b) whether there's time remaining to get that work done.
     * @return
     */
    default boolean hasRemainingTraining() {
        return !isBuilt() && hasRemainingTrainTime();
    }

    // helper methods for setting / getting params of the contract

    static String getTrainTimeLimitNanosFlag() {
        return "trtl";
    }

    @Override default ParamSet getParams() {
        return ParamHandler.super.getParams().add(getTrainTimeLimitNanosFlag(), getTrainTimeLimitNanos());
    }

    @Override default void setParams(ParamSet param) {
        ParamHandler.setParam(param, getTrainTimeLimitNanosFlag(), this::setTrainTimeLimitNanos, Long.class);
    }

}
