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

import utilities.StopWatch;
import utilities.params.ParamHandler;
import utilities.params.ParamSet;

import java.util.concurrent.TimeUnit;

/**
 * Interface that allows the user to impose a train time contract of a classifier that
    implements this interface

known classifiers: ShapeletTransformClassifier, RISE (not tested) HiveCote (partial),
* BOSS (check), TSF (check)
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

    default boolean hasTrainTimeLimit() {
        return getTrainTimeLimitNanos() >= 0;
    }

    default long getRemainingTrainTimeNanos() {
        long result = getTrainTimeLimitNanos() - getTrainTimeNanos();
        return result;
    }

    default boolean hasRemainingTrainTime() {
        if(!hasTrainTimeLimit()) {
            return true; // if there's no train time limit then there's always remaining train time
        }
        long prediction = predictNextTrainTimeNanos();
        return getRemainingTrainTimeNanos() > prediction;
    }

    default long predictNextTrainTimeNanos() {
        return 0;
    }

    default boolean isBuilt() {
        throw new UnsupportedOperationException();
    }

    default boolean hasRemainingTraining() {
        return !isBuilt() && hasRemainingTrainTime();
    }

    String TRAIN_TIME_LIMIT_NANOS_FLAG = "trtl";

    @Override default ParamSet getParams() {
        return ParamHandler.super.getParams().add(TRAIN_TIME_LIMIT_NANOS_FLAG, getTrainTimeLimitNanos());
    }

    @Override default void setParams(ParamSet param) {
        ParamHandler.setParam(param, TRAIN_TIME_LIMIT_NANOS_FLAG, this::setTrainTimeLimitNanos, Long.class);
    }

}
