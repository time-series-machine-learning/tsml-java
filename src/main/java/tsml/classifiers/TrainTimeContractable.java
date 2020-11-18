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

import java.util.concurrent.TimeUnit;

/**
 * Interface that allows the user to impose a train time contract of a classifier that
    implements this interface

known classifiers: ShapeletTransformClassifier, RISE, HIVE_COTE (partial),
* BOSS, TSF , ContractRotationForest
 *
 * ********************************NOTES********************************
 * 1) contract time of <=0 means no contract has been set, even if this is potentially contractable
 *
 */
public interface TrainTimeContractable {

    /**
     * This is the single method that must be implemented to store the contract time
      * @param time in nano seconds
     */
    void setTrainTimeLimit(long time);

    /**
     * Are we still within contract? Remove default when fully implemented
     * @param start classifier build start time
     * @return true if classifier is within the train time contract, false otherwise
     */
    default boolean withinTrainContract(long start){ return false;}

    default void setOneDayLimit(){ setTrainTimeLimit(TimeUnit.DAYS, 1); }
    
    default void setOneHourLimit(){ setTrainTimeLimit(TimeUnit.HOURS, 1); }

    default void setOneMinuteLimit(){ setTrainTimeLimit(TimeUnit.MINUTES, 1); }
    
    default void setDayLimit(int t){ setTrainTimeLimit(TimeUnit.DAYS, t); }

    default void setHourLimit(int t){ setTrainTimeLimit(TimeUnit.HOURS, t); }
    
    default void setMinuteLimit(int t){ setTrainTimeLimit(TimeUnit.MINUTES, t); }

    //pass in an value from the TimeUnit enum and the amount of said values.
    default void setTrainTimeLimit(TimeUnit time, long amount) {
        setTrainTimeLimit(TimeUnit.NANOSECONDS.convert(amount, time));
    }

    default void setTrainTimeLimit(long amount, TimeUnit time) {
        setTrainTimeLimit(time, amount);
    }
    
    default long getTrainContractTimeNanos() {
        throw new UnsupportedOperationException();
    }
}
