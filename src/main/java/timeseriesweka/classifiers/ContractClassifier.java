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
package timeseriesweka.classifiers;

/**
 * Interface that allows the user to impose a time contract of a classifier that 
    implements this interface

    known classifiers: ShapeletTransformClassifier, RISE (not tested) HiveCote (partial)
 * @author raj09hxu
 */
public interface ContractClassifier {
    public static double CHECKPOINTINTERVAL=2.0;    //Minimum interval between checkpoointing
    public enum TimeLimit {MINUTE, HOUR, DAY};

    public default void setOneDayLimit(){
        setTimeLimit(TimeLimit.DAY, 1);
    }
    
    public default void setOneHourLimit(){
        setTimeLimit(TimeLimit.HOUR, 1);
    }

    public default void setOneMinuteLimit(){
        setTimeLimit(TimeLimit.MINUTE, 1);
    }
    
    public default void setDayLimit(int t){
        setTimeLimit(TimeLimit.DAY, t);
    }

    public default void setHourLimit(int t){
        setTimeLimit(TimeLimit.HOUR, t);
    }
    
    public default void setMinuteLimit(int t){
        setTimeLimit(TimeLimit.MINUTE, t);
    }

    //set any value in nanoseconds you like.
    void setTimeLimit(long time);

    //pass in an enum of hour, minut, day, and the amount of them.
    void setTimeLimit(TimeLimit time, int amount);
    
}
