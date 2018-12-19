/*
 Interface that allows the user to impose a time contract of a classifier that 
implements this interface

known classifiers: ShapeletTransformClassifier, RISE (not tested) HiveCote (partial)
 */
package timeseriesweka.classifiers;

/**
 *
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
