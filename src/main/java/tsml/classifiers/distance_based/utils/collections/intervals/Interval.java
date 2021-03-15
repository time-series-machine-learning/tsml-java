package tsml.classifiers.distance_based.utils.collections.intervals;

import tsml.classifiers.distance_based.utils.collections.params.ParamHandler;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;

public interface Interval<A> extends ParamHandler {
    A getStart();

    A getEnd();

    void setStart(A start);
    
    void setEnd(A end);

    A size();
    
    boolean contains(A item);

    @Override String toString();
}
