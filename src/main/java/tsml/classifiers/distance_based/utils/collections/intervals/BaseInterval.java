package tsml.classifiers.distance_based.utils.collections.intervals;

import tsml.classifiers.distance_based.utils.collections.params.ParamSet;

import java.util.Objects;

public abstract class BaseInterval<A> implements Interval<A> {
    
    public static String START_FLAG = "s";
    public static String END_FLAG = "l";
    
    public BaseInterval(A start, A end) {
        setStart(start);
        setEnd(end);
    }
    
    private A start;
    private A end;

    @Override public A getStart() {
        return start;
    }

    @Override public void setStart(A start) {
        this.start = Objects.requireNonNull(start);
    }

    @Override public A getEnd() {
        return end;
    }

    @Override public void setEnd(final A end) {
        this.end = Objects.requireNonNull(end);
    }

    @Override public ParamSet getParams() {
        return Interval.super.getParams().add(START_FLAG, getStart()).add(END_FLAG, getEnd());
    }

    @Override public void setParams(final ParamSet paramSet) throws Exception {
        Interval.super.setParams(paramSet);
        setStart(paramSet.get(START_FLAG, getStart()));
        setEnd(paramSet.get(END_FLAG, getEnd()));
    }
    
    @Override public String toString() {
        return getStart() + " - " + getEnd();
    }

    @Override public boolean equals(final Object o) {
        if(this == o) {
            return true;
        }
        if(!(o instanceof BaseInterval)) {
            return false;
        }
        final BaseInterval<?> that = (BaseInterval<?>) o;
        return Objects.equals(start, that.start) && Objects.equals(end, that.end);
    }

    @Override public int hashCode() {
        return Objects.hash(start, end);
    }
}
