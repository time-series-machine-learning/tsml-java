package tsml.classifiers.distance_based.utils.collections.params.distribution;

import org.junit.Assert;
import tsml.classifiers.distance_based.utils.collections.intervals.Interval;

import java.util.Objects;

public abstract class ClampedDistribution<A> extends BaseDistribution<A> implements Interval<A> {
    
    private Interval<A> interval;
    
    public ClampedDistribution(Interval<A> interval) {
        setInterval(interval);
    }

    public void setInterval(final Interval<A> interval) {
        this.interval = Objects.requireNonNull(interval);
    }

    public Interval<A> getInterval() {
        return interval;
    }

    @Override public A getStart() {
        return interval.getStart();
    }

    @Override public A getEnd() {
        return interval.getEnd();
    }

    @Override public void setStart(final A start) {
        interval.setStart(start);
    }

    @Override public void setEnd(final A end) {
        interval.setEnd(end);
    }

    @Override public A size() {
        return interval.size();
    }

    @Override public boolean contains(final A item) {
        return interval.contains(item);
    }

    @Override public String toString() {
        return super.toString() + "(" + interval.getStart() + ", " + interval.getEnd() + ")";
    }

    @Override public boolean equals(final Object o) {
        if(!(o instanceof ClampedDistribution)) {
            return false;
        }
        ClampedDistribution<?> other = (ClampedDistribution<?>) o;
        return other.getClass().equals(getClass()) && other.interval.equals(interval);
    }
}
