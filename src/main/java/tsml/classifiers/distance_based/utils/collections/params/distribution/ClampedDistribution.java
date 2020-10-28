package tsml.classifiers.distance_based.utils.collections.params.distribution;

import org.junit.Assert;

public abstract class ClampedDistribution<A> extends BaseDistribution<A> {
    
    private A start;
    private A end;
    
    public ClampedDistribution(A end) {
        setEnd(end);
    }
    
    public ClampedDistribution(A start, A end) {
        setStart(start);
        setEnd(end);
    }

    public A getEnd() {
        return end;
    }

    public void setEnd(final A end) {
        Assert.assertNotNull(end);
        this.end = end;
    }

    public A getStart() {
        return start;
    }

    public void setStart(final A start) {
        Assert.assertNotNull(start);
        this.start = start;
    }

    @Override public String toString() {
        return getClass().getSimpleName() + "{" +
                       "start=" + start +
                       ", end=" + end +
                       '}';
    }

    @Override public boolean equals(final Object o) {
        if(!(o instanceof ClampedDistribution)) {
            return false;
        }
        ClampedDistribution<?> other = (ClampedDistribution<?>) o;
        return other.getClass().equals(getClass()) && other.getStart().equals(getStart()) && other.getEnd().equals(getEnd());
    }
}
