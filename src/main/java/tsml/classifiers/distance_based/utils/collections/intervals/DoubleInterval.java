package tsml.classifiers.distance_based.utils.collections.intervals;

public class DoubleInterval extends BaseInterval<Double> {
    
    public DoubleInterval() {
        this(0, 1);
    }
    
    public DoubleInterval(double start, double end) {
        super(start, end);
    }

    @Override public boolean contains(final Double item) {
        return item <= getEnd() && item >= getStart();
    }

    @Override public Double size() {
        return getEnd() - getStart();
    }
}
