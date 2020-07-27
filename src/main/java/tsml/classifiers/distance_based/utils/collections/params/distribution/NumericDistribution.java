package tsml.classifiers.distance_based.utils.collections.params.distribution;

import java.util.Objects;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public abstract class NumericDistribution<A extends Number> extends Distribution<A> {

    private A min;
    private A max;

    public NumericDistribution(final A min, final A max) {
        setMinAndMax(min, max);
    }

    protected abstract void checkSetMinAndMax(A min, A max);

    public void setMinAndMax(A min, A max) {
        checkSetMinAndMax(min, max);
        this.min = min;
        this.max = max;
    }

    public abstract A sample();

    public A getMax() {
        return max;
    }

    public void setMax(final A max) {
        setMinAndMax(min, max);
    }

    public A getMin() {
        return min;
    }

    public void setMin(final A min) {
        setMinAndMax(min, max);
    }

    @Override
    public String toString() {
        return getClass().getSimpleName() + "{" +
            "min=" + min +
            ", max=" + max +
            '}';
    }

    @Override
    public boolean equals(final Object o) {
        if(this == o) {
            return true;
        }
        if(o == null || getClass() != o.getClass()) {
            return false;
        }
        final NumericDistribution<?> that = (NumericDistribution<?>) o;
        return Objects.equals(getMin(), that.getMin()) &&
            Objects.equals(getMax(), that.getMax());
    }

    @Override
    public int hashCode() {
        return Objects.hash(getMin(), getMax());
    }
}
