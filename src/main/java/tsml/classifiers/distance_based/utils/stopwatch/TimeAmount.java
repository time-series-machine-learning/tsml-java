package tsml.classifiers.distance_based.utils.stopwatch;

import java.util.concurrent.TimeUnit;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public class TimeAmount implements Comparable<TimeAmount> {

    private long amount;
    private TimeUnit unit;

    public TimeAmount() {
        this(0, TimeUnit.NANOSECONDS);
    }

    public TimeAmount(long amount, TimeUnit unit) {
        setAmount(amount);
        setUnit(unit);
    }

    public static TimeAmount parse(String amount, String unit) {
        unit = unit.toUpperCase();
        return new TimeAmount(Long.parseLong(amount), TimeUnit.valueOf(unit));
    }

    @Override
    public String toString() {
        return getAmount() + " " + getUnit();
    }

    public long getAmount() {
        return amount;
    }

    public TimeAmount setAmount(final long amount) {
        this.amount = amount;
        return this;
    }

    public TimeUnit getUnit() {
        return unit;
    }

    public TimeAmount setUnit(final TimeUnit unit) {
        this.unit = unit;
        return this;
    }

    public TimeAmount convert(TimeUnit unit) {
        return new TimeAmount(unit.convert(getAmount(), getUnit()), unit);
    }

    @Override
    public int compareTo(final TimeAmount other) {
        TimeAmount otherNanos = other.convert(TimeUnit.NANOSECONDS);
        TimeAmount nanos = convert(TimeUnit.NANOSECONDS);
        return (int) (nanos.getAmount() - otherNanos.getAmount());
    }
}
