package tsml.classifiers.distance_based.utils.system.timing;

import java.util.concurrent.TimeUnit;
import tsml.classifiers.distance_based.utils.strings.StrUtils;

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

    public enum ShortTimeUnit {
        S(TimeUnit.SECONDS),
        SEC(TimeUnit.SECONDS),
        M(TimeUnit.MINUTES),
        MIN(TimeUnit.MINUTES),
        H(TimeUnit.HOURS),
        HR(TimeUnit.HOURS),
        D(TimeUnit.DAYS),
        ;


        private final TimeUnit alias;

        ShortTimeUnit(final TimeUnit unit) {
            this.alias = unit;
        }

        public TimeUnit getAlias() {
            return alias;
        }
    }

    public static TimeAmount parse(String amount, String unit) {
        unit = unit.trim();
        amount = amount.trim();
        unit = unit.toUpperCase();
        unit = StrUtils.depluralise(unit);
        TimeUnit timeUnit;
        try {
            timeUnit = ShortTimeUnit.valueOf(unit).getAlias();
        } catch(Exception e) {
            timeUnit = TimeUnit.valueOf(unit);
        }
        return new TimeAmount(Long.parseLong(amount), timeUnit);
    }

    public static TimeAmount parse(String str) {
        String[] parts = StrUtils.extractAmountAndUnit(str);
        return parse(parts[0], parts[1]);
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
        return (int) (otherNanos.getAmount() - nanos.getAmount());
    }
}
