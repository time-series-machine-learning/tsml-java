package tsml.classifiers.distance_based.utils.system.memory;

import tsml.classifiers.distance_based.utils.strings.StrUtils;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public class MemoryAmount implements Comparable<MemoryAmount> {

    public enum ShortMemoryUnit {
        B(MemoryUnit.BYTES),
        KB(MemoryUnit.KIBIBYTES),
        K(MemoryUnit.KIBIBYTES),
        MB(MemoryUnit.MEBIBYTES),
        M(MemoryUnit.MEBIBYTES),
        GB(MemoryUnit.GIBIBYTES),
        G(MemoryUnit.GIBIBYTES),
        ;

        private final MemoryUnit alias;

        ShortMemoryUnit(final MemoryUnit alias) {
            this.alias = alias;
        }

        public MemoryUnit getAlias() {
            return alias;
        }
    }

    private long amount;
    private MemoryUnit unit;

    public MemoryAmount() {
        this(0, MemoryUnit.BYTES);
    }

    public MemoryAmount(long amount, MemoryUnit unit) {
        setAmount(amount);
        setUnit(unit);
    }

    public static MemoryAmount parse(String amount, String unit) {
        unit = unit.trim();
        amount = amount.trim();
        unit = unit.toUpperCase();
        unit = StrUtils.depluralise(unit);
        MemoryUnit memoryUnit;
        try {
            memoryUnit = ShortMemoryUnit.valueOf(unit).getAlias();
        } catch(Exception e) {
            memoryUnit = MemoryUnit.valueOf(unit);
        }
        return new MemoryAmount(Long.parseLong(amount), memoryUnit);
    }

    public static MemoryAmount parse(String str) {
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

    public MemoryAmount setAmount(final long amount) {
        this.amount = amount;
        return this;
    }

    public MemoryUnit getUnit() {
        return unit;
    }

    public MemoryAmount setUnit(final MemoryUnit unit) {
        this.unit = unit;
        return this;
    }

    public MemoryAmount convert(MemoryUnit unit) {
        return new MemoryAmount(unit.convert(getAmount(), getUnit()), unit);
    }

    @Override
    public int compareTo(final MemoryAmount other) {
        MemoryAmount otherNanos = other.convert(MemoryUnit.BYTES);
        MemoryAmount nanos = convert(MemoryUnit.BYTES);
        return (int) (otherNanos.getAmount() - nanos.getAmount());
    }
}
