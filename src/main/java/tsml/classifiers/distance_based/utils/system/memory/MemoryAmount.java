package tsml.classifiers.distance_based.utils.system.memory;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public class MemoryAmount implements Comparable<MemoryAmount> {

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
        unit = unit.toUpperCase();
        return new MemoryAmount(Long.parseLong(amount), MemoryUnit.valueOf(unit));
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
        return (int) (nanos.getAmount() - otherNanos.getAmount());
    }
}
