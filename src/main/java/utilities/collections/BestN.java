package utilities.collections;

import weka.core.Randomizable;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

public class BestN<A extends Comparable<? super A>> implements Randomizable {

    private final PrunedMultiset<A> set;

    public void put(A item) {
        set.add(item);
    }


    public int getHardLimit() {
        return set.getHardLimit();
    }

    public int getSoftLimit() {
        return set.getSoftLimit();
    }

    public void setSoftLimit(int limit) {
        set.setSoftLimit(limit);
    }

    public void setHardLimit(int limit) {
        set.setHardLimit(limit);
    }

    public boolean hasHardLimit() {
        return set.hasHardLimit();
    }

    public boolean hasSoftLimit() {
        return set.hasSoftLimit();
    }

    public void hardPruneToSoftLimit() {
        set.hardPruneToSoftLimit();
    }
    
    public List<A> toList() {
        return set.toList();
    }

    public BestN() {
        this(1);
    }

    public BestN(int n) {
        set = PrunedMultiset.desc(ArrayList::new);
        set.setSoftLimit(n);
    }

    public BestN(Comparator<A> comparator, int n) {
        set = new PrunedMultiset<A>(comparator, ArrayList::new);
        set.setSoftLimit(n);
    }

    public BestN(Comparator<A> comparator) {
        this(comparator, 1);
    }

    public BestN(A item) {
        this();
        put(item);
    }

    @Override public void setSeed(final int seed) {
        set.setSeed(seed);
    }

    @Override public int getSeed() {
        return set.getSeed();
    }
}
