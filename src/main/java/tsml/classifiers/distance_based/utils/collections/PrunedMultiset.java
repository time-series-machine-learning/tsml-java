package tsml.classifiers.distance_based.utils.collections;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.List;
import java.util.function.Supplier;
import weka.core.Randomizable;

public class PrunedMultiset<K> implements DefaultMultiset<K>, Serializable, Randomizable {

    private final PrunedMultimap<K, K> map;


    public PrunedMultiset(Comparator<K> comparator) {
        map = new PrunedMultimap<>(comparator);
    }

    public PrunedMultiset(Comparator<K> comparator, Supplier<? extends Collection<K>> supplier) {
        map = new PrunedMultimap<>(comparator, supplier);
    }

    public static <K extends Comparable<? super K>> PrunedMultiset<K> asc(Supplier<? extends Collection<K>> supplier) {
        return new PrunedMultiset<K>(Comparator.naturalOrder(), supplier);
    }

    public static <K extends Comparable<? super K>> PrunedMultiset<K> desc(Supplier<? extends Collection<K>> supplier) {
        return new PrunedMultiset<K>(Comparator.reverseOrder(), supplier);
    }

    public int getHardLimit() {
        return map.getHardLimit();
    }

    public void setHardLimit(int limit) {
        map.setHardLimit(limit);
    }

    public int getSoftLimit() {
        return map.getSoftLimit();
    }

    public void setSoftLimit(int limit) {
        map.setSoftLimit(limit);
    }

    public boolean hasHardLimit() {
        return map.hasHardLimit();
    }

    public boolean hasSoftLimit() {
        return map.hasSoftLimit();
    }

    public void hardPruneToSoftLimit() {
        map.hardPruneToSoftLimit();
    }

    public List<K> toList() {
        return new ArrayList<>(map.keys());
    }

    @Override
    public int size() {
        return map.size();
    }

    @Override
    public boolean isEmpty() {
        return map.isEmpty();
    }

    @Override
    public boolean add(final K k) {
        return map.put(k, k);
    }

    @Override
    public void clear() {

    }

    @Override
    public int getSeed() {
        return map.getSeed();
    }

    @Override
    public void setSeed(final int seed) {
        map.setSeed(seed);
    }
}
