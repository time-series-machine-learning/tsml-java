package utilities.collections;

import com.google.common.collect.*;
import utilities.Utilities;
import utilities.serialisation.SerComparator;
import utilities.serialisation.SerSupplier;
import weka.core.Randomizable;

import java.io.*;
import java.util.*;
import java.util.function.Supplier;

public class PrunedMultimap<K, V> extends DecoratedMultimap<K, V> implements Randomizable, Serializable {

    private int softLimit = -1;
    private int hardLimit = -1;
    private int seed = 0;
    private Random rand = new Random(seed);
    private final TreeMap<K, Collection<V>> backingMap;

    public static <K extends Comparable<? super K>, V> PrunedMultimap<K, V> asc(Supplier<? extends Collection<V>> supplier) {
        return new PrunedMultimap<K, V>(Comparator.naturalOrder(), supplier);
    }

    public static <K extends Comparable<? super K>, V> PrunedMultimap<K, V> desc(Supplier<? extends Collection<V>> supplier) {
        return new PrunedMultimap<K, V>(Comparator.reverseOrder(), supplier);
    }

    public K lastKey() {
        return backingMap.lastKey();
    }

    public PrunedMultimap(Comparator<? super K> comparator, Supplier<? extends Collection<V>> supplier) {
        backingMap = new TreeMap<>((Serializable & Comparator<K>) comparator::compare);
        setMap(Multimaps.newMultimap(backingMap,
                                     (Serializable & com.google.common.base.Supplier<? extends Collection<V>>) supplier::get));
    }

    public PrunedMultimap(Comparator<? super K> comparator) {
        this(comparator, (Serializable & com.google.common.base.Supplier<? extends Collection<V>>) ArrayList::new);
    }

    public boolean hasSoftLimit() {
        return getSoftLimit() > 0;
    }

    public boolean hasHardLimit() {
        return getHardLimit() > 0;
    }

    private void softPrune(int limit) {
        if(limit < 0) throw new IllegalStateException();
        int diff = size() - limit;
        if(diff > 0) {
            Map.Entry<K, Collection<V>> entry = backingMap.lastEntry();
            Collection<V> values = entry.getValue();
            while(diff > 0 && values != null && values.size() <= diff) {
                Collection<V> copy = new ArrayList<>(entry.getValue());
                for(V v : copy) {
                    boolean removed = remove(entry.getKey(), v);
                    if(!removed) {
                        throw new IllegalStateException("this shouldn't happen");
                    }
                }
                diff -= copy.size();
                entry = backingMap.lastEntry();
                if(entry != null) {
                    values = entry.getValue();
                } else {
                    break;
                }
            }
        }
    }

    private void hardPrune(int limit) {
        if(limit < 0) throw new IllegalStateException();
        softPrune(limit);
        int diff = size() - limit;
        if(diff > 0) {
            Map.Entry<K, Collection<V>> entry = backingMap.lastEntry();
            Collection<V> values = entry.getValue();
            List<V> toRemove = Utilities.randPickN(values, diff, rand);
            K k = entry.getKey();
            for(V v : toRemove) {
                remove(k, v);
            }
        }
    }

    protected void prune() {
        if(!isEmpty()) {
            if(hasSoftLimit()) {
                softPrune(getSoftLimit());
            } else if(hasHardLimit()) {
                hardPrune(getHardLimit());
            }
        }
    }

    @Override public boolean put(final K k, final V v) {
        boolean result = super.put(k, v);
        if(result) prune();
        return result;
    }

    @Override public boolean putAll(final Multimap<? extends K, ? extends V> multimap) {
        boolean result = super.putAll(multimap);
        if(result) {
            prune();
        }
        return result;
    }

    @Override public boolean putAll(final K k, final Iterable<? extends V> iterable) {
        boolean result = super.putAll(k, iterable);
        if(result) {
            prune();
        }
        return result;
    }

    @Override public void setSeed(final int seed) {
        this.seed = seed;
        rand.setSeed(seed);
    }

    @Override public int getSeed() {
        return seed;
    }

    public int getSoftLimit() {
        return softLimit;
    }

    public void setSoftLimit(final int softLimit) {
        this.softLimit = softLimit;
        prune();
    }

    public int getHardLimit() {
        return hardLimit;
    }

    public void setHardLimit(final int hardLimit) {
        this.hardLimit = hardLimit;
        prune();
    }

    public Random getRand() {
        return rand;
    }

    public void setRand(final Random rand) {
        this.rand = rand;
    }

    public void disableHardLimit() {
        setHardLimit(-1);
    }

    public void disableLimit() {
        setSoftLimit(-1);
    }

    public static void main(String[] args) {
        PrunedMultimap<Integer, String> map = new PrunedMultimap<>(Integer::compare, HashSet::new);
        map.setHardLimit(1);
//        map.setSoftLimit(1);
        map.put(3, "a");
        map.put(3, "b");
        map.put(3, "c");
        map.put(4, "d");
        map.put(2, "e");
        map.put(2, "f");
    }

    public void hardPruneToSoftLimit() {
        if(hasSoftLimit()) {
            int origHardLimit = getHardLimit();
            setHardLimit(getSoftLimit());
            setHardLimit(origHardLimit);
        }
    }

}
