package tsml.classifiers.distance_based.utils.collections.pruned;

import com.google.common.base.Supplier;
import com.google.common.collect.*;
import java.util.Map.Entry;
import java.util.function.BiConsumer;
import org.junit.Assert;
import tsml.classifiers.distance_based.utils.system.random.RandomUtils;
import utilities.Utilities;

import java.io.*;
import java.util.*;

public class PrunedMultimap<K, V> implements Serializable, ListMultimap<K, V> {

    private int softLimit = -1;
    private int hardLimit = -1;
    private Random random = null;
    private final TreeMap<K, Collection<V>> backingMap;
    private final ListMultimap<K, V> listMultimap;
    private DiscardType discardType = DiscardType.NEWEST;

    public DiscardType getDiscardType() {
        return discardType;
    }

    public PrunedMultimap<K, V> setDiscardType(
        final DiscardType discardType) {
        this.discardType = discardType;
        return this;
    }

    public enum DiscardType {
        OLDEST,
        NEWEST,
        RANDOM,
    }

    public static <K extends Comparable<? super K>, V> PrunedMultimap<K, V> ascSoftSingle() {
        PrunedMultimap<K, V> map = asc(ArrayList::new);
        map.setSoftLimit(1);
        return map;
    }

    public static <K extends Comparable<? super K>, V> PrunedMultimap<K, V> descSoftSingle() {
        PrunedMultimap<K, V> map = desc(ArrayList::new);
        map.setSoftLimit(1);
        return map;
    }

    public static <K extends Comparable<? super K>, V> PrunedMultimap<K, V> asc() {
        return asc(ArrayList::new);
    }

    public static <K extends Comparable<? super K>, V> PrunedMultimap<K, V> asc(Supplier<? extends List<V>> supplier) {
        return new PrunedMultimap<K, V>(Comparator.naturalOrder(), supplier);
    }

    public static <K extends Comparable<? super K>, V> PrunedMultimap<K, V> desc() {
        return desc(ArrayList::new);
    }

    public static <K extends Comparable<? super K>, V> PrunedMultimap<K, V> desc(Supplier<? extends List<V>> supplier) {
        return new PrunedMultimap<K, V>(Comparator.reverseOrder(), supplier);
    }

    public K lastKey() {
        return backingMap.lastKey();
    }

    public K firstKey() {
        return backingMap.firstKey();
    }

    public PrunedMultimap(Comparator<? super K> comparator, Supplier<? extends List<V>> supplier) {
        backingMap = new TreeMap<>((Serializable & Comparator<K>) comparator::compare);
        listMultimap = Multimaps.newListMultimap(backingMap,
                                     (Serializable & Supplier<? extends List<V>>) supplier::get);
    }

    public PrunedMultimap(Comparator<? super K> comparator) {
        this(comparator, (Serializable & Supplier<? extends List<V>>) ArrayList::new);
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
            K lastKey = backingMap.lastKey();
            List<V> values = get(lastKey);
            while(diff > 0 && values != null && values.size() <= diff) {
                diff -= values.size();
                listMultimap.removeAll(lastKey);
                lastKey = backingMap.lastKey();
                values = get(lastKey);
            }
        }
    }

    private void hardPrune(int limit) {
        if(limit < 0) throw new IllegalStateException();
        softPrune(limit);
        int diff = size() - limit;
        if(diff > 0) {
            K lastKey = backingMap.lastKey();
            List<V> values = get(lastKey);
            switch(discardType) {
                case OLDEST:
                    for(int i = 0; i < diff; i++) {
                        values.remove(0);
                    }
                    break;
                case NEWEST:
                    for(int i = 0; i < diff; i++) {
                        values.remove(values.size() - 1);
                    }
                    break;
                case RANDOM:
                    // need random to be set in order to random pick
                    Assert.assertNotNull(random);
                    List<V> toRemove = RandomUtils.choice(values, getRandom(), diff);
                    for(V v : toRemove) {
                        remove(lastKey, v);
                    }
                    break;
                default:
                    throw new UnsupportedOperationException();
            }
        }
    }

    public void prune() {
        if(!isEmpty()) {
            if(hasSoftLimit()) {
                softPrune(softLimit);
            }
            if(hasHardLimit()) {
                hardPrune(hardLimit);
            }
        }
    }

    @Override public boolean put(final K k, final V v) {
        boolean result = listMultimap.put(k, v);
        if(result) prune();
        return result;
    }

    @Override public boolean putAll(final Multimap<? extends K, ? extends V> multimap) {
        boolean result = listMultimap.putAll(multimap);
        if(result) {
            prune();
        }
        return result;
    }

    @Override public boolean putAll(final K k, final Iterable<? extends V> iterable) {
        boolean result = listMultimap.putAll(k, iterable);
        if(result) {
            prune();
        }
        return result;
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

    public Random getRandom() {
        return random;
    }

    public void setRandom(final Random random) {
        this.random = random;
    }

    public void disableHardLimit() {
        setHardLimit(-1);
    }

    public void disableSoftLimit() {
        setSoftLimit(-1);
    }

    public void disableLimits() {
        disableHardLimit();
        disableSoftLimit();
    }

    @Override
    public List<V> get(final K k) {
        return listMultimap.get(k);
    }

    @Override
    public List<V> removeAll(final Object o) {
        return listMultimap.removeAll(o);
    }

    @Override
    public List<V> replaceValues(final K k, final Iterable<? extends V> iterable) {
        return listMultimap.replaceValues(k, iterable);
    }

    @Override
    public Map<K, Collection<V>> asMap() {
        return listMultimap.asMap();
    }

    @Override
    public boolean equals(final Object o) {
        return listMultimap.equals(o);
    }

    @Override
    public int size() {
        return listMultimap.size();
    }

    @Override
    public boolean isEmpty() {
        return listMultimap.isEmpty();
    }

    @Override
    public boolean containsKey(final Object o) {
        return listMultimap.containsKey(o);
    }

    @Override
    public boolean containsValue(final Object o) {
        return listMultimap.containsValue(o);
    }

    @Override
    public boolean containsEntry(final Object o,
        final Object o1) {
        return listMultimap.containsEntry(o, o1);
    }

    @Override
    public boolean remove(final Object o,
        final Object o1) {
        return listMultimap.remove(o, o1);
    }

    @Override
    public void clear() {
        listMultimap.clear();
    }

    @Override
    public Set<K> keySet() {
        return listMultimap.keySet();
    }

    @Override
    public Multiset<K> keys() {
        return listMultimap.keys();
    }

    @Override
    public Collection<V> values() {
        return listMultimap.values();
    }

    @Override
    public Collection<Entry<K, V>> entries() {
        return listMultimap.entries();
    }

    @Override
    public void forEach(final BiConsumer<? super K, ? super V> action) {
        listMultimap.forEach(action);
    }

    @Override public String toString() {
        return "PrunedMultimap{" +
               "softLimit=" + softLimit +
               ", hardLimit=" + hardLimit +
               ", size=" + listMultimap.size() +
               ", discardType=" + discardType +
               ", listMultimap=" + listMultimap +
               '}';
    }
}
