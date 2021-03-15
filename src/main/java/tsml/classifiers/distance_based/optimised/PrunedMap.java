package tsml.classifiers.distance_based.optimised;

import tsml.classifiers.distance_based.utils.collections.CollectionUtils;

import java.io.Serializable;
import java.util.*;

public class PrunedMap<A, B> implements Map<A, List<B>>, 
                                                NavigableMap<A, List<B>>, 
                                                SortedMap<A, List<B>>,
                                                                    Serializable {

    public static void main(String[] args) {
        class A {}
        final TreeMap<A, Integer> map = new TreeMap<>();
    }
    
    private int limit;
    private int size;
    private final TreeMap<A, List<B>> map;
    
    public static <A extends Comparable<? super A>, B> PrunedMap<A, B> asc() {
        return new PrunedMap<>(Comparator.naturalOrder());
    }
    
    public static <A extends Comparable<? super A>, B> PrunedMap<A, B> desc() {
        return new PrunedMap<>(Comparator.reverseOrder());
    }

    public static <A extends Comparable<? super A>, B> PrunedMap<A, B> asc(int limit) {
        final PrunedMap<A, B> map = asc();
        map.setLimit(limit);
        return map;
    }

    public static <A extends Comparable<? super A>, B> PrunedMap<A, B> desc(int limit) {
        final PrunedMap<A, B> map = desc();
        map.setLimit(limit);
        return map;
    }
    
    public static <A extends Comparable<? super A>, B> PrunedMap<A, B> newPrunedMap(int limit, boolean asc) {
        if(asc) {
            return asc(limit);
        } else {
            return desc(limit);
        }
    }
    
    public static <A extends Comparable<? super A>, B> PrunedMap<A, B> newPrunedMap(boolean asc) {
        if(asc) {
            return asc();
        } else {
            return desc();
        }
    }
    
    public PrunedMap(Comparator<? super A> comparator) {
        this(1, comparator);
    }

    public PrunedMap(int limit, Comparator<? super A> comparator) {
        map = new TreeMap<>(comparator);
        setLimit(limit);
        clear();
    }
    
    public int getLimit() {
        return limit;
    }

    public boolean setLimit(final int limit) {
        this.limit = limit;
        return prune();
    }

    @Override public void clear() {
        map.clear();
        size = 0;
    }

    @Override public Entry<A, List<B>> firstEntry() {
        return map.firstEntry();
    }

    @Override public Entry<A, List<B>> lastEntry() {
        return map.lastEntry();
    }

    @Override public Entry<A, List<B>> pollFirstEntry() {
        final Entry<A, List<B>> entry = map.pollFirstEntry();
        size -= entry.getValue().size();
        return entry;
    }

    @Override public Entry<A, List<B>> pollLastEntry() {
        final Entry<A, List<B>> entry = map.pollLastEntry();
        size -= entry.getValue().size();
        return entry;
    }

    @Override public Entry<A, List<B>> lowerEntry(final A a) {
        return map.lowerEntry(a);
    }

    @Override public A lowerKey(final A a) {
        return map.lowerKey(a);
    }

    @Override public Entry<A, List<B>> floorEntry(final A a) {
        return map.floorEntry(a);
    }

    @Override public A floorKey(final A a) {
        return map.floorKey(a);
    }

    @Override public Entry<A, List<B>> ceilingEntry(final A a) {
        return map.ceilingEntry(a);
    }

    @Override public A ceilingKey(final A a) {
        return map.ceilingKey(a);
    }

    @Override public Entry<A, List<B>> higherEntry(final A a) {
        return map.higherEntry(a);
    }

    @Override public A higherKey(final A a) {
        return map.higherKey(a);
    }

    @Override public Set<A> keySet() {
        return map.keySet();
    }

    @Override public NavigableSet<A> navigableKeySet() {
        return map.navigableKeySet();
    }

    @Override public NavigableSet<A> descendingKeySet() {
        return map.descendingKeySet();
    }

    @Override public Collection<List<B>> values() {
        return map.values();
    }

    @Override public Set<Entry<A, List<B>>> entrySet() {
        return map.entrySet();
    }

    @Override public NavigableMap<A, List<B>> descendingMap() {
        return map.descendingMap();
    }

    @Override public NavigableMap<A, List<B>> subMap(final A a, final boolean b, final A k1, final boolean b1) {
        return map.subMap(a, b, k1, b1);
    }

    @Override public NavigableMap<A, List<B>> headMap(final A a, final boolean b) {
        return map.headMap(a, b);
    }

    @Override public NavigableMap<A, List<B>> tailMap(final A a, final boolean b) {
        return map.tailMap(a, b);
    }

    @Override public SortedMap<A, List<B>> subMap(final A a, final A k1) {
        return map.subMap(a, k1);
    }

    @Override public SortedMap<A, List<B>> headMap(final A a) {
        return map.headMap(a);
    }

    @Override public SortedMap<A, List<B>> tailMap(final A a) {
        return map.tailMap(a);
    }

    @Override public List<B> remove(final Object o) {
        final List<B> removed = map.remove(o);
        if(removed != null) {
            size -= removed.size();
        } 
        return removed;
    }

    @Override public boolean remove(final Object o, final Object o1) {
        final List<B> bs = map.get(o);
        boolean removed = false;
        if(bs != null) {
            if(bs.remove(o1)) {
                size--;
                removed = true;
            }
            if(bs.isEmpty()) {
                map.remove(o);
            }
        }
        return removed;
    }
    
    public boolean putAll(A key, List<B> value) {
        return addAll(key, value);
    }

    public boolean add(A key, B value) {
        map.computeIfAbsent(key, k -> new ArrayList<>()).add(value);
        size++;
        prune();
        return map.comparator().compare(key, lastKey()) <= 0;
    }
    
    public boolean addAll(A key, List<B> values) {
        for(B value : values) {
            map.computeIfAbsent(key, k -> new ArrayList<>()).add(value);
            size++;
        }
        prune();
        return map.comparator().compare(key, lastKey()) <= 0;
    }
    
    private boolean prune() {
        boolean pruned = false;
        if(size > limit) {
            Map.Entry<A, List<B>> lastEntry = lastEntry();
            while(size() - lastEntry.getValue().size() >= limit) {
                pollLastEntry();
                lastEntry = lastEntry();
                pruned = true;
            }
        }
        return pruned;
    }

    @Override public int size() {
        return size;
    }

    @Override public boolean isEmpty() {
        return map.isEmpty();
    }

    @Override public boolean equals(final Object o) {
        if(o instanceof PrunedMap) {
            return map.equals(o) && ((PrunedMap<?, ?>) o).limit == limit;
        }
        return false;
    }

    @Override public int hashCode() {
        return map.hashCode();
    }

    @Override public String toString() {
        return map.toString();
    }

    @Override public boolean containsKey(final Object o) {
        return map.containsKey(o);
    }

    @Override public boolean containsValue(final Object o) {
        for(Map.Entry<A, List<B>> entry : entrySet()) {
            if(entry.getValue().equals(o)) {
                return true;
            }
            for(B value : entry.getValue()) {
                if(value.equals(o)) {
                    return true;
                }
            }
        }
        return false;
    }

    @Override public List<B> get(final Object o) {
        final List<B> list = map.get(o);
        if(list == null) {
            return null;
        }
        return Collections.unmodifiableList(list);
    }

    @Override public List<B> put(final A a, final List<B> bs) {
        addAll(a, bs);
        return get(a);
    }

    @Override public Comparator<? super A> comparator() {
        return map.comparator();
    }

    @Override public A firstKey() {
        return map.firstKey();
    }

    @Override public A lastKey() {
        return map.lastKey();
    }

    @Override public void putAll(final Map<? extends A, ? extends List<B>> map) {
        for(Map.Entry<? extends A, ? extends List<B>> entry : map.entrySet()) {
            put(entry.getKey(), entry.getValue());
        }
    }

    public List<B> valuesList() {
        return CollectionUtils.concat(values());
    }
}
