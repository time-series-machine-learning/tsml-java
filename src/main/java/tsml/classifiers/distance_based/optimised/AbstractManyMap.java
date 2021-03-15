package tsml.classifiers.distance_based.optimised;

import tsml.classifiers.distance_based.utils.system.serial.SerSupplier;

import java.util.*;

public class AbstractManyMap<A extends Comparable<A>, B, C extends Collection<B>> implements ManyMap<A, B, C> {
    
    public AbstractManyMap(final SerSupplier<C> supplier) {
        this(Comparator.naturalOrder(), supplier);
    }

    public AbstractManyMap(final Comparator<? super A> comparator, final SerSupplier<C> supplier) {
        map = new TreeMap<>(Objects.requireNonNull(comparator));
        this.supplier = Objects.requireNonNull(supplier);
        clear();
    }
    
    private final TreeMap<A, C> map;
    private final SerSupplier<C> supplier;
    private int size;

    @Override public void add(final A key, final B value) {
        map.computeIfAbsent(key, k -> supplier.get()).add(value);
        size++;
    }
    
    @Override public void addAll(final A key, final Iterable<B> values) {
        for(B value : values) {
            add(key, value);
        }
    }

    @Override public int size() {
        return size;
    }

    @Override public boolean isEmpty() {
        return size == 0;
    }

    @Override public boolean containsKey(final Object o) {
        return map.containsKey(o);
    }

    @Override public boolean containsValue(final Object o) {
        for(Map.Entry<A, C> entry : entrySet()) {
            if(entry.getValue().contains(o)) {
                return true;
            }
        }
        return false;
    }

    @Override public C get(final Object o) {
        return map.get(o);
    }

    @Override public C put(final A a, final C bs) {
        map.computeIfAbsent(a, k -> supplier.get()).addAll(bs);
        size += bs.size();
        return null;
    }

    @Override public C remove(final Object o) {
        final C removed = map.remove(o);
        if(removed != null) {
            size -= removed.size();
        }
        return removed;
    }

    @Override public boolean remove(final Object o, final Object o1) {
        final C bs = get(o);
        if(bs != null) {
            return bs.remove(o1);
        }
        return false;
    }

    @Override public void putAll(final Map<? extends A, ? extends C> map) {
        for(Map.Entry<? extends A, ? extends C> entry : map.entrySet()) {
            put(entry.getKey(), entry.getValue());
        }
    }

    @Override public void clear() {
        size = 0;
        map.clear();
    }

    @Override public Set<A> keySet() {
        return map.keySet();
    }

    @Override public Collection<C> values() {
        return map.values();
    }

    @Override public Set<Entry<A, C>> entrySet() {
        return map.entrySet();
    }
}
