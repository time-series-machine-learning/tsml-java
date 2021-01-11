package tsml.classifiers.distance_based.utils.collections.maps;

import java.util.Collection;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Take two maps. Any modifications to the first map are reflected in the second, assuming the second is the inverse mapping of the first.
 */
public class BiMapper<A, B> implements Map<A, B> {

    public BiMapper(final Map<A, B> map, final Map<B, A> inverseMap) {
        this.map = Objects.requireNonNull(map);
        this.inverseMap = Objects.requireNonNull(inverseMap);
    }

    private final Map<A, B> map;
    private final Map<B, A> inverseMap;
    
    public int size() {
        return map.size();
    }

    public boolean isEmpty() {
        return map.isEmpty();
    }

    public boolean containsKey(final Object o) {
        return map.containsKey(o);
    }

    public boolean containsValue(final Object o) {
        return map.containsValue(o);
    }

    public B get(final Object o) {
        return map.get(o);
    }

    public B put(final A a, final B b) {
        inverseMap.put(b, a);
        return map.put(a, b);
    }

    public B remove(final Object o) {
        final B removed = map.remove(o);
        if(removed != null) {
            inverseMap.remove(removed);
        }
        return removed;
    }

    @Override public void putAll(final Map<? extends A, ? extends B> other) {
        other.forEach(this::put);
    }

    public void clear() {
        map.clear();
        inverseMap.clear();
    }

    public Set<A> keySet() {
        return map.keySet();
    }

    public Collection<B> values() {
        return map.values();
    }

    public Set<Entry<A, B>> entrySet() {
        return map.entrySet();
    }

    @Override public boolean equals(final Object o) {
        return map.equals(o);
    }

    @Override public int hashCode() {
        return map.hashCode();
    }

    @Override public String toString() {
        return map.toString();
    }
}
