package utilities.collections;

import com.google.common.collect.Multimap;
import com.google.common.collect.Multiset;

import java.io.Serializable;
import java.util.Collection;
import java.util.Map;
import java.util.Set;
import java.util.function.BiConsumer;

public class DecoratedMultimap<K, V> implements Multimap<K, V>, Serializable {

    private Multimap<K, V> map;

    protected DecoratedMultimap() {}

    protected final Multimap<K, V> getMap() {
        return map;
    }

    protected final void setMap(Multimap<K, V> map) {
        this.map = map;
    }

    public DecoratedMultimap(final Multimap<K, V> map) {this.map = map;}


    @Override public int size() {
        return map.size();
    }

    @Override public boolean isEmpty() {
        return map.isEmpty();
    }

    @Override public boolean containsKey(final Object o) {
        return map.containsKey(o);
    }

    @Override public boolean containsValue(final Object o) {
        return map.containsValue(o);
    }

    @Override public boolean containsEntry(final Object o, final Object o1) {
        return map.containsEntry(o, o1);
    }

    @Override public boolean put(final K k, final V v) {
        return map.put(k, v);
    }

    @Override public boolean remove(final Object o, final Object o1) {
        return map.remove(o, o1);
    }

    @Override public boolean putAll(final K k, final Iterable<? extends V> iterable) {
        return map.putAll(k, iterable);
    }

    @Override public boolean putAll(final Multimap<? extends K, ? extends V> multimap) {
        return map.putAll(multimap);
    }

    @Override public Collection<V> replaceValues(final K k, final Iterable<? extends V> iterable) {
        return map.replaceValues(k, iterable);
    }

    @Override public Collection<V> removeAll(final Object o) {
        return map.removeAll(o);
    }

    @Override public void clear() {
        map.clear();
    }

    @Override public Collection<V> get(final K k) {
        return map.get(k);
    }

    @Override public Set<K> keySet() {
        return map.keySet();
    }

    @Override public Multiset<K> keys() {
        return map.keys();
    }

    @Override public Collection<V> values() {
        return map.values();
    }

    @Override public Collection<Map.Entry<K, V>> entries() {
        return map.entries();
    }

    @Override public void forEach(final BiConsumer<? super K, ? super V> action) {
        map.forEach(action);
    }

    @Override public Map<K, Collection<V>> asMap() {
        return map.asMap();
    }
}
