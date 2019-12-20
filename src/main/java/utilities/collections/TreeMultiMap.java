package utilities.collections;

import utilities.ArrayUtilities;

import java.io.Serializable;
import java.util.*;

public class TreeMultiMap<B, A> implements Serializable { // todo implement map interface

    // todo super / extends generics

    public static <A> TreeMultiMap<Double, A> newAscDouble() {
        return new TreeMultiMap<>(Double::compare);
    }

    public static <A> TreeMultiMap<Integer, A> newAscInt() {
        return new TreeMultiMap<>(Integer::compare);
    }

    public static <A> TreeMultiMap<Double, A> newDescDouble() {
        return new TreeMultiMap<>(((Comparator<Double>) Double::compare).reversed());
    }

    public static <A> TreeMultiMap<Integer, A> newDescInt() {
        return new TreeMultiMap<>(((Comparator<Integer>) Integer::compare).reversed());
    }

    public static <A extends Comparable<? super A>, B> TreeMultiMap<A, B> newNaturalAsc() {
        return new TreeMultiMap<A, B>(Comparator.naturalOrder());
    }

    public static <A extends Comparable<? super A>, B> TreeMultiMap<A, B> newNaturalDesc() {
        return new TreeMultiMap<A, B>(Comparator.reverseOrder());
    }

    private final TreeMap<B, List<A>> map;


    public TreeMultiMap(Comparator<? super B> comparator) {
        map = new TreeMap<>(comparator);
    }

    public void put(final B key, final A value) {
        List<A> list = map.computeIfAbsent(key, k -> new ArrayList<>());
        list.add(value);
    }

    public int size() {
        int size = values().size();
        assert size >= 0; // todo replace with debug
        return size;
    }

    protected TreeMap<B, List<A>> decorated() {
        return map;
    }

    public Map.Entry<B, List<A>> lastEntry() {
        return map.lastEntry();
    }

    public Map.Entry<B, List<A>> pollLastEntry() {
        return map.pollLastEntry();
    }

    public Collection<List<A>> values() {
        return map.values();
    }

    public Set<Map.Entry<B, List<A>>> entrySet() {
        return map.entrySet();
    }

    public boolean isEmpty() {
        return size() == 0;
    }

    public Map.Entry<B, List<A>> firstEntry() {
        return map.firstEntry();
    }

    public Map.Entry<B, List<A>> pollFirstEntry() {
        return map.pollFirstEntry();
    }

    public List<A> get(B key) {
        return map.get(key);
    }
}
