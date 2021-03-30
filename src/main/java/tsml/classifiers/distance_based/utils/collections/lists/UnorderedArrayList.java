package tsml.classifiers.distance_based.utils.collections.lists;

import java.io.Serializable;
import java.util.*;
import java.util.function.Consumer;
import java.util.function.Predicate;
import java.util.function.UnaryOperator;
import java.util.stream.Stream;

/**
 * An ArrayList except operations may reorder the list for efficiency purposes, e.g. the remove(int i) function swaps the element to the end and then removes to avoid shuffling all elements > i down 1 place. 
 * @param <A>
 */
public class UnorderedArrayList<A> extends AbstractList<A> implements Serializable {

    private final ArrayList<A> list;
    
    public UnorderedArrayList() {
        this(new ArrayList<>());
    }
    
    public UnorderedArrayList(int size) {
        this(new ArrayList<>(size));
    }
    
    public UnorderedArrayList(Collection<A> other) {
        list = new ArrayList<>(other);
    }
    
    @Override public A remove(final int i) {
        int endIndex = list.size() - 1;
        // swap the element to be removed to the end
        Collections.swap(list, i, endIndex);
        // remove the end element
        return list.remove(endIndex);
    }

    @Override public boolean remove(final Object o) {
        return list.remove(o);
    }

    @Override public void clear() {
        list.clear();
    }

    @Override public A get(final int i) {
        return list.get(i);
    }

    @Override public void add(final int i, final A a) {
        list.add(a);
        Collections.swap(list, i, list.size() - 1);
    }

    @Override public A set(final int i, final A a) {
        return list.set(i, a);
    }

    @Override public int size() {
        return list.size();
    }
}
