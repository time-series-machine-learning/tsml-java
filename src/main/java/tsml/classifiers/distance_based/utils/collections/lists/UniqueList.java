package tsml.classifiers.distance_based.utils.collections.lists;

import java.util.*;

public class UniqueList<A> extends AbstractList<A> {
    
    private Set<A> set;
    private List<A> list;

    public UniqueList() {
        clear();
    }

    @Override public int size() {
        return list.size();
    }

    @Override public A get(final int i) {
        return list.get(i);
    }

    @Override public A set(final int i, final A item) {
        if(!set.add(item)) {
            // already contains item so don't bother adding it, just return the current value in it's place
            return list.get(i);
        } else {
            // does not contain item so set in the list
            return list.set(i, item);
        }
    }

    @Override public void add(final int i, final A item) {
        if(set.add(item)) {
            // item was not in the set so add to the list
            list.add(i, item);
        } else {
            // cannot add the item as already in set
        }
    }

    @Override public A remove(final int i) {
        final A removed = list.remove(i);
        set.remove(removed);
        return removed;
    }

    @Override public boolean remove(final Object o) {
        if(set.remove(o)) {
            return list.remove(o);
        } else {
            return false;
        }
    }
    
    

    @Override public void clear() {
        set = new HashSet<>();
        list = new ArrayList<>();
    }
}
