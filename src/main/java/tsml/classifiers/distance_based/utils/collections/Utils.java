package tsml.classifiers.distance_based.utils.collections;

import java.util.Collection;
import java.util.Iterator;
import java.util.Set;

public class Utils {
    private Utils() {}

    public static <A> A get(Iterator<A> iterator, int index) {
        if(index < 0) {
            throw new ArrayIndexOutOfBoundsException();
        }
        A result = null;
        for(int i = 0; i < index; i++) {
            if(!iterator.hasNext()) {
                throw new ArrayIndexOutOfBoundsException();
            }
            result = iterator.next();
        }
        return result;
    }


    public static <A> void replace(Set<A> set, A item) {
        set.remove(item);
        set.add(item);
    }

    public static <A> void replace(Set<A> set, Collection<A> collection) {
        for(A item  : collection) {
            replace(set, item);
        }
    }

    public static <A> A get(Iterable<A> iterable, int index) {
        return get(iterable.iterator(), index);
    }

    public static <A> int size(Iterator<A> iterator) {
        int count = 0;
        while (iterator.hasNext()) {
            count++;
            iterator.next();
        }
        return count;
    }

    public static <A> int size(Iterable<A> iterable) {
        return size(iterable.iterator());
    }

    public static <A> void put(A item, Set<A> set) {
        boolean result = set.add(item);
        if(!result) {
            throw new IllegalStateException("already contains item " + item.toString());
        }
    }
}
