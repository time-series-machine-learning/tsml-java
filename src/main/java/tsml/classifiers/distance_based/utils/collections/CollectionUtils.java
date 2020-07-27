package tsml.classifiers.distance_based.utils.collections;

import java.util.*;
import java.util.function.BiConsumer;
import java.util.function.BiFunction;
import java.util.function.Consumer;
import org.junit.Assert;

public class CollectionUtils {
    private CollectionUtils() {}

    public static <A> ArrayList<A> newArrayList(A... elements) {
        final ArrayList<A> list = new ArrayList<>(elements.length);
        list.addAll(Arrays.asList(elements));
        return list;
    }

    public static <B> void forEachGroup(int groupSize, List<B> list, Consumer<List<B>> consumer) {
        List<B> group;
        int i = 0;
        int limit = groupSize;
        while(i < list.size() && i < limit) {
            group = new ArrayList<>();
            for(; i < list.size() && i < limit; i++) {
                group.add(list.get(i));
            }
            consumer.accept(group);
            limit += groupSize;
        }
    }

    public static <B> void forEachPair(List<B> pairs, BiConsumer<B, B> consumer) {
        forEachGroup(2, pairs, pair -> {
            if(pair.size() != 2) {
                throw new IllegalStateException("expected pair");
            }
            consumer.accept(pair.get(0), pair.get(1));
        });
    }

    public static <B, A> List<A> convertPairs(List<B> pairs, BiFunction<B, B, A> func) {
        List<A> objs = new ArrayList<>();
        forEachPair(pairs, (a, b) -> {
            final A obj = func.apply(a, b);
            objs.add(obj);
        });
        return objs;
    }

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
