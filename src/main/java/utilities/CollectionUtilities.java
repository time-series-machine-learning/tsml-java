package utilities;

import java.util.*;
import java.util.function.Function;

public class CollectionUtilities {

    // todo make A/B extended
    public static <A extends Comparable<A>> A findBest(Collection<A> collection, Random random) {
        return findBest(collection, Comparable::compareTo, random);
    }

    public static <A extends Comparable<A>> A findBest(Collection<A> collection) {
        return findBest(collection, Comparable::compareTo);
    }

    public static <A> A findBest(Collection<A> collection, Comparator<A> comparator) {
        return findBest(collection, comparator, item -> item);
    }

    public static <A, B> A findBest(Collection<A> collection, Comparator<B> comparator, Function<A, B> getter) {
        return findBest(collection, comparator, null, getter);
    }

    public static <A, B> A findBest(Collection<A> collection, Comparator<B> comparator, Random random, Function<A, B> getter) {
        List<A> bests = findBests(collection, comparator, getter);
        int size = bests.size();
        if(size <= 0) {
            return null;
        } else if(random != null) {
            return bests.get(random.nextInt(size));
        } else {
            return bests.get(0);
        }
    }

    public static <A> A findBest(Collection<A> collection, Comparator<A> comparator, Random random) {
        return findBest(collection, comparator, random, item -> item);
    }

    public static <A> List<A> findBests(Collection<A> collection, Comparator<A> comparator) {
        return findBests(collection, comparator, item -> item);
    }

    public static <A, B> List<A> findBests(Collection<A> collection, Comparator<B> comparator, Function<A, B> getter) {
        Iterator<A> iterator = collection.iterator();
        List<A> bestItems = new ArrayList<>();
        if(iterator.hasNext()) {
            A bestItem = iterator.next();
            bestItems.add(bestItem);
            B bestValue = getter.apply(bestItem);
            while (iterator.hasNext()) {
                A item = iterator.next();
                B value = getter.apply(item);
                int comparison = comparator.compare(value, bestValue);
                if(comparison >= 0) {
                    if(comparison > 0) {
                        bestItems.clear();
                        bestValue = value;
                    }
                    bestItems.add(item);
                }
            }
        }
        return bestItems;
    }
}
