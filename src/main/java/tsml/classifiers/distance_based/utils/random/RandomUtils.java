package tsml.classifiers.distance_based.utils.random;

import java.util.*;

import org.junit.Assert;
import tsml.classifiers.distance_based.utils.collections.CollectionUtils;
import tsml.classifiers.distance_based.utils.iteration.RandomIterator;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public class RandomUtils {

    private RandomUtils() {}

    public static List<Integer> choiceIndex(int size, Random random, int numChoices,
                                                              boolean withReplacement) {
        Assert.assertTrue(size > 0);
        Assert.assertNotNull(random);
        Assert.assertTrue(numChoices > 0);
        if(size == 1) {
            return new ArrayList<>(Collections.singletonList(0));
        }
        final List<Integer> indices = CollectionUtils.sequence(size);
        final RandomIterator<Integer> iterator = new RandomIterator<>(random, indices);
        final List<Integer> choices = new ArrayList<>();
        for(int i = 0; i < size; i++) {
            Assert.assertTrue(iterator.hasNext());
            choices.add(iterator.next());
        }
        return choices;
    }

    public static Integer choiceIndex(int size, Random random) {
        return choiceIndex(size, random, 1, false).get(0);
    }

    public static List<Integer> choiceIndex(int size, Random random, int numChoices) {
        return choiceIndex(size, random, numChoices, false);
    }

    public static <A> List<A> choice(Collection<A> collection, Random random, int numChoices, boolean withReplacement) {
        final List<Integer> indices = choiceIndex(collection.size(), random, numChoices, withReplacement);
        final List<A> chosen = new ArrayList<>();
        for(Integer index : indices) {
            A element = null;
            if(collection instanceof List) {
                element = ((List<A>) collection).get(index);
            } else {
                final Iterator<A> iterator = collection.iterator();
                for(int i = 0; i <= index; i++) {
                    Assert.assertTrue(iterator.hasNext());
                    element = iterator.next();
                }
            }
            chosen.add(element);
        }
        return chosen;
    }

    public static <A> List<A> choice(Collection<A> collection, Random random, int numChoices) {
        return choice(collection, random, numChoices, false);
    }

    public static <A> A choice(Collection<A> collection, Random random) {
        return choice(collection, random, 1, false).get(0);
    }

}
