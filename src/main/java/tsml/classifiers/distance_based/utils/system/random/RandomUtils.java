package tsml.classifiers.distance_based.utils.system.random;

import java.util.*;

import org.junit.Assert;
import tsml.classifiers.distance_based.utils.collections.CollectionUtils;
import tsml.classifiers.distance_based.utils.collections.iteration.RandomIterator;
import utilities.ArrayUtilities;
import utilities.Utilities;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public class RandomUtils {
    
    private static boolean skipSingleElementPick = false;

    private RandomUtils() {}

    public static void setSkipSingleElementPick(final boolean skipSingleElementPick) {
        RandomUtils.skipSingleElementPick = skipSingleElementPick;
    }

    public static boolean isSkipSingleElementPick() {
        return skipSingleElementPick;
    }

    public static List<Integer> choiceIndex(int size, Random random, int numChoices,
                                                              boolean withReplacement) {
        Assert.assertTrue(size > 0);
        if(skipSingleElementPick && size == 1) {
            return new ArrayList<>(Collections.singletonList(0));
        }
        Assert.assertNotNull(random);
        Assert.assertTrue(numChoices > 0);
        final List<Integer> indices = ArrayUtilities.sequence(size);
        final RandomIterator<Integer> iterator = new RandomIterator<>(random, indices, withReplacement);
        final List<Integer> choices = new ArrayList<>(numChoices);
        for(int i = 0; i < numChoices; i++) {
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
        final List<A> chosen = new ArrayList<>(numChoices);
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

    public static <A> List<A> choice(RandomIterator<A> iterator, int numChoices) {
        List<A> choices = new ArrayList<>();
        for(int i = 0; i < numChoices; i++) {
            Assert.assertTrue(iterator.hasNext());
            final A choice = iterator.next();
            choices.add(choice);
        }
        return choices;
    }

    public static <A> A choice(RandomIterator<A> iterator) {
        return choice(iterator, 1).get(0);
    }

}
