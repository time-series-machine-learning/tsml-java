package tsml.classifiers.distance_based.utils.system.random;

import java.util.*;
import java.util.stream.Collectors;

import org.junit.Assert;
import tsml.classifiers.distance_based.utils.collections.CollectionUtils;
import tsml.classifiers.distance_based.utils.collections.iteration.BaseRandomIterator;
import tsml.classifiers.distance_based.utils.collections.iteration.RandomIterator;
import utilities.ArrayUtilities;
import utilities.Utilities;

import static tsml.classifiers.distance_based.utils.collections.CollectionUtils.newArrayList;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public class RandomUtils {

    public static <A> ArrayList<A> choice(
            RandomIterator<A> iterator, int numChoices) {
        final ArrayList<A> choices = new ArrayList<>(numChoices);
        for(int i = 0; i < numChoices; i++) {
            if(!iterator.hasNext()) throw new IllegalStateException("iterator has no items remaining at iteration step " + i);
            choices.add(iterator.next());
        }
        return choices;
    }

    /**
     * choice several indices from a set range.
     * @param size the max size (i.e. max index is size-1, min index is 0)
     * @param random the random source
     * @param numChoices the number of choices to make
     * @param withReplacement whether to allow indices to be picked more than once
     * @return
     */
    public static ArrayList<Integer> choiceIndex(int size, Random random, int numChoices, boolean withReplacement) {
        Assert.assertTrue(size > 0);
        Assert.assertTrue(numChoices > 0);
        if(numChoices == 1) {
            final int i = random.nextInt(size);
            return newArrayList(i);
        }
        if(!withReplacement && numChoices > size) {
            throw new IllegalArgumentException("cannot choose " + numChoices + " from 0.." + size + " without replacement");
        }
        final List<Integer> indices = ArrayUtilities.sequence(size);
        final BaseRandomIterator<Integer>
                iterator = new BaseRandomIterator<>();
        iterator.setWithReplacement(withReplacement);
        iterator.setRandom(random);
        iterator.buildIterator(indices);
        return choice(iterator, numChoices);
    }

    // choice elements by index

    public static Integer choiceIndex(int size, Random random) {
        return choiceIndex(size, random, 1, false).get(0);
    }

    public static ArrayList<Integer> choiceIndexWithReplacement(int size, Random random, int numChoices) {
        return choiceIndex(size, random, numChoices, true);
    }

    public static ArrayList<Integer> choiceIndex(int size, Random random, int numChoices) {
        return choiceIndex(size, random, numChoices, false);
    }

    // choose elements directly

    public static <A> ArrayList<A> choice(List<A> list, Random random, int numChoices, boolean withReplacement) {
        return Utilities.apply( choiceIndex(list.size(), random, numChoices, withReplacement), list::get);
    }

    public static <A> A choice(List<A> list, Random random) {
        return choice(list, random, 1, false).get(0);
    }

    public static <A> ArrayList<A> choiceWithReplacement(List<A> list, Random random, int numChoices) {
        return choice(list, random, numChoices, true);
    }

    public static <A> ArrayList<A> choice(List<A> list, Random random, int numChoices) {
        return choice(list, random, numChoices, false);
    }

    // pick elements from list as well as choosing (i.e. make choice of elements and remove from the source list)

    /**
     * 
     * @param list
     * @param random
     * @param numChoices
     * @param withReplacement
     * @param <A>
     * @return
     */
    public static <A> ArrayList<A> pick(List<A> list, Random random, int numChoices, boolean withReplacement) {
        final ArrayList<Integer> indices = choiceIndex(list.size(), random, numChoices, withReplacement);
        final ArrayList<A> chosen = Utilities.apply(indices, list::get);
        CollectionUtils.removeAllUnordered(list, indices);
        return chosen;
    }
    
    public static <A> A pick(List<A> list, Random random) {
        return pick(list, random, 1).get(0);
    }

    public static <A> ArrayList<A> pick(List<A> list, Random random, int numChoices) {
        return pick(list, random, numChoices, false);
    }

    public static <A> ArrayList<A> pickWithReplacement(List<A> list, Random random, int numChoices) {
        return pick(list, random, numChoices, true);
    }

}
