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
     * @param skipSingleOption whether to short circuit choices from lists of size 1
     * @param allowReordering whether to allow the indices to be reordered during selection without replacement. This makes choosing from a large list faster, but does not maintain the ordering of list for efficiency purposes. Reordering whilst choosing is only concerned with the order of the indices left to choose from and does not reorder the list itself.
     * @return
     */
    public static ArrayList<Integer> choiceIndex(int size, Random random, int numChoices, boolean withReplacement, boolean skipSingleOption, boolean allowReordering) {
        Assert.assertTrue(size > 0);
        if(skipSingleOption && size == 1) {
            return newArrayList(0);
        }
        Assert.assertTrue(numChoices > 0);
        if(numChoices == 1) {
            return newArrayList(random.nextInt(size));
        }
        if(!withReplacement && numChoices > size) {
            throw new IllegalArgumentException("cannot choose " + numChoices + " from 0.." + size + " without replacement");
        }
        final List<Integer> indices = ArrayUtilities.sequence(size);
        final BaseRandomIterator<Integer>
                iterator = new BaseRandomIterator<>();
        iterator.setWithReplacement(withReplacement);
        iterator.setSkipSingleOption(skipSingleOption);
        iterator.setAllowReordering(allowReordering);
        iterator.setRandom(random);
        iterator.buildIterator(indices);
        return choice(iterator, numChoices);
    }

    // choice elements by index

    public static Integer choiceIndex(int size, Random random) {
        return choiceIndex(size, random, 1, false, true, true).get(0);
    }

    public static Integer choiceIndexWithNoSkip(int size, Random random) {
        return choiceIndex(size, random, 1, false, false, true).get(0);
    }

    public static ArrayList<Integer> choiceIndexWithReplacement(int size, Random random, int numChoices) {
        return choiceIndex(size, random, numChoices, true, true, true);
    }

    public static ArrayList<Integer> choiceIndexWithReplacementWithNoSkip(int size, Random random, int numChoices) {
        return choiceIndex(size, random, numChoices, true, false, true);
    }

    public static ArrayList<Integer> choiceIndex(int size, Random random, int numChoices) {
        return choiceIndex(size, random, numChoices, false, true, true);
    }

    public static ArrayList<Integer> choiceIndexWithNoSkip(int size, Random random, int numChoices) {
        return choiceIndex(size, random, numChoices, false, false, true);
    }

    // choose elements directly

    public static <A> ArrayList<A> choice(List<A> list, Random random, int numChoices, boolean withReplacement, boolean skipIfSingleElement, boolean allowReordering) {
        return Utilities.apply( choiceIndex(list.size(), random, numChoices, withReplacement, skipIfSingleElement, allowReordering), list::get);
    }

    public static <A> A choice(List<A> list, Random random) {
        return choice(list, random, 1, false, false, true).get(0);
    }

    public static <A> A choiceWithNoSkip(List<A> list, Random random) {
        return choice(list, random, 1, false, false, true).get(0);
    }

    public static <A> ArrayList<A> choiceWithReplacement(List<A> list, Random random, int numChoices) {
        return choice(list, random, numChoices, true, true, true);
    }

    public static <A> ArrayList<A> choiceWithReplacementWithNoSkip(List<A> list, Random random, int numChoices) {
        return choice(list, random, numChoices, true, false, true);
    }

    public static <A> ArrayList<A> choice(List<A> list, Random random, int numChoices) {
        return choice(list, random, numChoices, false, true, true);
    }

    public static <A> ArrayList<A> choiceWithNoSkip(List<A> list, Random random, int numChoices) {
        return choice(list, random, numChoices, false, false, true);
    }

    // pick elements from list as well as choosing (i.e. make choice of elements and remove from the source list)

    /**
     * 
     * @param list
     * @param random
     * @param numChoices
     * @param withReplacement
     * @param skipIfSingleElement
     * @param allowReordering conversely to choice, this will reorder the list when picking >1 item and this parameter is true.
     * @param <A>
     * @return
     */
    public static <A> ArrayList<A> pick(List<A> list, Random random, int numChoices, boolean withReplacement, boolean skipIfSingleElement, boolean allowReordering) {
        final ArrayList<Integer> indices = choiceIndex(list.size(), random, numChoices, withReplacement, skipIfSingleElement, allowReordering);
        final ArrayList<A> chosen = Utilities.apply(indices, list::get);
        CollectionUtils.removeAll(list, indices, allowReordering);
        return chosen;
    }
    
    public static <A> A pick(List<A> list, Random random) {
        return pick(list, random, 1, false, true, true).get(0);
    }

    public static <A> A pickWithNoSkip(List<A> list, Random random) {
        return pick(list, random, 1, false, false, true).get(0);
    }

    public static <A> ArrayList<A> pick(List<A> list, Random random, int numChoices) {
        return pick(list, random, numChoices, false, true, true);
    }

    public static <A> ArrayList<A> pickWithNoSkip(List<A> list, Random random, int numChoices) {
        return pick(list, random, numChoices, false, false, true);
    }

    public static <A> ArrayList<A> pickWithReplacement(List<A> list, Random random, int numChoices) {
        return pick(list, random, numChoices, true, true, true);
    }

    public static <A> ArrayList<A> pickWithReplacementWithNoSkip(List<A> list, Random random, int numChoices) {
        return pick(list, random, numChoices, true, false, true);
    }
}
