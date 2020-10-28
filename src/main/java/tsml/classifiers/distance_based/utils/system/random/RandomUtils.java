package tsml.classifiers.distance_based.utils.system.random;

import java.util.*;
import java.util.stream.Collectors;

import org.junit.Assert;
import tsml.classifiers.distance_based.utils.collections.iteration.RandomIterator;
import utilities.ArrayUtilities;

import static tsml.classifiers.distance_based.utils.collections.CollectionUtils.newArrayList;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public class RandomUtils {

    public static <A> ArrayList<A> choice(RandomIterator<A> iterator, int numChoices) {
        
    }
    
    /**
     * choice several indices from a set range.
     * @param size the max size (i.e. max index is size-1, min index is 0)
     * @param random the random source
     * @param numChoices the number of choices to make
     * @param withReplacement whether to allow indices to be picked more than once
     * @param skipSingle whether to short circuit choices from lists of size 1
     * @return
     */
    public static ArrayList<Integer> choiceIndex(int size, Random random, int numChoices, boolean withReplacement, boolean skipSingle) {
        Assert.assertTrue(size > 0);
        if(skipSingle && size == 1) {
            return newArrayList(0);
        }
        Assert.assertTrue(numChoices > 0);
        if(numChoices == 1) {
            return newArrayList(random.nextInt(size));
        }
        final List<Integer> indices = ArrayUtilities.sequence(size);
        final RandomIterator<Integer> iterator = new RandomIterator<>(random, indices, withReplacement);
        final ArrayList<Integer> choices = new ArrayList<>(size);
        for(int i = 0; i < numChoices; i++) {
            choices.add(iterator.next());
        }
        return choices;
    }
    
    // choice elements by index
    
    public static Integer choiceIndex(int size, Random random) {
        return choiceIndex(size, random, 1, false, true).get(0);
    }
    
    public static Integer choiceIndexWithNoSkip(int size, Random random) {
        return choiceIndex(size, random, 1, false, false).get(0);
    }
    
    public static ArrayList<Integer> choiceIndexWithReplacement(int size, Random random, int numChoices) {
        return choiceIndex(size, random, numChoices, true, true);
    }

    public static ArrayList<Integer> choiceIndexWithReplacementWithNoSkip(int size, Random random, int numChoices) {
        return choiceIndex(size, random, numChoices, true, false);
    }
    
    public static ArrayList<Integer> choiceIndex(int size, Random random, int numChoices) {
        return choiceIndex(size, random, numChoices, false, true);
    }

    public static ArrayList<Integer> choiceIndexWithNoSkip(int size, Random random, int numChoices) {
        return choiceIndex(size, random, numChoices, false, false);
    }
    
    // choose elements directly
    
    public static <A> ArrayList<A> choice(List<A> list, Random random, int numChoices, boolean withReplacement, boolean skipIfSingleElement) {
        return choiceIndex(list.size(), random, numChoices, withReplacement, skipIfSingleElement).stream().map(
                        list::get).collect(Collectors.toCollection(ArrayList::new));
    }
    
    public static <A> A choice(List<A> list, Random random) {
        return choice(list, random, 1, false, true).get(0);
    }

    public static <A> A choiceWithNoSkip(List<A> list, Random random) {
        return choice(list, random, 1, false, false).get(0);
    }
    
    public static <A> ArrayList<A> choiceWithReplacement(List<A> list, Random random, int numChoices) {
        return choice(list, random, numChoices, true, true);
    }
    
    public static <A> ArrayList<A> choiceWithReplacementWithNoSkip(List<A> list, Random random, int numChoices) {
        return choice(list, random, numChoices, true, false);
    }
    
    public static <A> ArrayList<A> choice(List<A> list, Random random, int numChoices) {
        return choice(list, random, numChoices, false, true);
    }
    
    public static <A> ArrayList<A> choiceWithNoSkip(List<A> list, Random random, int numChoices) {
        return choice(list, random, numChoices, false, false);
    }

}
