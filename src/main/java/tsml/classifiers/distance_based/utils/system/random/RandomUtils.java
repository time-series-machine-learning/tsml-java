/*
 * This file is part of the UEA Time Series Machine Learning (TSML) toolbox.
 *
 * The UEA TSML toolbox is free software: you can redistribute it and/or 
 * modify it under the terms of the GNU General Public License as published 
 * by the Free Software Foundation, either version 3 of the License, or 
 * (at your option) any later version.
 *
 * The UEA TSML toolbox is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with the UEA TSML toolbox. If not, see <https://www.gnu.org/licenses/>.
 */
 
package tsml.classifiers.distance_based.utils.system.random;

import java.util.*;
import java.util.stream.Collectors;

import tsml.classifiers.distance_based.utils.collections.iteration.BaseRandomIterator;
import tsml.classifiers.distance_based.utils.collections.iteration.RandomIterator;
import tsml.classifiers.distance_based.utils.collections.lists.IndexList;
import utilities.Utilities;

import static tsml.classifiers.distance_based.utils.collections.CollectionUtils.newArrayList;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public class RandomUtils {

    public static <A> ArrayList<A> choice(RandomIterator<A> iterator, int numChoices) {
        final ArrayList<A> choices = new ArrayList<>(numChoices);
        for(int i = 0; i < numChoices; i++) {
            if(!iterator.hasNext()) throw new IllegalStateException("iterator has no items remaining at iteration step " + i);
            choices.add(iterator.next());
        }
        return choices;
    }

    public static int choiceIndex(int size, Random random) {
        if(size == 1) {
            // only 1 element, no need to randomly choose
            return 0;
        } else {
            // multiple elements, randomly choose
            return random.nextInt(size);
        }
    }
    
    /**
     * choice several indices from a set range.
     * @param size the max size (i.e. max index is size-1, min index is 0)
     * @param random the random source
     * @param numChoices the number of choices to make
     * @param withReplacement whether to allow indices to be picked more than once
     * @return
     */
    public static List<Integer> choiceIndex(int size, Random random, int numChoices, boolean withReplacement) {
        if(numChoices == 1) {
            // single choice
            return newArrayList(choiceIndex(size, random));
        }
        if(numChoices > size && !withReplacement) {
            // too many choices given size
            throw new IllegalArgumentException("cannot choose " + numChoices + " from 0.." + size + " without replacement");
        }
        final List<Integer> indices = new IndexList(size);
        final RandomIterator<Integer>
                iterator = new BaseRandomIterator<>();
        iterator.setWithReplacement(withReplacement);
        iterator.setRandom(random);
        iterator.buildIterator(indices);
        return choice(iterator, numChoices);
    }

    // choice elements by index

    /**
     * Avoids a span of numbers when choosing an index
     * @param size
     * @param random
     * @return
     */
    public static Integer choiceIndexExcept(int size, Random random, Collection<Integer> exceptions) {
        int index = choiceIndex(size - exceptions.size(), random);
        // if the chosen index lies within the exception zone, then the index needs to be shifted by the zone length to avoid these indices
        exceptions = exceptions.stream().distinct().sorted().collect(Collectors.toList());
        for(Integer exception : exceptions) {
            if(index >= exception) {
                index++;
            } else {
                break;
            }
        }
        return index;
    }
    
    public static Integer choiceIndexExcept(int size, Random random, int exception) {
        return choiceIndexExcept(size, random, Collections.singletonList(exception));        
    }

    public static List<Integer> choiceIndexWithReplacement(int size, Random random, int numChoices) {
        return choiceIndex(size, random, numChoices, true);
    }

    public static List<Integer> choiceIndex(int size, Random random, int numChoices) {
        return choiceIndex(size, random, numChoices, false);
    }
    
    public static List<Integer> shuffleIndices(int size, Random random) {
        return choiceIndex(size, random, size);
    }

    // choose elements directly

    public static <A> List<A> shuffle(List<A> list, Random random) {
        return Utilities.apply(shuffleIndices(list.size(), random), list::get);
    }
    
    public static <A> List<A> choice(List<A> list, Random random, int numChoices, boolean withReplacement) {
        return Utilities.apply( choiceIndex(list.size(), random, numChoices, withReplacement), list::get);
    }

    public static <A> A choice(List<A> list, Random random) {
        final int i = choiceIndex(list.size(), random);
        return list.get(i);
    }

    public static <A> List<A> choiceWithReplacement(List<A> list, Random random, int numChoices) {
        return choice(list, random, numChoices, true);
    }

    public static <A> List<A> choice(List<A> list, Random random, int numChoices) {
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
    public static <A> List<A> remove(List<A> list, Random random, int numChoices, boolean withReplacement) {
        List<Integer> indices = choiceIndex(list.size(), random, numChoices, withReplacement);
        final ArrayList<A> chosen = Utilities.apply(indices, list::get);
        indices = indices.stream().distinct().sorted(Comparator.reverseOrder()).collect(Collectors.toList());
        for(int index : indices) {
            list.remove(index);
        }
        return chosen;
    }
    
    public static <A> A remove(List<A> list, Random random) {
        final int i = choiceIndex(list.size(), random);
        return list.remove(i);
    }

    public static <A> List<A> remove(List<A> list, Random random, int numChoices) {
        return remove(list, random, numChoices, false);
    }

    public static <A> List<A> pickWithReplacement(List<A> list, Random random, int numChoices) {
        return remove(list, random, numChoices, true);
    }

}
