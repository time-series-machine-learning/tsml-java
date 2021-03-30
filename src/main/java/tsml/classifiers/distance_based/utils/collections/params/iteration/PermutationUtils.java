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
 
package tsml.classifiers.distance_based.utils.collections.params.iteration;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Objects;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public class PermutationUtils {

    /**
     * Find the index of the collection which contains the given index. I.e. given several lists,
     *      [ [0,1,2,3], [4], [5,6], [7,8,9] ] 
     *      these give the sizes
     *      sizes: [ 4, 1, 2, 3 ]
     *      view those lists as one contiguous list and find the index of the list which the specified index falls into.
     *      cumulative sizes: [ 4, 5, 7, 10 ] 
     *      For this example, say the specified index is 6. This would fall into the 3rd list, so we'd return 2.
     *      If the specified index were 2, we'd return 0 for the first list.
     *      If the specified index were 8, we'd return 3 for the last list.
     *      Etc...
     * 
     * @param collections
     * @param index
     * @return
     */
    public static int spannedIndexOf(Collection<Integer> collections, int index) {
        if(index < 0) {
            throw new IllegalArgumentException("index negative");
        }
        Objects.requireNonNull(collections);
        if(collections.isEmpty()) {
            throw new IllegalArgumentException("empty set of sizes");
        }
        int collectionIndex = 0;
        for(Integer size : collections) {
            index -= size;
            if(index < 0) {
                return collectionIndex;
            }
            collectionIndex++;
        }
        throw new IndexOutOfBoundsException();
    }
    
    public static List<Integer> fromPermutation(int permutation, List<Integer> binSizes) {
        int maxCombination = numPermutations(binSizes) - 1;
        if(permutation > maxCombination || binSizes.size() == 0 || permutation < 0) {
            throw new IndexOutOfBoundsException();
        }
        List<Integer> result = new ArrayList<>();
        for(int binSize : binSizes) {
            if(binSize > 1) {
                result.add(permutation % binSize);
                permutation /= binSize;
            } else if(binSize == 1) {
                result.add(0);
            } else {
                // binSize is 0 or less (i.e. no index as that bin cannot be indexed as size <=0)
                result.add(-1);
            }
        }
        return result;
    }


    public static int toPermutation(List<Integer> values, List<Integer> binSizes) {
        if(values.size() != binSizes.size()) {
            throw new IllegalArgumentException("incorrect number of args");
        }
        int permutation = 0;
        for(int i = binSizes.size() - 1; i >= 0; i--) {
            int binSize = binSizes.get(i);
            if(binSize > 1) {
                int value = values.get(i);
                permutation *= binSize;
                permutation += value;
            }
        }
        return permutation;
    }

    public static int numPermutations(List<Integer> binSizes) {
        List<Integer> maxValues = new ArrayList<>();
        boolean positive = false;
        for(Integer binSize : binSizes) {
            int size = binSize - 1;
            if(size < 0) {
                size = 0;
            } else {
                positive = true;
            }
            maxValues.add(size);
        }
        int result = toPermutation(maxValues, binSizes);
        if(positive && !maxValues.isEmpty()) {
            result++;
        }
        return result;
    }
}
