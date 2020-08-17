package tsml.classifiers.distance_based.utils.collections.params.iteration;

import static org.hamcrest.CoreMatchers.is;
import static org.hamcrest.CoreMatchers.not;
import static org.junit.Assert.assertThat;

import java.util.ArrayList;
import java.util.List;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public class PermutationUtils {

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
