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

import static org.hamcrest.CoreMatchers.is;
import static org.hamcrest.CoreMatchers.not;
import static org.junit.Assert.assertThat;
import static tsml.classifiers.distance_based.utils.collections.params.iteration.PermutationUtils.fromPermutation;
import static tsml.classifiers.distance_based.utils.collections.params.iteration.PermutationUtils.numPermutations;
import static tsml.classifiers.distance_based.utils.collections.params.iteration.PermutationUtils.toPermutation;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
@RunWith(Parameterized.class)
public class PermutationUtilsTest {
    private List<Integer> bins;
    private int index;

    public PermutationUtilsTest(List<Integer> bins, int index) {
        this.bins = bins;
        this.index = index;
    }

    @Parameterized.Parameters
    public static Collection permutations() {
        return Arrays.asList(new Object[][]{
            {Arrays.asList(2, 3, 4), 24},
            {Arrays.asList(0, 0, 0), 0},
            {Arrays.asList(0, 1, 0), 1},
            {Arrays.asList(0, -1, 0), 0},
            {Arrays.asList(2, -1, 5), 10},
            {Arrays.asList(-3392, -348642, -4), 0},
        });
    }

    @Test
    public void testToAndFromUniquePermutations() {
        List<List<Integer>> seenPermutations = new ArrayList<>();
        for(int i = 0; i < index; i++) {
            List<Integer> permutation = fromPermutation(i, bins);
            int index = toPermutation(permutation, bins);
//            System.out.println("permutation for " + i + " is " + permutation + " and reverse is " + index);
            Assert.assertEquals(index, i);
            for(List<Integer> seenPermutation : seenPermutations) {
                assertThat(permutation, is(not(seenPermutation)));
            }
//            System.out.println("permutation " + permutation + " with index " + i + " is unique");
            seenPermutations.add(permutation);
        }
    }

    @Test
    public void testNumPermutations() {
        int index = numPermutations(bins);
//        System.out.println("num perms for " + bins + " is " + index);
        Assert.assertEquals(index, this.index);
    }
}