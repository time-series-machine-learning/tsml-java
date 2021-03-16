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
 
package tsml.classifiers.distance_based.utils.collections.iteration;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import java.util.*;

public class BaseRandomIteratorTest {
    private RandomIterator<String> iterator;
    private List<String> elements;

    @Before
    public void before() {
        elements = new ArrayList<>(Arrays.asList("a", "b", "c", "d", "e", "f", "g"));
        iterator = new BaseRandomIterator<>();
        iterator.setRandom(new Random(0));
        iterator.buildIterator(elements);
    }

    @Test
    public void testOrderWithoutReplacement() {
        iterator.setWithReplacement(false);
        StringBuilder builder = new StringBuilder();
        Set<String> set = new HashSet<>();
        while(iterator.hasNext()) {
            final String next = iterator.next();
            final boolean added = set.add(next);
            Assert.assertTrue(added);
            builder.append(next);
        }
//        System.out.println(builder.toString());
        Assert.assertEquals("fegcdab", builder.toString());
    }

    @Test
    public void testOrderWithReplacement() {
        iterator.setWithReplacement(true);
        int i = 0;
        StringBuilder builder = new StringBuilder();
        Set<String> set = new HashSet<>();
        boolean dupe = false;
        while(iterator.hasNext() && i++ < 10) {
            final String next = iterator.next();
            dupe = dupe || set.add(next);
            builder.append(next);
        }
        Assert.assertTrue(dupe);
        //        System.out.println(builder.toString());
        Assert.assertEquals("fceceacbgc", builder.toString());
    }

}
