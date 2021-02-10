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

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import weka.core.Debug;

import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

import static tsml.classifiers.distance_based.utils.collections.CollectionUtils.newArrayList;

public class RandomUtilsTest {

    private List<Integer> list;
    private Random random;

    @Before
    public void before() {
        random = new Random(0);
        list = newArrayList(1,2,3,4,5,6,7,8,9,10);
    }

    @Test
    public void testRandomChoiceSingle() {
        int choice = RandomUtils.choice(list, random);
        Assert.assertEquals(1, choice);
        choice = RandomUtils.choice(list, random);
        Assert.assertEquals(9, choice);
        choice = RandomUtils.choice(list, random);
        Assert.assertEquals(10, choice);
        choice = RandomUtils.choice(list, random);
        Assert.assertEquals(8, choice);
        choice = RandomUtils.choice(list, random);
        Assert.assertEquals(6, choice);
    }

    @Test
    public void testRandomChoiceMultiple() {
        final List<Integer> choice = RandomUtils.choice(list, random, 5);
        Assert.assertEquals(5, choice.size());
        Assert.assertEquals(new Integer(1), choice.get(0));
        Assert.assertEquals(new Integer(9), choice.get(1));
        Assert.assertEquals(new Integer(3), choice.get(2));
        Assert.assertEquals(new Integer(5), choice.get(3));
        Assert.assertEquals(new Integer(10), choice.get(4));
    }

    @Test
    public void testRandomChoiceWithReplacement() {
        final List<Integer> choice = RandomUtils.choice(list, random, list.size() * 10, true);
        final Set<Integer> set = new HashSet<>(choice);
        Assert.assertEquals(list.size(), set.size());
    }

    @Test
    public void testRandomChoiceWithoutReplacement() {
        final List<Integer> choice = RandomUtils.choice(list, random, 10, false);
        final Set<Integer> set = new HashSet<>(choice);
        Assert.assertEquals(choice.size(), set.size());
    }

    @Test(expected = AssertionError.class)
    public void testRandomChoiceTooMany() {
        RandomUtils.choice(list, random, list.size() + 1);
    }
}
