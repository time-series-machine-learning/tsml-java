package tsml.classifiers.distance_based.utils.collections.iteration;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import java.util.*;

public class RandomIteratorTest {
    private RandomIterator<String> iterator;
    private List<String> elements;

    @Before
    public void before() {
        elements = new ArrayList<>(Arrays.asList("a", "b", "c", "d", "e", "f", "g"));
        iterator = new RandomIterator<>(new Random(0), elements);
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
