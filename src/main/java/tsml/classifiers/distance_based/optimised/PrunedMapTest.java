package tsml.classifiers.distance_based.optimised;

import org.junit.Assert;
import org.junit.Test;
import tsml.classifiers.distance_based.utils.classifiers.CopierUtils;

import java.util.Arrays;

public class PrunedMapTest {
    
    @Test
    public void testPrune() {
        final PrunedMap<Integer, Integer> map = new PrunedMap<>(3);
        // test overflow when equal
        for(int i = 0; i < 5; i++) {
            map.add(3, i);
        }
        Assert.assertEquals("{3=[0, 1, 2, 3, 4]}", map.toString());
        Assert.assertEquals(5, map.size());
        // test underflow
        for(int i = 0; i < 2; i++) {
            map.add(4, i);
        }
        Assert.assertEquals("{4=[0, 1], 3=[0, 1, 2, 3, 4]}", map.toString());
        Assert.assertEquals(7, map.size());
        // test prune
        map.add(5, 0);
        Assert.assertEquals("{5=[0], 4=[0, 1]}", map.toString());
        Assert.assertEquals(3, map.size());
        // test prune on set limit
        map.setLimit(1);
        Assert.assertEquals("{5=[0]}", map.toString());
        Assert.assertEquals(1, map.size());
        // test can't beat best
        for(int i = 0; i < 10; i++) {
            map.add(1, i);
        }
        Assert.assertEquals("{5=[0]}", map.toString());
        Assert.assertEquals(1, map.size());
        map.clear();
        Assert.assertEquals("{}", map.toString());
        Assert.assertEquals(0, map.size());
        map.setLimit(20);
        for(int i = 0; map.size() < map.getLimit(); i++) {
            for(int j = 0; j < 3; j++) {
                map.add(i, j);
            }
        }
        Assert.assertEquals("{6=[0, 1, 2], 5=[0, 1, 2], 4=[0, 1, 2], 3=[0, 1, 2], 2=[0, 1, 2], 1=[0, 1, 2], 0=[0, 1, 2]}", map.toString());
        Assert.assertEquals(21, map.size());
        map.setLimit(10);
        Assert.assertEquals("{6=[0, 1, 2], 5=[0, 1, 2], 4=[0, 1, 2], 3=[0, 1, 2]}", map.toString());
        Assert.assertEquals(12, map.size());
        map.setLimit(1);
        Assert.assertEquals("{6=[0, 1, 2]}", map.toString());
        Assert.assertEquals(3, map.size());
        
        map.clear();
        Assert.assertEquals(0, map.size());
        map.setLimit(5);
        map.put(1, Arrays.asList(1,2,3,4,5));
        map.put(2, Arrays.asList(1,2,3,4,5));
        map.put(3, Arrays.asList(1,2,3,4,5));
        Assert.assertEquals("{3=[1, 2, 3, 4, 5]}", map.toString());
        Assert.assertEquals(5, map.size());
        map.put(4, Arrays.asList(1,2));
        map.put(5, Arrays.asList(3,4));
        Assert.assertEquals("{5=[3, 4], 4=[1, 2], 3=[1, 2, 3, 4, 5]}", map.toString());
        Assert.assertEquals(9, map.size());
        map.pollLastEntry();
        Assert.assertEquals("{5=[3, 4], 4=[1, 2]}", map.toString());
        Assert.assertEquals(4, map.size());
        map.pollFirstEntry();
        Assert.assertEquals("{4=[1, 2]}", map.toString());
        Assert.assertEquals(2, map.size());
        
        map.putAll(CopierUtils.deepCopy(map));
        Assert.assertEquals("{4=[1, 2, 1, 2]}", map.toString());
        Assert.assertEquals(4, map.size());
        
        map.remove(4, 1);
        map.remove(4, 1);
        Assert.assertEquals("{4=[2, 2]}", map.toString());
        Assert.assertEquals(2, map.size());
        map.remove(4);
        Assert.assertEquals("{}", map.toString());
        Assert.assertEquals(0, map.size());
        map.addAll(4, Arrays.asList(4,4,4,4));
        Assert.assertEquals("{4=[4, 4, 4, 4]}", map.toString());
        Assert.assertEquals(4, map.size());
        map.remove(5);
        Assert.assertEquals("{4=[4, 4, 4, 4]}", map.toString());
        Assert.assertEquals(4, map.size());
        map.remove(5,5);
        Assert.assertEquals("{4=[4, 4, 4, 4]}", map.toString());
        Assert.assertEquals(4, map.size());
        map.remove(4);
        Assert.assertEquals("{}", map.toString());
        Assert.assertEquals(0, map.size());
        
        map.setLimit(3);
        map.clear();
        Assert.assertTrue(map.add(3, 3));
        Assert.assertTrue(map.add(4, 4));
        Assert.assertTrue(map.add(5, 5));
        Assert.assertTrue(map.add(5, 5));
        Assert.assertFalse(map.add(2, 2));
        Assert.assertFalse(map.add(1, 1));
        map.setLimit(1);
        Assert.assertTrue(map.add(6, 6));
        Assert.assertTrue(map.add(6, 6));
        Assert.assertFalse(map.add(5, 5));
    }
}
