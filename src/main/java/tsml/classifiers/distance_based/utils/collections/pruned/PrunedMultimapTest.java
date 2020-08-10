package tsml.classifiers.distance_based.utils.collections.pruned;

import java.util.Random;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import tsml.classifiers.distance_based.utils.collections.pruned.PrunedMultimap.DiscardType;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public class PrunedMultimapTest {

    private PrunedMultimap<Integer, String> map;

    @Before
    public void before() {
        map = PrunedMultimap.desc();
    }

    @Test
    public void testSoftLimit() {
        map.setSoftLimit(2);
        map.put(1, "a");
        Assert.assertEquals(1, map.size());
        Assert.assertTrue(map.containsValue("a"));
        map.put(2, "b");
        Assert.assertEquals(2, map.size());
        map.put(2, "c");
        Assert.assertEquals(2, map.size());
        map.put(2, "d");
        Assert.assertEquals(3, map.size());
        map.put(2, "e");
        Assert.assertEquals(4, map.size());
        Assert.assertTrue(map.containsValue("b"));
        Assert.assertTrue(map.containsValue("c"));
        Assert.assertTrue(map.containsValue("d"));
        Assert.assertTrue(map.containsValue("e"));
        Assert.assertFalse(map.containsValue("a"));
        map.put(3, "f");
        Assert.assertEquals(5, map.size());
        Assert.assertTrue(map.containsValue("f"));
        map.put(1, "g");
        Assert.assertEquals(5, map.size());
        Assert.assertFalse(map.containsValue("g"));
        map.put(3, "h");
        Assert.assertEquals(2, map.size());
        map.put(3, "i");
        Assert.assertEquals(3, map.size());
        Assert.assertTrue(map.containsValue("f"));
        Assert.assertTrue(map.containsValue("h"));
        Assert.assertTrue(map.containsValue("i"));
        map.put(4, "j");
        Assert.assertEquals(4, map.size());
        Assert.assertTrue(map.containsValue("j"));
        map.put(4, "k");
        Assert.assertEquals(2, map.size());
        Assert.assertTrue(map.containsValue("k"));
        Assert.assertFalse(map.containsValue("f"));
        Assert.assertFalse(map.containsValue("h"));
        Assert.assertFalse(map.containsValue("i"));
    }

    @Test
    public void testHardLimitDiscardYoungest() {
        map.setHardLimit(2);
        map.setDiscardType(DiscardType.NEWEST);
        map.put(1, "a");
        Assert.assertEquals(1, map.size());
        Assert.assertTrue(map.containsValue("a"));
        map.put(2, "b");
        Assert.assertEquals(2, map.size());
        map.put(2, "c");
        Assert.assertEquals(2, map.size());
        map.put(2, "d");
        Assert.assertEquals(2, map.size());
        map.put(2, "e");
        Assert.assertEquals(2, map.size());
        Assert.assertTrue(map.containsValue("b"));
        Assert.assertTrue(map.containsValue("c"));
        Assert.assertFalse(map.containsValue("d"));
        Assert.assertFalse(map.containsValue("e"));
        Assert.assertFalse(map.containsValue("a"));
        map.put(3, "f");
        Assert.assertEquals(2, map.size());
        Assert.assertTrue(map.containsValue("f"));
        map.put(1, "g");
        Assert.assertEquals(2, map.size());
        Assert.assertFalse(map.containsValue("g"));
        map.put(3, "h");
        Assert.assertEquals(2, map.size());
        map.put(3, "i");
        Assert.assertEquals(2, map.size());
        Assert.assertTrue(map.containsValue("f"));
        Assert.assertTrue(map.containsValue("h"));
        Assert.assertFalse(map.containsValue("i"));
        map.put(4, "j");
        Assert.assertEquals(2, map.size());
        Assert.assertTrue(map.containsValue("j"));
        map.put(4, "k");
        Assert.assertEquals(2, map.size());
        Assert.assertTrue(map.containsValue("k"));
        Assert.assertFalse(map.containsValue("f"));
        Assert.assertFalse(map.containsValue("h"));
        Assert.assertFalse(map.containsValue("i"));
    }

    @Test
    public void testHardLimitDiscardOldest() {
        map.setHardLimit(2);
        map.setDiscardType(DiscardType.OLDEST);
        map.put(1, "a");
        Assert.assertEquals(1, map.size());
        Assert.assertTrue(map.containsValue("a"));
        map.put(2, "b");
        Assert.assertEquals(2, map.size());
        map.put(2, "c");
        Assert.assertEquals(2, map.size());
        map.put(2, "d");
        Assert.assertEquals(2, map.size());
        map.put(2, "e");
        Assert.assertEquals(2, map.size());
        Assert.assertFalse(map.containsValue("b"));
        Assert.assertFalse(map.containsValue("c"));
        Assert.assertTrue(map.containsValue("d"));
        Assert.assertTrue(map.containsValue("e"));
        Assert.assertFalse(map.containsValue("a"));
        map.put(3, "f");
        Assert.assertEquals(2, map.size());
        Assert.assertTrue(map.containsValue("f"));
        map.put(1, "g");
        Assert.assertEquals(2, map.size());
        Assert.assertFalse(map.containsValue("g"));
        map.put(3, "h");
        Assert.assertEquals(2, map.size());
        map.put(3, "i");
        Assert.assertEquals(2, map.size());
        Assert.assertFalse(map.containsValue("f"));
        Assert.assertTrue(map.containsValue("h"));
        Assert.assertTrue(map.containsValue("i"));
        map.put(4, "j");
        Assert.assertEquals(2, map.size());
        Assert.assertTrue(map.containsValue("j"));
        map.put(4, "k");
        Assert.assertEquals(2, map.size());
        Assert.assertTrue(map.containsValue("k"));
        Assert.assertFalse(map.containsValue("f"));
        Assert.assertFalse(map.containsValue("h"));
        Assert.assertFalse(map.containsValue("i"));
    }

    @Test
    public void testHardLimitDiscardRandom() {
        map.setHardLimit(2);
        map.setDiscardType(DiscardType.RANDOM);
        map.setRandom(new Random(0));
        map.put(1, "a");
        Assert.assertEquals(1, map.size());
        Assert.assertTrue(map.containsValue("a"));
        map.put(2, "b");
        Assert.assertEquals(2, map.size());
        map.put(2, "c");
        Assert.assertEquals(2, map.size());
        map.put(2, "d");
        Assert.assertEquals(2, map.size());
        map.put(2, "e");
        Assert.assertEquals(2, map.size());
        Assert.assertFalse(map.containsValue("b"));
        Assert.assertTrue(map.containsValue("c"));
        Assert.assertFalse(map.containsValue("d"));
        Assert.assertTrue(map.containsValue("e"));
        Assert.assertFalse(map.containsValue("a"));
        map.put(3, "f");
        Assert.assertEquals(2, map.size());
        Assert.assertTrue(map.containsValue("f"));
        map.put(1, "g");
        Assert.assertEquals(2, map.size());
        Assert.assertFalse(map.containsValue("g"));
        map.put(3, "h");
        Assert.assertEquals(2, map.size());
        map.put(3, "i");
        Assert.assertEquals(2, map.size());
        Assert.assertTrue(map.containsValue("f"));
        Assert.assertTrue(map.containsValue("h"));
        Assert.assertFalse(map.containsValue("i"));
        map.put(4, "j");
        Assert.assertEquals(2, map.size());
        Assert.assertTrue(map.containsValue("j"));
        map.put(4, "k");
        Assert.assertEquals(2, map.size());
        Assert.assertTrue(map.containsValue("k"));
        Assert.assertFalse(map.containsValue("f"));
        Assert.assertFalse(map.containsValue("h"));
        Assert.assertFalse(map.containsValue("i"));
    }
}
