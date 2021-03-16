package tsml.classifiers.distance_based.utils.collections.patience;

import org.junit.Assert;
import org.junit.Test;

public class PatienceTest {
    
    @Test
    public void testPatience() {
        final Patience patience = new Patience(4);
        Assert.assertEquals(4, patience.getWindowSize());
        Assert.assertFalse(patience.isExpired());

        Assert.assertEquals(0, patience.size());
        Assert.assertTrue(patience.add(4));
        Assert.assertFalse(patience.isExpired());
        Assert.assertEquals(1, patience.size());
        Assert.assertTrue(patience.add(5));
        Assert.assertFalse(patience.isExpired());
        Assert.assertEquals(2, patience.size());
        Assert.assertFalse(patience.add(3));
        Assert.assertFalse(patience.isExpired());
        Assert.assertEquals(3, patience.size());
        Assert.assertFalse(patience.add(1));
        Assert.assertFalse(patience.isExpired());
        Assert.assertEquals(4, patience.size());
        Assert.assertFalse(patience.add(2));
        Assert.assertFalse(patience.isExpired());
        Assert.assertEquals(5, patience.size());
        Assert.assertFalse(patience.add(2));
        Assert.assertTrue(patience.isExpired());
        Assert.assertEquals(6, patience.size());
        
        Assert.assertEquals(5, patience.getBest(), 0d);
        Assert.assertEquals(1, patience.getBestIndex());
        
        patience.resetPatience();
        Assert.assertEquals(6, patience.size());
        Assert.assertFalse(patience.add(2));
        Assert.assertFalse(patience.isExpired());
        Assert.assertEquals(7, patience.size());
        Assert.assertFalse(patience.add(4));
        Assert.assertFalse(patience.isExpired());
        Assert.assertEquals(8, patience.size());
        Assert.assertTrue(patience.add(6));
        Assert.assertFalse(patience.isExpired());
        Assert.assertEquals(9, patience.size());
        Assert.assertFalse(patience.add(1));
        Assert.assertFalse(patience.isExpired());
        Assert.assertEquals(10, patience.size());
        Assert.assertFalse(patience.add(2));
        Assert.assertFalse(patience.isExpired());
        Assert.assertEquals(11, patience.size());
        Assert.assertFalse(patience.add(4));
        Assert.assertFalse(patience.isExpired());
        Assert.assertEquals(12, patience.size());
        Assert.assertFalse(patience.add(2));
        Assert.assertTrue(patience.isExpired());
        Assert.assertEquals(13, patience.size());
        
        Assert.assertEquals(8, patience.getBestIndex());
        Assert.assertEquals(6, patience.getBest(), 0d);
        Assert.assertEquals(8, patience.getWindowStart());
        Assert.assertEquals(12, patience.getIndex());

        patience.setTolerance(2.5);
        Assert.assertFalse(patience.add(8.4));
        Assert.assertEquals(14, patience.size());
        Assert.assertTrue(patience.add(8.6));
        Assert.assertEquals(15, patience.size());
        
        Assert.assertFalse(patience.isEmpty());
        patience.reset();
        Assert.assertEquals(-1, patience.getIndex());
        Assert.assertEquals(0, patience.getWindowStart());
        Assert.assertEquals(4, patience.getWindowSize());
        Assert.assertEquals(-1, patience.getBestIndex());
        Assert.assertEquals(-1, patience.getBest(), 0);
        Assert.assertEquals(2.5, patience.getTolerance(), 0d);
        Assert.assertTrue(patience.isEmpty());
        Assert.assertFalse(patience.isExpired());
        
    }
}
