package tsml.classifiers.distance_based.utils.collections;

import org.junit.Assert;
import org.junit.Test;

import java.util.ArrayList;

import static tsml.classifiers.distance_based.utils.collections.CollectionUtils.complement;
import static tsml.classifiers.distance_based.utils.collections.CollectionUtils.newArrayList;

public class CollectionUtilsTest {
    
    @Test
    public void testComplementOutOfOrder() {
        ArrayList<Integer> list = newArrayList(7,5,2,1);
        Assert.assertEquals(newArrayList(0,3,4,6,8,9), complement(10, list));
    }
    
    @Test
    public void testComplement() {
        ArrayList<Integer> list = newArrayList(1, 2, 5, 7);
        Assert.assertEquals(newArrayList(0,3,4,6,8,9), complement(10, list));
    }

    @Test
    public void testComplementEmpty() {
        ArrayList<Integer> list = newArrayList();
        Assert.assertEquals(newArrayList(0,1,2,3), complement(4, list));
    }
    
    @Test
    public void testComplementFull() {
        ArrayList<Integer> list = newArrayList(0,1,2,3);
        Assert.assertEquals(newArrayList(), complement(4, list));
    }

    @Test
    public void testComplementEdge() {
        ArrayList<Integer> list = newArrayList(0,3);
        Assert.assertEquals(newArrayList(1,2), complement(4, list));
    }
}
