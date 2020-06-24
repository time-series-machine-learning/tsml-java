package tsml.classifiers.distance_based.utils.collections.intervals;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

public class IntervalTest {

    private Interval interval;
    private int start;
    private int length;

    @Before
    public void before() {
        interval = new Interval();
        this.start = 50;
        this.length = 11;
        interval.setStart(start);
        interval.setLength(length);
    }

    @Test
    public void testIntervalSize() {
        Assert.assertEquals(length, interval.size());
    }

    @Test
    public void testTranslate() {
        for(int i = 0; i < length; i++) {
            final int index = interval.translate(i);
            Assert.assertEquals(i + start, index);
        }
    }

    @Test
    public void testInverseTranslate() {
        for(int i = 0; i < length; i++) {
            final int index = interval.inverseTranslate(i + start);
            Assert.assertEquals(i, index);
        }
    }

    @Test(expected = ArrayIndexOutOfBoundsException.class)
    public void testInverseTranslateOutOfBoundsAbove() {
        interval.inverseTranslate(start + length);
    }

    @Test(expected = ArrayIndexOutOfBoundsException.class)
    public void testInverseTranslateOutOfBoundsBelow() {
        interval.inverseTranslate(start - 1);
    }


    @Test(expected = ArrayIndexOutOfBoundsException.class)
    public void testTranslateOutOfBoundsAbove() {
        interval.translate(length);
    }

    @Test(expected = ArrayIndexOutOfBoundsException.class)
    public void testTranslateOutOfBoundsBelow() {
        interval.translate(-1);
    }
}
