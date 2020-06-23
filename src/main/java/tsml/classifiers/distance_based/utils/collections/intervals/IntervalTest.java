package tsml.classifiers.distance_based.utils.collections.intervals;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

public class IntervalTest {

    private Interval interval;

    @Before
    public void before() {
        interval = new Interval();
        interval.setStart(50);
        interval.setEnd(60);
    }

    private void reverseInterval() {
        final int end = interval.getEnd();
        interval.setEnd(interval.getStart());
        interval.setStart(end);
    }

    @Test
    public void testIntervalSize() {
        Assert.assertEquals(11, interval.size());
    }

    @Test
    public void testReversedIntervalSize() {
        reverseInterval();
        Assert.assertEquals(11, interval.size());
    }

    @Test
    public void testIntervalIndexToAttributeIndex() {
        for(int i = 0; i < 11; i++) {
            final int index = interval.translate(i);
            Assert.assertEquals(i + interval.getStart(), index);
        }
    }

    @Test
    public void testAttributeIndexToIntervalIndex() {
        for(int i = 0; i < 11; i++) {
            final int index = interval.inverseTranslate(i + interval.getStart());
            Assert.assertEquals(i, index);
        }
    }

    @Test
    public void testReversedIntervalIndexToAttributeIndex() {
        reverseInterval();
        for(int i = 0; i < 11; i++) {
            final int index = interval.translate(i);
            Assert.assertEquals(interval.getStart() - i, index);
        }
    }

    @Test
    public void testReversedAttributeIndexToIntervalIndex() {
        reverseInterval();
        for(int i = 0; i < 11; i++) {
            final int index = interval.inverseTranslate(interval.getStart() - i);
            Assert.assertEquals(i, index);
        }
    }
}
