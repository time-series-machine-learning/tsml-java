package tsml.classifiers.distance_based.utils.collections.intervals;

import org.junit.Assert;
import org.junit.Test;
import weka.core.DenseInstance;

public class IntervalInstanceTest {

    @Test
    public void testIndices() {
        final double[] attributes = new double[10];
        for(int i = 0; i < attributes.length; i++) {
            attributes[i] = i;
        }
        final DenseInstance instance = new DenseInstance(1, attributes);
        int start = 5;
        int length = 4;
        final IntervalInstance intervalInstance = new IntervalInstance(new Interval(start, length), instance);
        for(int i = 0; i < intervalInstance.numAttributes() - 1; i++) {
            final double intervalValue = intervalInstance.value(i);
            final double instanceValue = instance.value(i + start);
            Assert.assertEquals(intervalValue, instanceValue, 0);
        }
    }
}
