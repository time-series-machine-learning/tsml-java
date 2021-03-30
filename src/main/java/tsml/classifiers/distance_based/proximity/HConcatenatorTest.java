package tsml.classifiers.distance_based.proximity;

import org.junit.Assert;
import org.junit.Test;
import tsml.data_containers.TimeSeries;
import tsml.data_containers.TimeSeriesInstance;

import java.util.Arrays;

public class HConcatenatorTest {

    @Test
    public void test() {
        final HConcatenator t = new HConcatenator();
        final TimeSeries a = new TimeSeries(new double[]{1, 2, 3, 4, 5});
        final TimeSeries b = new TimeSeries(new double[]{6, 7, 8, 9, 10});
        final TimeSeriesInstance inst = new TimeSeriesInstance(1, Arrays.asList(a, b));
        final TimeSeriesInstance other = t.transform(inst);
        Assert.assertEquals(1, other.getNumDimensions());
        Assert.assertArrayEquals(new double[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, other.get(0).toValueArray(), 0d);
    }
}
