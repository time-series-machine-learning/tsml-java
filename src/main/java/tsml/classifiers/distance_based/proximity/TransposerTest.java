package tsml.classifiers.distance_based.proximity;

import org.junit.Assert;
import org.junit.Test;
import tsml.data_containers.TimeSeries;
import tsml.data_containers.TimeSeriesInstance;

import java.util.Arrays;

public class TransposerTest {
    
    @Test
    public void test() {

        final Transposer t = new Transposer();
        final TimeSeries a = new TimeSeries(new double[]{1, 2, 3, 4, 5});
        final TimeSeries b = new TimeSeries(new double[]{6, 7, 8, 9, 10});
        final TimeSeriesInstance inst = new TimeSeriesInstance(1, Arrays.asList(a, b));
        final TimeSeriesInstance other = t.transform(inst);
        Assert.assertEquals(5, other.getNumDimensions());
        Assert.assertArrayEquals(new double[] {1,6}, other.get(0).toValueArray(), 0d);
        Assert.assertArrayEquals(new double[] {2,7}, other.get(1).toValueArray(), 0d);
        Assert.assertArrayEquals(new double[] {3,8}, other.get(2).toValueArray(), 0d);
        Assert.assertArrayEquals(new double[] {4,9}, other.get(3).toValueArray(), 0d);
        Assert.assertArrayEquals(new double[] {5,10}, other.get(4).toValueArray(), 0d);
    }
}
