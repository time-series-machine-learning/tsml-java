package tsml.classifiers.distance_based.proximity;

import org.junit.Assert;
import org.junit.Test;
import tsml.data_containers.TimeSeries;
import tsml.data_containers.TimeSeriesInstance;

import java.util.Arrays;

public class VSlicerTest {

    @Test
    public void test() {
        final VSlicer vSlicer = new VSlicer();
        vSlicer.setIndices(Arrays.asList(1, 3));
        final TimeSeries a = new TimeSeries(new double[]{1, 2, 3, 4, 5});
        final TimeSeries b = new TimeSeries(new double[]{6, 7, 8, 9, 10});
        final TimeSeriesInstance inst = new TimeSeriesInstance(1, Arrays.asList(a, b));
        final TimeSeriesInstance other = vSlicer.transform(inst);
        Assert.assertEquals(2, other.getNumDimensions());
        Assert.assertArrayEquals(new double[] {2,4}, other.get(0).toValueArray(), 0d);
        Assert.assertArrayEquals(new double[] {7,9}, other.get(1).toValueArray(), 0d);
    }
}
