package tsml.classifiers.distance_based.proximity;

import org.junit.Assert;
import org.junit.Test;
import tsml.data_containers.TimeSeries;
import tsml.data_containers.TimeSeriesInstance;

import java.util.Arrays;

public class HSlicerTest {
    
    @Test
    public void test() {
        final HSlicer hSlicer = new HSlicer();
        hSlicer.setIndices(Arrays.asList(1));
        final TimeSeries a = new TimeSeries(new double[]{1, 2, 3, 4, 5});
        final TimeSeries b = new TimeSeries(new double[]{6, 7, 8, 9, 10});
        final TimeSeriesInstance inst = new TimeSeriesInstance(1, Arrays.asList(a, b));
        final TimeSeriesInstance other = hSlicer.transform(inst);
        Assert.assertEquals(1, other.getNumDimensions());
        Assert.assertEquals(b, other.get(0));
    }
}
