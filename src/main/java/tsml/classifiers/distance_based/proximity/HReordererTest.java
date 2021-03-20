package tsml.classifiers.distance_based.proximity;

import org.junit.Assert;
import org.junit.Test;
import tsml.data_containers.TimeSeries;
import tsml.data_containers.TimeSeriesInstance;

import java.util.Arrays;

public class HReordererTest {

    @Test
    public void test() {
        final HReorderer t = new HReorderer();
        final TimeSeries a = new TimeSeries(new double[]{1, 2, 3, 4, 5});
        final TimeSeries b = new TimeSeries(new double[]{6, 7, 8, 9, 10});
        t.setIndices(Arrays.asList(1, 0));
        final TimeSeriesInstance inst = new TimeSeriesInstance(1, Arrays.asList(a, b));
        final TimeSeriesInstance other = t.transform(inst);
        Assert.assertEquals(2, other.getNumDimensions());
        Assert.assertEquals(inst.get(0), other.get(1));
        Assert.assertEquals(inst.get(1), other.get(0));
    }
}
