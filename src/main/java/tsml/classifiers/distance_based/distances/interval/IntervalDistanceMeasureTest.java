package tsml.classifiers.distance_based.distances.interval;

import experiments.data.DatasetLoading;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import tsml.classifiers.distance_based.distances.ed.EuclideanDistance;
import tsml.classifiers.distance_based.utils.intervals.Interval;
import weka.core.Instance;
import weka.core.Instances;

public class IntervalDistanceMeasureTest {

    private IntervalDistanceMeasure idf;

    @Before
    public void before() {
        idf = new IntervalDistanceMeasure();
    }

    @Test
    public void testGunPoint() throws Exception {
        final Instances data = DatasetLoading.loadGunPoint();
        final Instance a = data.get(0);
        final Instance b = data.get(data.size() - 1);
        idf.setDistanceFunction(new EuclideanDistance());
        idf.setInterval(new Interval(3, 7));
        double distance = idf.distance(a, b);
        Assert.assertEquals(2.0065039811768686, distance, 0);
        idf.setInterval(new Interval(5, 10));
        distance = idf.distance(a, b);
        Assert.assertEquals(2.3778470534877503, distance, 0);
    }
}
