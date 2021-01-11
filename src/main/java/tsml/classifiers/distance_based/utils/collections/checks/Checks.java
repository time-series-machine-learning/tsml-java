package tsml.classifiers.distance_based.utils.collections.checks;

import org.junit.Assert;

public class Checks {
    public static void assertReal(double number) {
        Assert.assertNotEquals(Double.NaN, number);
        Assert.assertNotEquals(Double.POSITIVE_INFINITY, number);
        Assert.assertNotEquals(Double.NEGATIVE_INFINITY, number);
    }
}
