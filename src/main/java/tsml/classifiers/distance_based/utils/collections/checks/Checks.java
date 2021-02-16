package tsml.classifiers.distance_based.utils.collections.checks;

import org.junit.Assert;

import java.util.Collection;
import java.util.List;
import java.util.Objects;

public class Checks {
    public static void assertReal(double number) {
        Assert.assertNotEquals(Double.NaN, number);
        Assert.assertNotEquals(Double.POSITIVE_INFINITY, number);
        Assert.assertNotEquals(Double.NEGATIVE_INFINITY, number);
    }
    
    public static <A> A requireNonNull(A obj) {
        Objects.requireNonNull(obj);
        if(obj instanceof Iterable<?>) {
            ((Iterable<?>) obj).forEach(Objects::requireNonNull);
        }
        return obj;
    }
    
    public static <A> A requireSingle(List<A> list) {
        if(list == null) {
            return null;
        }
        if(list.size() != 1) {
            throw new IllegalArgumentException("expected a single element in list");
        }
        return list.get(0);
    }
}
