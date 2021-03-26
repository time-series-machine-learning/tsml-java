package tsml.classifiers.distance_based.utils.collections.checks;

import java.util.Collection;
import java.util.List;
import java.util.Objects;

public class Checks {
    
    public static double requireUnitInterval(double v) {
        requireReal(v);
        requireNonNegative(v);
        if(v > 1) {
            throw new IllegalStateException(v + " > 1");
        }
        return v;
    }
    
    public static double requirePercentage(double v) {
        return requireUnitInterval(v / 100);
    }
    
    public static double requireNonNaN(double v) {
        if(Double.isNaN(v)) {
            throw new IllegalArgumentException("NaN not allowed");
        }
        return v;
    }
    
    public static double requireReal(double v) {
        requireNonNaN(v);
        if(v == Double.POSITIVE_INFINITY) {
            throw new IllegalArgumentException("Positive infinity not allowed");
        }
        if(v == Double.NEGATIVE_INFINITY) {
            throw new IllegalArgumentException("Negative infinity not allowed");
        }
        return v;
    }
    
    public static double requireNonNegative(double v) {
        if(v < 0) throw new IllegalArgumentException(v + " < 0");
        return v;
    }
    
    public static double requireNonPositive(double v) {
        if(v > 0) throw new IllegalArgumentException(v + " > 0");
        return v;
    }
    
    public static double requireNonZero(double v) {
        if(v == 0) throw new IllegalArgumentException(v + " == 0");
        return v;
    }
    public static double requireNegative(double v) {
        if(v >= 0) throw new IllegalArgumentException(v + " >= 0");
        return v;
    }
    
    public static double requirePositive(double v) {
        if(v <= 0) throw new IllegalArgumentException(v + " <= 0");
        return v;
    }

    public static int requireNonNegative(int v) {
        if(v < 0) throw new IllegalArgumentException(v + " < 0");
        return v;
    }

    public static int requireNonPositive(int v) {
        if(v > 0) throw new IllegalArgumentException(v + " > 0");
        return v;
    }

    public static int requireNonZero(int v) {
        if(v == 0) throw new IllegalArgumentException(v + " == 0");
        return v;
    }
    public static int requireNegative(int v) {
        if(v >= 0) throw new IllegalArgumentException(v + " >= 0");
        return v;
    }

    public static int requirePositive(int v) {
        if(v <= 0) throw new IllegalArgumentException(v + " <= 0");
        return v;
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
