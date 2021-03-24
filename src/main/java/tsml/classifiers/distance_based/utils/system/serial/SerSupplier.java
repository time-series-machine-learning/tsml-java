package tsml.classifiers.distance_based.utils.system.serial;

import java.io.Serializable;
import java.util.function.Supplier;

public interface SerSupplier<A> extends Serializable, Supplier<A> {
}
