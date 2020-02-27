package utilities.serialisation;

import java.io.Serializable;
import java.util.function.Supplier;

public interface SerSupplier<A> extends Supplier<A>, Serializable {
}
