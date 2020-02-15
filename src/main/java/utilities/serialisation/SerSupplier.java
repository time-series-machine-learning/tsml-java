package utilities.serialisation;

import org.checkerframework.checker.units.qual.A;

import java.io.Serializable;
import java.util.function.Supplier;

public interface SerSupplier<A> extends Supplier<A>, Serializable {
}
