package utilities.serialisation;

import java.io.Serializable;
import java.util.function.Consumer;

public interface SerConsumer<A> extends Serializable, Consumer<A> {
}
