package utilities.serialisation;

import java.io.Serializable;
import java.util.function.BiFunction;

public interface SerialisedBiFunction<A, B, C>
    extends BiFunction<A, B, C>,
            Serializable {
}
