package utilities.serialisation;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.function.Function;

public interface SerialisedFunction<A, B> extends Function<A, B>,
                                                  Serializable {

}
