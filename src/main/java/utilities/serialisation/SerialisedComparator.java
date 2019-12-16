package utilities.serialisation;

import java.io.Serializable;
import java.util.Comparator;

public abstract class SerialisedComparator<A>
    implements Comparator<A>,
               Serializable {
}
