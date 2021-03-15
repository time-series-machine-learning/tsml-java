package tsml.classifiers.distance_based.optimised;

import java.util.Collection;
import java.util.Map;
import java.util.Set;

public interface ManyMap<A extends Comparable<A>, B, C extends Collection<B>> extends Map<A, C> {
    void add(A key, B value);

    void addAll(A key, Iterable<B> values);

}
