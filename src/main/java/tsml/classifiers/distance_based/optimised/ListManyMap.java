package tsml.classifiers.distance_based.optimised;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

public class ListManyMap<A extends Comparable<A>, B> extends AbstractManyMap<A, B, List<B>> {
    
    public ListManyMap() {
        super(ArrayList::new);
    }
    
    public ListManyMap(Comparator<A> comparator) {
        super(comparator, ArrayList::new);
    }

    @Override public List<B> get(final Object o) {
        return Collections.unmodifiableList(super.get(o));
    }
}
