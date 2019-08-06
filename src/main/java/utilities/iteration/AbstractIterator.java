package utilities.iteration;

import java.util.Collection;
import java.util.Iterator;

public abstract class AbstractIterator<A>
    implements Iterator<A>, Iterable<A> {

    public void addAll(Collection<A> collection) {
        for(A item : collection) {
            add(item);
        }
    }

    public abstract void add(A item);

    @Override
    public abstract AbstractIterator<A> iterator();
}
