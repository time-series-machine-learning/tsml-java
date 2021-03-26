package tsml.classifiers.distance_based.utils.collections.iteration;

import java.io.Serializable;
import java.util.Iterator;

public abstract class AbstractIterator<A> implements Iterator<A>, Serializable {
    private boolean hasNextCalled = false;
    private boolean hasNext = false;
    
    protected abstract boolean findHasNext();
    
    protected abstract A findNext();
    
    @Override public boolean hasNext() {
        if(hasNextCalled) {
            return hasNext;
        }
        hasNext = findHasNext();
        hasNextCalled = true;
        return hasNext;
    }

    @Override public A next() {
        if(!hasNext()) {
            throw new IllegalStateException("hasNext false");
        }
        hasNextCalled = false;
        return findNext();
    }
}
