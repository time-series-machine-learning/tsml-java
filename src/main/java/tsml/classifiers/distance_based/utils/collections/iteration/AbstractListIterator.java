package tsml.classifiers.distance_based.utils.collections.iteration;

import java.io.Serializable;
import java.util.List;
import java.util.Objects;

public abstract class AbstractListIterator<A> implements DefaultListIterator<A>, Serializable {

    private int index = -1;
    private int nextIndex = -1;
    private List<A> list;
    private boolean findNextIndex = true;
    private boolean hasNext;
    private boolean findHasNext = true;
    
    public AbstractListIterator(List<A> list) {
        buildIterator(list);
    }
    
    public AbstractListIterator() {}
    
    protected int getIndex() {
        return index;
    }
    
    protected void setIndex(int index) {
        this.index = index;
        findNextIndex = true;
        findHasNext = true;
    }
    
    @Override public A next() {
        setIndex(nextIndex());
        return list.get(index);
    }

    @Override final public int nextIndex() {
        if(!hasNext()) {
            throw new IllegalStateException("hasNext false");
        }
        if(findNextIndex) {
            nextIndex = findNextIndex();
            findNextIndex = false;
        }
        return nextIndex;
    }

    @Override final public boolean hasNext() {
        if(findHasNext) {
            if(list == null) throw new IllegalStateException("iterator has not been built, list is null");
            hasNext = findHasNext();
            findHasNext = false;
        }
        return hasNext;
    }
    
    abstract protected boolean findHasNext();

    abstract protected int findNextIndex();

    @Override public void remove() {
        list.remove(index);
    }

    @Override public void set(final A a) {
        list.set(index, a);
    }

    @Override public void add(final A a) {
        list.add(index, a);
    }

    public void buildIterator(final List<A> list) {
        this.list = Objects.requireNonNull(list);
    }
    
    protected List<A> getList() {
        return list;
    }

}
