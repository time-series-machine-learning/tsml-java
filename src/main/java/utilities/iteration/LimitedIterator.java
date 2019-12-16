package utilities.iteration;

import weka.core.OptionHandler;

import java.util.Enumeration;
import java.util.ListIterator;

public class LimitedIterator<A>
    implements DefaultListIterator<A>,
               OptionHandler {

    private int limit = -1;

    public int getLimit() {
        return limit;
    }

    public void setLimit(final int limit) {
        this.limit = limit;
    }

    private int count = 0;
    private ListIterator<A> iterator;

    public ListIterator<A> getIterator() {
        return iterator;
    }

    public void setIterator(final ListIterator<A> iterator) {
        this.iterator = iterator;
    }

    @Override
    public boolean hasNext() {
        return (count < limit || limit < 0) && iterator.hasNext();
    }

    @Override
    public A next() {
        count++;
        return iterator.next();
    }

    public void resetCount() {
        count = 0;
    }

    @Override
    public void add(final A a) {
        iterator.add(a);
    }

    @Override
    public void remove() {
        iterator.remove();
    }

    @Override
    public Enumeration listOptions() {
        throw new UnsupportedOperationException();
    }

    @Override
    public void setOptions(final String[] options) throws
                                                   Exception { // todo

    }

    @Override
    public String[] getOptions() {
        return new String[] {
            "-l",
            String.valueOf(limit)
        }; // todo
    }

    // todo pass through other iterator funcs
}
