//package utilities.iteration.limited;
//
//import go.iteration.AbstractIterator;
//
//public class LimitedIterator<A>
//    extends AbstractIterator<A> {
//
//    private AbstractIterator<A> iterator;
//    private int limit;
//    private int count = 0;
//
//    public AbstractIterator<A> getIterator() {
//        return iterator;
//    }
//
//    public void setIterator(final AbstractIterator<A> iterator) {
//        this.iterator = iterator;
//    }
//
//    public void setCount(final int count) {
//        this.count = count;
//    }
//
//    public LimitedIterator() {}
//
//    public LimitedIterator(final AbstractIterator<A> iterator, final int limit) {this.iterator = iterator;
//        this.limit = limit;
//    }
//
//    public LimitedIterator(LimitedIterator<A> other) {
//        this(other.iterator, other.limit);
//        count = other.count;
//    }
//
//    @Override
//    public LimitedIterator<A> iterator() {
//        return new LimitedIterator<>(this);
//    }
//
//    @Override
//    public boolean hasNext() {
//        return iterator.hasNext() && (count < limit || limit < 0);
//    }
//
//    @Override
//    public A next() {
//        count++;
//        return iterator.next();
//    }
//
//    @Override
//    public void remove() {
//        iterator.remove();
//    }
//
//    @Override
//    public void add(final A a) {
//        iterator.add(a);
//    }
//
//    public void resetCount() {
//        count = 0;
//    }
//
//    public int getLimit() {
//        return limit;
//    }
//
//    public void setLimit(final int limit) {
//        this.limit = limit;
//    }
//
//    public int getCount() {
//        return count;
//    }
//
//}
