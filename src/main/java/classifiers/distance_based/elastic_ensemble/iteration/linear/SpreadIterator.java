//package classifiers.distance_based.elastic_ensemble.iteration.linear;
//
//import classifiers.distance_based.elastic_ensemble.iteration.DynamicIterator;
//
//import java.util.*;
//
//public class SpreadIterator<A> extends DynamicIterator<A> {
//    protected final Queue<List<? extends A>> queue = new LinkedList<>();
//    private A value = null;
//
//    public SpreadIterator(final Collection<? extends A> queue) {
//        this.queue.add(new ArrayList<>(queue));
//    }
//
//    public SpreadIterator() {
//    }
//
//    public SpreadIterator(SpreadIterator<A> other) {
//        queue.addAll(other.queue);
//    }
//
//    @Override
//    public void remove() {
//        value = null;
//    }
//
//    @Override
//    public void add(final A a) {
//        throw new UnsupportedOperationException();
//    }
//
//    @Override
//    public boolean hasNext() {
//        value = null;
//        return !queue.isEmpty();
//    }
//
//    @Override
//    public A next() {
//        if(value == null) {
//            List<? extends A> values = this.queue.remove();
//            final int index = values.size() / 2;
//            A value = values.get(index);
//            int lowerIndex = index - 1;
//            int upperIndex = index + 1;
//            if(lowerIndex >= 0) {
//                List<? extends A> lower = values.subList(0, lowerIndex);
//                queue.add(lower);
//            }
//            if(upperIndex <= values.size() - 1) {
//                List<? extends A> upper = values.subList(upperIndex, values.size() - 1);
//                queue.add(upper);
//            }
//        }
//        return value;
//    }
//
//    @Override
//    public SpreadIterator<A> iterator() {
//        return new SpreadIterator<>(this);
//    }
//}
