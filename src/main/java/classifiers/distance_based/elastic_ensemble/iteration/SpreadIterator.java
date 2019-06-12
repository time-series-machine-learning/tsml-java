package classifiers.distance_based.elastic_ensemble.iteration;

import java.util.*;

public class SpreadIterator<A> implements Iterator<A> {
    protected final Queue<List<? extends A>> queue;
    private A value = null;

    public SpreadIterator(final List<? extends A> queue) {
        this.queue = new LinkedList<>();
        this.queue.add(queue);
    }

    @Override
    public void remove() {
        value = null;
    }

    @Override
    public boolean hasNext() {
        value = null;
        return !queue.isEmpty();
    }

    @Override
    public A next() {
        if(value == null) {
            List<? extends A> values = this.queue.remove();
            final int index = values.size() / 2;
            A value = values.get(index);
            int lowerIndex = index - 1;
            int upperIndex = index + 1;
            if(lowerIndex >= 0) {
                List<? extends A> lower = values.subList(0, lowerIndex);
                queue.add(lower);
            }
            if(upperIndex <= values.size() - 1) {
                List<? extends A> upper = values.subList(upperIndex, values.size() - 1);
                queue.add(upper);
            }
        }
        return value;
    }
}
