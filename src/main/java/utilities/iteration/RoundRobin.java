package utilities.iteration;

import java.util.List;

public class RoundRobin<A> extends LinearListIterator<A> {
    public RoundRobin(List<A> list) {
        super(list);
    }

    public RoundRobin() {}

    @Override
    public A next() {
        A next = super.next();
        if(index == list.size()) {
            index = 0;
        }
        return next;
    }

    @Override
    public void remove() {
        super.remove();
        if(index < 0) {
            index = list.size() - 1;
        }
    }

    @Override
    public int nextIndex() {
        return super.nextIndex() % list.size();
    }
}
