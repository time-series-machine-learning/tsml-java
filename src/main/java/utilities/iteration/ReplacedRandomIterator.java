package utilities.iteration;

public class ReplacedRandomIterator<A> extends RandomIterator<A> {
    @Override public A next() {
//        A element = list.remove(nextIndex());
        A element = list.get(indices.get(nextIndex()));
        nextIndexSetup = false;
        return element;
    }

}
